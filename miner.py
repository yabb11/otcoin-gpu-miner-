#!/usr/bin/env python3
"""
OTCoin CuPy GPU Miner - RTX 4080 SUPER 优化版 (稳定版)
"""

import hashlib
import time
import requests
import cupy as cp
import numpy as np

# ============== 配置 ==============
API_URL = "http://76.13.192.203:8080"
WALLET = "1e85b2d09e2f2e8477e3a142c7ca5a6019f29847f"

# RTX 4080 SUPER 优化参数
BLOCKS = 4096
THREADS = 1024
BATCH_SIZE = BLOCKS * THREADS * 256

sha256_kernel = cp.RawKernel(r'''
extern "C" {

__constant__ unsigned int K[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
    0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
    0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
    0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
    0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
    0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

__device__ __forceinline__ unsigned int rotr(unsigned int x, int n) {
    return __funnelshift_r(x, x, n);
}

__device__ void sha256_transform(unsigned int* state, const unsigned char* block) {
    unsigned int W[64];
    unsigned int a, b, c, d, e, f, g, h;

    #pragma unroll
    for (int i = 0; i < 16; i++) {
        W[i] = ((unsigned int)block[i*4] << 24) |
               ((unsigned int)block[i*4+1] << 16) |
               ((unsigned int)block[i*4+2] << 8) |
               ((unsigned int)block[i*4+3]);
    }

    #pragma unroll
    for (int i = 16; i < 64; i++) {
        unsigned int s0 = rotr(W[i-15], 7) ^ rotr(W[i-15], 18) ^ (W[i-15] >> 3);
        unsigned int s1 = rotr(W[i-2], 17) ^ rotr(W[i-2], 19) ^ (W[i-2] >> 10);
        W[i] = W[i-16] + s0 + W[i-7] + s1;
    }

    a = state[0]; b = state[1]; c = state[2]; d = state[3];
    e = state[4]; f = state[5]; g = state[6]; h = state[7];

    #pragma unroll
    for (int i = 0; i < 64; i++) {
        unsigned int S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
        unsigned int ch = (e & f) ^ ((~e) & g);
        unsigned int temp1 = h + S1 + ch + K[i] + W[i];
        unsigned int S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
        unsigned int maj = (a & b) ^ (a & c) ^ (b & c);
        unsigned int temp2 = S0 + maj;

        h = g; g = f; f = e; e = d + temp1;
        d = c; c = b; b = a; a = temp1 + temp2;
    }

    state[0] += a; state[1] += b; state[2] += c; state[3] += d;
    state[4] += e; state[5] += f; state[6] += g; state[7] += h;
}

__device__ void sha256(const unsigned char* msg, int len, unsigned int* hash) {
    unsigned int state[8] = {
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
    };

    unsigned char block[128];
    int padded_len = ((len + 9 + 63) / 64) * 64;

    for (int i = 0; i < padded_len; i++) {
        if (i < len) block[i % 64] = msg[i];
        else if (i == len) block[i % 64] = 0x80;
        else if (i < padded_len - 8) block[i % 64] = 0;
        else {
            int shift = (padded_len - 1 - i) * 8;
            block[i % 64] = ((unsigned long long)len * 8 >> shift) & 0xff;
        }

        if ((i + 1) % 64 == 0) {
            sha256_transform(state, block);
        }
    }

    for (int i = 0; i < 8; i++) hash[i] = state[i];
}

__device__ int uint_to_str(unsigned long long n, unsigned char* buf) {
    if (n == 0) { buf[0] = '0'; return 1; }

    int len = 0;
    unsigned long long temp = n;
    while (temp > 0) { len++; temp /= 10; }

    for (int i = len - 1; i >= 0; i--) {
        buf[i] = '0' + (n % 10);
        n /= 10;
    }
    return len;
}

__global__ void mine_kernel(
    const unsigned char* prefix,
    int prefix_len,
    unsigned long long nonce_start,
    int hashes_per_thread,
    unsigned long long* result_nonce,
    int* found
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long base_nonce = nonce_start + (unsigned long long)idx * hashes_per_thread;

    unsigned char msg[200];
    unsigned int hash[8];

    for (int i = 0; i < prefix_len; i++) {
        msg[i] = prefix[i];
    }

    for (int h = 0; h < hashes_per_thread; h++) {
        if (*found) return;

        unsigned long long nonce = base_nonce + h;

        unsigned char nonce_buf[20];
        int nonce_len = uint_to_str(nonce, nonce_buf);

        int msg_len = prefix_len;
        for (int i = 0; i < nonce_len; i++) {
            msg[msg_len++] = nonce_buf[i];
        }

        sha256(msg, msg_len, hash);

        if (hash[0] == 0 && (hash[1] >> 24) == 0) {
            int old = atomicCAS(found, 0, 1);
            if (old == 0) {
                *result_nonce = nonce;
            }
            return;
        }
    }
}

}
''', 'mine_kernel')


def safe_request(method, url, **kwargs):
    """带重试的请求"""
    for attempt in range(3):
        try:
            if method == 'get':
                return requests.get(url, timeout=10, **kwargs)
            else:
                return requests.post(url, timeout=15, **kwargs)
        except Exception as e:
            if attempt < 2:
                time.sleep(1)
            else:
                raise e
    return None


def main():
    print(f"\n{'=' * 60}")
    print(f"  OTCoin GPU 挖矿 - RTX 4080 SUPER (稳定版)")
    print(f"  钱包: {WALLET}")
    print(f"  批量: {BATCH_SIZE:,} hashes/轮")
    print(f"{'=' * 60}\n")

    # GPU 内存预分配
    prefix_gpu = cp.zeros(200, dtype=cp.uint8)
    result_nonce = cp.zeros(1, dtype=cp.uint64)
    found = cp.zeros(1, dtype=cp.int32)

    total_hashes = 0
    blocks_found = 0
    start_time = time.time()

    while True:
        # 获取工作
        try:
            work = safe_request('get', f"{API_URL}/api/getwork").json()
        except Exception as e:
            print(f"\n⚠️ 连接失败: {e}")
            print("  5秒后重试...")
            time.sleep(5)
            continue

        block_index = work['block_index']
        prev_hash = work['previous_hash']

        print(f"\n⛏️ 挖掘区块 #{block_index}")

        gpu_nonce = 0

        while True:
            timestamp = time.time()
            prefix = f"{block_index}{timestamp}{prev_hash}"
            prefix_bytes = np.frombuffer(prefix.encode(), dtype=np.uint8)
            prefix_len = len(prefix_bytes)

            prefix_gpu[:prefix_len] = cp.asarray(prefix_bytes)
            found[0] = 0

            hashes_per_thread = 64
            sha256_kernel(
                (BLOCKS,), (THREADS,),
                (prefix_gpu, prefix_len, gpu_nonce, hashes_per_thread,
                 result_nonce, found)
            )
            cp.cuda.Stream.null.synchronize()

            total_hashes += BATCH_SIZE
            gpu_nonce += BATCH_SIZE

            elapsed = time.time() - start_time
            hashrate = total_hashes / elapsed
            print(f"\r  Nonces: {gpu_nonce:,} | {hashrate / 1e6:.1f} MH/s | 已挖: {blocks_found} 区块", end="",
                  flush=True)

            if found[0] == 1:
                nonce = int(result_nonce[0])
                verify_hash = hashlib.sha256(f"{block_index}{timestamp}{prev_hash}{nonce}".encode()).hexdigest()

                print(f"\n\n🎉 GPU 找到!")
                print(f"  Nonce: {nonce:,}")
                print(f"  Hash: {verify_hash}")

                # 提交（带重试）
                for attempt in range(5):
                    try:
                        result = requests.post(f"{API_URL}/api/mine", json={
                            "miner": WALLET,
                            "index": block_index,
                            "hash": verify_hash,
                            "previous_hash": prev_hash,
                            "nonce": nonce,
                            "difficulty": 10,
                            "timestamp": timestamp,
                            "transactions": []
                        }, timeout=30).json()

                        print(f"  结果: {result}")

                        if result.get('status') == 'ok':
                            blocks_found += 1
                            print(f"\n✅ 区块确认! 总计: {blocks_found} 区块 = {blocks_found * 50} OTC")
                        elif result.get('status') == 'duplicate':
                            print(f"  ⚠️ 区块已被其他人提交")
                        break
                    except Exception as e:
                        print(f"  ⚠️ 提交失败 (尝试 {attempt + 1}/5): {e}")
                        if attempt < 4:
                            time.sleep(2)
                        else:
                            print("  ❌ 放弃此区块，继续挖矿...")
                break

            # 检查新区块
            if gpu_nonce % (BATCH_SIZE * 10) == 0:
                try:
                    new_work = requests.get(f"{API_URL}/api/getwork", timeout=3).json()
                    if new_work['block_index'] != block_index:
                        print(f"\n📢 新区块 #{new_work['block_index']}，切换...")
                        break
                except:
                    pass


if __name__ == "__main__":
    main()