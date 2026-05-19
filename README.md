# OTCoin CuPy GPU Miner

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/CUDA-11.0+-green.svg" alt="CUDA">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License">
  <img src="https://img.shields.io/badge/GPU-NVIDIA-76B900.svg" alt="GPU">
</p>

一款基于 CuPy 开发的高性能 OTCoin 加密货币 GPU 挖矿程序，支持所有 NVIDIA CUDA 显卡。

## ✨ 特性

- **🚀 CUDA 原生加速** - 使用 CuPy RawKernel 实现的高性能 SHA-256 哈希计算，完全在 GPU 上运行
- **🎮 广泛兼容** - 支持 GTX 9 系列到 RTX 50 系列的所有 NVIDIA CUDA 显卡
- **⚡ 高吞吐量** - 根据显卡配置优化批处理大小，最大化 GPU 利用率
- **🛡️ 稳定可靠** - 内置智能连接重试机制和完善的错误处理
- **📊 实时监控** - 实时显示算力（MH/s）、nonce 进度和挖矿统计信息
- **🔧 易于配置** - 提供详细的显卡配置指南，轻松调优

---

## 📋 系统要求

| 组件 | 最低要求 | 推荐配置 |
|------|----------|----------|
| 操作系统 | Windows 10 / Linux | Ubuntu 22.04 LTS / Windows 11 |
| Python | 3.8+ | 3.10+ |
| GPU | NVIDIA GTX 950 (2GB) | RTX 3060 或更高 |
| CUDA | 11.0+ | 12.0+ |
| 显存 | 2GB | 8GB+ |
| 内存 | 8GB | 16GB |
| 驱动 | 450.0+ | 最新版本 |

---

## 🔧 安装指南

### 步骤 1：安装 NVIDIA 驱动和 CUDA

#### Windows 用户

1. 访问 [NVIDIA 驱动下载页面](https://www.nvidia.cn/Download/index.aspx)
2. 选择你的显卡型号，下载并安装最新驱动
3. 下载 [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)
4. 安装时选择"自定义安装"，勾选 CUDA 组件

#### Linux 用户 (Ubuntu/Debian)

```bash
# 安装 NVIDIA 驱动
sudo apt update
sudo apt install nvidia-driver-535

# 安装 CUDA Toolkit
wget https://developer.download.nvidia.com/compute/cuda/12.2.0/local_installers/cuda_12.2.0_535.54.03_linux.run
sudo sh cuda_12.2.0_535.54.03_linux.run

# 添加环境变量
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# 验证安装
nvidia-smi
nvcc --version

步骤 2：克隆仓库

git clone https://github.com/你的用户名/otcoin-gpu-miner.git
cd otcoin-gpu-miner

步骤 3：创建虚拟环境（推荐）

python -m venv venv

# Windows PowerShell
.\venv\Scripts\Activate.ps1

# Windows CMD
venv\Scripts\activate.bat

# Linux / macOS
source venv/bin/activate

步骤 4：安装 Python 依赖

# 首先查看你的 CUDA 版本
nvidia-smi
# 输出右上角会显示 CUDA Version: 12.x 或 11.x

# CUDA 12.x 用户
pip install cupy-cuda12x numpy requests

# CUDA 11.x 用户
pip install cupy-cuda11x numpy requests

# 或直接使用 requirements.txt（默认 CUDA 12.x）
pip install -r requirements.txt

步骤 5：验证安装

python -c "
import cupy as cp
print(f'✅ CuPy 版本: {cp.__version__}')
print(f'✅ GPU: {cp.cuda.runtime.getDeviceProperties(0)[\"name\"].decode()}')
print(f'✅ 显存: {cp.cuda.runtime.getDeviceProperties(0)[\"totalGlobalMem\"] / 1024**3:.1f} GB')
"


⚙️ 显卡配置详解
配置参数说明
编辑 miner.py 文件顶部的配置：


# ============== 配置 ==============
API_URL = "http://76.13.192.203:8080"  # OTCoin 节点 API 地址
WALLET = "你的钱包地址"                   # 替换为你的钱包地址

# GPU 挖矿参数
BLOCKS = 4096      # CUDA Grid 大小（block 数量）
THREADS = 1024     # 每个 Block 的线程数
BATCH_SIZE = BLOCKS * THREADS * 256  # 每轮处理的哈希数


参数详解


参数	说明	影响
BLOCKS	GPU 并行执行的 block 数量	增大可提高并行度，但受 GPU SM 数量限制
THREADS	每个 block 中的线程数	最大 1024，受寄存器和共享内存限制
BATCH_SIZE	每轮 GPU 计算的总哈希数	BLOCKS × THREADS × 256

如何确定你的显卡参数
运行以下命令查看显卡规格：


python -c "
import cupy as cp

props = cp.cuda.runtime.getDeviceProperties(0)
print('='*50)
print('显卡信息')
print('='*50)
print(f'显卡名称: {props[\"name\"].decode()}')
print(f'显存大小: {props[\"totalGlobalMem\"] / 1024**3:.1f} GB')
print(f'SM 数量: {props[\"multiProcessorCount\"]}')
print(f'每 Block 最大线程数: {props[\"maxThreadsPerBlock\"]}')
print(f'计算能力: {props[\"major\"]}.{props[\"minor\"]}')
print('='*50)
sm_count = props['multiProcessorCount']
print(f'推荐 BLOCKS = {sm_count * 32}')
print(f'推荐 THREADS = 512')
print('='*50)
"

📊 手动调优步骤
步骤 1：使用保守配置启动

BLOCKS = 1024
THREADS = 256

步骤 2：逐步增加 THREADS
测试序列：256 → 512 → 768 → 1024

每次运行 2-3 分钟，记录算力，选择最高值。

# 公式：BLOCKS = SM数量 × 倍数
# 从 16 开始，逐步增加到 64
# 例如显卡有 68 个 SM
BLOCKS = 68 * 32  # 测试 68*16, 68*32, 68*48, 68*64


# 实时监控
watch -n 1 nvidia-smi

# 或使用 nvtop（Linux）
sudo apt install nvtop && nvtop

🚀 使用方法
启动挖矿

python miner.py

运行输出示例

============================================================
  OTCoin GPU 挖矿 - RTX 4080 SUPER (稳定版)
  钱包: 1e85b2d09e2f2e8477e3a142c7ca5a6019f29847f
  批量: 1,342,177,280 hashes/轮
============================================================

⛏️ 挖掘区块 #12345
  Nonces: 5,368,709,120 | 178.5 MH/s | 已挖: 2 区块

🎉 GPU 找到!
  Nonce: 5,234,567,890
  Hash: 000000003a7b8c9d...
  结果: {'status': 'ok'}

✅ 区块确认! 总计: 3 区块 = 150 OTC




















