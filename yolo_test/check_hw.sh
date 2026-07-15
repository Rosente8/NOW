#!/bin/bash
# ============================================================
# 训练环境硬件检查（只检查，不安装任何东西）
# ============================================================

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

pass() { echo -e "${GREEN}[✓]${NC} $1"; }
fail() { echo -e "${RED}[✗]${NC} $1"; }
warn() { echo -e "${YELLOW}[!]${NC} $1"; }
info() { echo -e "${BLUE}[i]${NC} $1"; }
title() { echo -e "\n${BLUE}--- $1 ---${NC}"; }

ERRORS=0
WARNS=0

# ============================================================
title "1. 操作系统和 Python 版本"

# 系统信息
OS=$(lsb_release -d 2>/dev/null | cut -f2 || echo "未知")
info "操作系统: $OS"

# Python 版本
PYTHON_VER=$(python3 --version 2>/dev/null | awk '{print $2}')
if [ -n "$PYTHON_VER" ]; then
    pass "Python 版本: $PYTHON_VER"
    # 检查是否是 3.8+
    PY_MAJOR=$(echo $PYTHON_VER | cut -d'.' -f1)
    PY_MINOR=$(echo $PYTHON_VER | cut -d'.' -f2)
    if [ "$PY_MAJOR" -ge 3 ] && [ "$PY_MINOR" -ge 8 ]; then
        pass "Python 版本符合要求 (>=3.8)"
    else
        warn "Python 版本低于 3.8，YOLOv5 需要 3.8+"
        ((WARNS++))
    fi
else
    fail "Python3 未安装"
    ((ERRORS++))
fi

# Python 路径
PYTHON_PATH=$(which python3 2>/dev/null || echo "未找到")
info "Python 路径: $PYTHON_PATH"

# ============================================================
title "2. NVIDIA 驱动和 GPU"

if command -v nvidia-smi &> /dev/null; then
    pass "NVIDIA 驱动已安装"
    
    # GPU 名称
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)
    info "GPU: $GPU_NAME"
    
    # 驱动版本
    DRIVER_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n1)
    pass "驱动版本: $DRIVER_VER"
    
    # 系统 CUDA 版本（驱动支持的最高版本）
    SYS_CUDA=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}')
    if [ -n "$SYS_CUDA" ]; then
        pass "系统 CUDA 版本（驱动支持）: $SYS_CUDA"
    else
        warn "无法读取 CUDA 版本"
        ((WARNS++))
    fi
    
    # 显存大小
    MEM_TOTAL=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -n1)
    info "显存总量: $MEM_TOTAL"
    
    # 当前显存使用
    MEM_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader | head -n1)
    info "当前显存使用: $MEM_USED"
else
    fail "NVIDIA 驱动未安装或 nvidia-smi 不可用"
    info "请安装 NVIDIA 驱动: sudo apt install nvidia-driver-535"
    ((ERRORS++))
fi

# ============================================================
title "3. PyTorch 检查（使用系统 Python 3.10）"

# 检查 PyTorch 是否安装
if python3 -c "import torch" 2>/dev/null; then
    pass "PyTorch 已安装"
    
    # PyTorch 版本
    TORCH_VER=$(python3 -c "import torch; print(torch.__version__)" 2>/dev/null)
    pass "PyTorch 版本: $TORCH_VER"
    
    # PyTorch CUDA 版本（编译时使用的 CUDA）
    TORCH_CUDA=$(python3 -c "import torch; print(torch.version.cuda)" 2>/dev/null)
    if [ -n "$TORCH_CUDA" ]; then
        pass "PyTorch CUDA 版本: $TORCH_CUDA"
    else
        fail "PyTorch 为 CPU 版本（无 CUDA 支持）"
        info "请安装 GPU 版 PyTorch"
        ((ERRORS++))
    fi
    
    # CUDA 是否可用
    CUDA_AVAIL=$(python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null)
    if [ "$CUDA_AVAIL" = "True" ]; then
        pass "CUDA 可用 ✓"
        GPU_DETECTED=$(python3 -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null)
        info "PyTorch 检测到的 GPU: $GPU_DETECTED"
    else
        fail "CUDA 不可用 (torch.cuda.is_available() = False)"
        info "PyTorch GPU 版本未正确安装或与驱动不匹配"
        ((ERRORS++))
    fi
else
    fail "PyTorch 未安装"
    info "安装命令: pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
    ((ERRORS++))
fi

# ============================================================
title "4. 版本匹配检查（关键）"

if [ -n "$SYS_CUDA" ] && [ -n "$TORCH_CUDA" ] && [ "$CUDA_AVAIL" = "True" ]; then
    # 提取主版本号比较
    SYS_MAJOR=$(echo "$SYS_CUDA" | cut -d'.' -f1)
    SYS_MINOR=$(echo "$SYS_CUDA" | cut -d'.' -f2)
    TORCH_MAJOR=$(echo "$TORCH_CUDA" | cut -d'.' -f1)
    TORCH_MINOR=$(echo "$TORCH_CUDA" | cut -d'.' -f2)
    
    SYS_NUM=$((SYS_MAJOR * 10 + SYS_MINOR))
    TORCH_NUM=$((TORCH_MAJOR * 10 + TORCH_MINOR))
    
    if [ "$TORCH_NUM" -le "$SYS_NUM" ]; then
        pass "版本匹配: PyTorch CUDA $TORCH_CUDA ≤ 系统 CUDA $SYS_CUDA ✓"
    else
        fail "版本不匹配: PyTorch CUDA $TORCH_CUDA > 系统 CUDA $SYS_CUDA"
        info "需要降级 PyTorch 或升级 NVIDIA 驱动"
        info "建议安装: pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
        ((ERRORS++))
    fi
fi

# ============================================================
title "5. 其他必要软件"

# Git
if command -v git &> /dev/null; then
    pass "Git 已安装: $(git --version)"
else
    fail "Git 未安装"
    info "安装: sudo apt install git"
    ((ERRORS++))
fi

# pip
if command -v pip3 &> /dev/null; then
    pass "pip3 已安装"
else
    fail "pip3 未安装"
    info "安装: sudo apt install python3-pip"
    ((ERRORS++))
fi

# ============================================================
title "6. YOLOv5 仓库和数据集（可选检查）"

# 检查 YOLOv5 仓库
YOLOV5_DIR="$HOME/yolov5"
if [ -d "$YOLOV5_DIR" ] && [ -f "$YOLOV5_DIR/train.py" ]; then
    pass "YOLOv5 仓库存在: $YOLOV5_DIR"
else
    warn "YOLOv5 仓库不存在"
    info "克隆: git clone https://github.com/ultralytics/yolov5.git $YOLOV5_DIR"
    ((WARNS++))
fi

# 检查数据集（需用户配置）
DATASET_PATH="/home/your_username/datasets/drop_zone"
if [ -f "$DATASET_PATH/data.yaml" ]; then
    pass "data.yaml 存在: $DATASET_PATH"
    IMG_COUNT=$(find "$DATASET_PATH/images/train" -type f 2>/dev/null | wc -l)
    info "训练集图片: $IMG_COUNT 张"
elif [ -d "$DATASET_PATH" ]; then
    warn "data.yaml 不存在: $DATASET_PATH/data.yaml"
    ((WARNS++))
else
    warn "数据集路径不存在: $DATASET_PATH（请修改脚本中的路径）"
    info "如果数据集在其他位置，请忽略此警告"
fi

# ============================================================
title "检查完成"

echo ""
echo "📊 汇总:"
echo "   ❌ 错误: $ERRORS 项（必须修复才能训练）"
echo "   ⚠️ 警告: $WARNS 项（建议处理）"

if [ $ERRORS -eq 0 ]; then
    echo ""
    echo "🎉 硬件和 PyTorch 环境检查通过！可以开始训练。"
    echo ""
    echo "下一步："
    echo "  1. 克隆 YOLOv5: git clone https://github.com/ultralytics/yolov5.git"
    echo "  2. 安装依赖: pip3 install -r requirements.txt"
    echo "  3. 准备数据集（images/train, labels/train, data.yaml）"
    echo "  4. 开始训练: python3 train.py --img 416 --batch 16 --epochs 100 --data data.yaml --weights yolov5s.pt"
else
    echo ""
    echo "❌ 发现 $ERRORS 项错误，请先解决后再训练。"
    echo ""
    echo "常见解决方案："
    echo "  - NVIDIA 驱动: sudo apt install nvidia-driver-535"
    echo "  - PyTorch GPU: pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
    echo "  - 版本不匹配: 降级 PyTorch 或升级驱动"
fi








这个是对drop_zone进行训练
使用方法——————————————————————————————————————————————————————————————
# 1. 创建脚本
nano check_hw.sh
# 粘贴上面的内容，保存退出

# 2. 修改数据集路径（第 140 行附近）
# 将 /home/your_username/datasets/drop_zone 改成你的实际数据集路径
# 如果还没有数据集，可以暂时不改，会显示警告但不影响检查

# 3. 赋予执行权限
chmod +x check_hw.sh

# 4. 运行
./check_hw.sh



————————————————————————————————————————————————————————————————————
检查通过后的下一步操作
如果检查全部通过（错误数为 0），按以下顺序执行
# 1. 克隆 YOLOv5
git clone https://github.com/ultralytics/yolov5.git
cd yolov5

# 2. 安装依赖
pip3 install -r requirements.txt

# 3. 准备数据集
# 确保你的数据集在 /home/your_username/datasets/drop_zone/ 目录下，，，大概这个样子

/home/your_username/datasets/drop_zone/
├── images/
│   ├── train/          # 训练图片（至少 200 张，越多越好）
│   └── val/            # 验证图片（占总图片 20-30%）
├── labels/
│   ├── train/          # 训练标签（与图片同名 .txt）
│   └── val/            # 验证标签
└── data.yaml           # 配置文件（见下方）


cd /home/your_username/datasets/drop_zone/
nano data.yaml

粘贴以下内容

path: /home/your_username/datasets/drop_zone
train: images/train
val: images/val
nc: 2
names: ['circle', 'stuffed']

# 4. 开始训练
python3 train.py \
    --img 416 \              # 训练时将图片缩放到 416×416
    --batch 8 \              # 每批处理 8 张图片############如果显存不足，可以调小 batch 4
    --epochs 150 \           # 总共训练 150 轮
    --data /home/your_username/datasets/drop_zone/data.yaml \   # 数据集配置文件
    --weights yolov5m.pt \   # 使用 YOLOv5m 预训练权重
    --device 0 \             # 使用第一块 GPU
    --cache \                # 将图片缓存到内存中加速训练
    --patience 50 \          # 50 轮 mAP 不提升则提前停止训练
    --cos-lr \               # 使用余弦退火学习率调度器
    --mixup 0.5 \            # mixup 数据增强系数（0.5 表示一半图片做 mixup）
    --workers 4              # 使用 4 个线程加载数据





    ————————————————————————————————————————————————————————————
    训练完成后，模型保存在：ls -l ~/yolov5/runs/train/exp/weights/










222222——————————————————————————
训练H时
# 1. 准备数据集
# 确保你的数据集在 /home/your_username/datasets/h_zone/ 目录下，，，大概这个样子

/home/your_username/datasets/h_zone/
├── images/
│   ├── train/          # 训练图片（至少 200 张，越多越好）
│   └── val/            # 验证图片（占总图片 20-30%）
├── labels/
│   ├── train/          # 训练标签（与图片同名 .txt）
│   └── val/            # 验证标签
└── data.yaml           # 配置文件（见下方）

cd /home/your_username/datasets/h_zone/
nano data.yaml

粘贴以下内容

path: /home/your_username/datasets/h_zone
train: images/train
val: images/val
nc: 1
names: ['H']

# 2. 开始训练
python3 train.py \
    --img 416 \
    --batch 8 \
    --epochs 150 \
    --data /home/your_username/datasets/h_zone/data.yaml \
    --weights yolov5s.pt \
    --device 0 \
    --cache \
    --patience 50 \
    --cos-lr \
    --workers 4 \
    