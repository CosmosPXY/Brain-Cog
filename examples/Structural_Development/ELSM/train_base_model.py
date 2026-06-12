"""
train_base_model.py - 构建 970.t7 预训练权重文件

这个脚本在 MNIST 上训练基础的 mSNN 模型，生成用于 ELSM 进化的初始权重。

步骤：
1. 初始化 mSNN 模型
2. 在 MNIST 上训练
3. 生成随机连接矩阵
4. 保存权重为 970.t7
"""

from __future__ import print_function
import torchvision
import torchvision.transforms as transforms
import os
import time
import numpy as np
import torch
from torch import nn as nn
from mnistmodel import SNN
from tqdm import tqdm
from datetime import datetime
import logging
from braincog.base.utils import UnilateralMse
from braincog.base.learningrule.STDP import MutliInputSTDP
from braincog.base.node.node import LIFNode

# ===== 配置参数 =====
DEVICE = 'cuda:0'  # 修改为您的 GPU 设备
BATCH_SIZE = 100
LIQUID_SIZE = 8000
LEARNING_RATE = 1e-3
NUM_EPOCHS = 50  # 可以调整，更多 epoch 获得更好的初始权重
DATA_PATH = './data'  # 修改为您的 MNIST 数据路径
SAVE_PATH = './970.t7'

# 连接矩阵稀疏度（0.01 = 1% 连接）
CONNECTIVITY_SPARSITY = 0.01

# ========================

def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

logger = setup_logging()

def randbool(size, p=0.5):
    """生成随机二值矩阵"""
    return torch.rand(*size) < p

def lr_scheduler(optimizer, epoch, init_lr=0.001, lr_decay_epoch=10):
    """学习率衰减"""
    if epoch % lr_decay_epoch == 0 and epoch > 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.5
            logger.info(f"学习率降低到: {param_group['lr']}")
    return optimizer

def build_model(device):
    """构建模型"""
    logger.info("=" * 60)
    logger.info("构建 mSNN 模型")
    logger.info("=" * 60)
    
    # 生成随机连接矩阵
    connectivity_matrix = randbool(
        [LIQUID_SIZE, LIQUID_SIZE], 
        p=CONNECTIVITY_SPARSITY
    ).to(device).int()
    
    logger.info(f"连接矩阵形状: {connectivity_matrix.shape}")
    logger.info(f"非零连接数: {connectivity_matrix.sum().item()}")
    logger.info(f"稀疏度: {CONNECTIVITY_SPARSITY}")
    
    # 创建模型
    snn = SNN(
        ins=784,
        batchsize=BATCH_SIZE,
        device=device,
        liquid_size=LIQUID_SIZE,
        connectivity_matrix=connectivity_matrix,
        lsm_tau=3.0,
        lsm_th=0.3,
        fc_tau=3.0,
        fc_th=0.3,
        num_classes=10
    )
    
    snn.to(device)
    snn.eval()
    
    logger.info(f"模型创建完成")
    return snn, connectivity_matrix

def load_data(data_path):
    """加载 MNIST 数据集"""
    logger.info("=" * 60)
    logger.info("加载 MNIST 数据集")
    logger.info("=" * 60)
    
    # 确保数据目录存在
    os.makedirs(data_path, exist_ok=True)
    
    # 加载训练集
    train_dataset = torchvision.datasets.MNIST(
        root=data_path, 
        train=True, 
        download=True,  # 如果本地没有数据，自动下载
        transform=transforms.ToTensor()
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=4
    )
    
    # 加载测试集
    test_set = torchvision.datasets.MNIST(
        root=data_path, 
        train=False, 
        download=True,
        transform=transforms.ToTensor()
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=2
    )
    
    logger.info(f"训练集大小: {len(train_dataset)}")
    logger.info(f"测试集大小: {len(test_set)}")
    
    return train_loader, test_loader

def train_epoch(model, train_loader, optimizer, criterion, device, epoch):
    """训练一个 epoch"""
    model.train()
    total_loss = 0
    
    for batch_idx, (images, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
        images = images.float().to(device)
        labels = labels.to(device)
        
        # 前向传播
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        model.reset()
        
        total_loss += loss.item()
        
        if (batch_idx + 1) % 100 == 0:
            logger.info(f"  Batch [{batch_idx+1}/{len(train_loader)}] Loss: {loss.item():.6f}")
    
    avg_loss = total_loss / len(train_loader)
    logger.info(f"Epoch {epoch+1} 平均损失: {avg_loss:.6f}")
    return avg_loss

def evaluate(model, test_loader, device):
    """评估模型"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images = images.float().to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            model.reset()
    
    accuracy = 100 * correct / total
    logger.info(f"测试精度: {accuracy:.2f}%")
    return accuracy

def save_checkpoint(model, connectivity_matrix, device, save_path):
    """保存模型检查点"""
    logger.info("=" * 60)
    logger.info(f"保存模型检查点到: {save_path}")
    logger.info("=" * 60)
    
    # 提取权重
    checkpoint = {
        'fc': model.fc.state_dict(),  # 输出层权重
        'lsm0': model.con[0].state_dict(),  # 输入到液体层权重
        'connectivity_matrix': connectivity_matrix.cpu(),  # 连接矩阵
        'liquid_weight': model.con[1].weight.data.cpu(),  # 液体层内部权重
    }
    
    # 保存为 .t7 格式（PyTorch torch 格式）
    torch.save(checkpoint, save_path)
    logger.info(f"✓ 模型已保存")
    
    # 打印检查点信息
    logger.info("检查点内容:")
    for key, value in checkpoint.items():
        if isinstance(value, torch.Tensor):
            logger.info(f"  - {key}: {value.shape}")
        elif isinstance(value, dict):
            logger.info(f"  - {key}: 状态字典，包含 {len(value)} 个张量")

def main():
    logger.info("\n" + "=" * 60)
    logger.info("ELSM 基础模型预训练")
    logger.info("=" * 60 + "\n")
    
    # 检查 CUDA
    if torch.cuda.is_available():
        logger.info(f"✓ CUDA 可用")
        logger.info(f"  GPU: {torch.cuda.get_device_name(0)}")
    else:
        logger.warning("⚠ CUDA 不可用，将使用 CPU（较慢）")
    
    # 构建模型
    model, connectivity_matrix = build_model(DEVICE)
    
    # 加载数据
    train_loader, test_loader = load_data(DATA_PATH)
    
    # 设置损失函数和优化器
    criterion = UnilateralMse(1.0)
    optimizer = torch.optim.AdamW(
        model.fc.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-4
    )
    
    logger.info("=" * 60)
    logger.info("开始训练")
    logger.info("=" * 60)
    
    best_accuracy = 0
    
    try:
        for epoch in range(NUM_EPOCHS):
            logger.info(f"\n--- Epoch {epoch+1}/{NUM_EPOCHS} ---")
            
            # 训练
            train_epoch(model, train_loader, optimizer, criterion, DEVICE, epoch)
            
            # 评估
            accuracy = evaluate(model, test_loader, DEVICE)
            
            # 保存最佳模型
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                logger.info(f"✓ 新的最佳精度: {best_accuracy:.2f}%")
                save_checkpoint(model, connectivity_matrix, DEVICE, SAVE_PATH)
            
            # 学习率衰减
            optimizer = lr_scheduler(optimizer, epoch, LEARNING_RATE, 10)
    
    except KeyboardInterrupt:
        logger.info("\n⚠ 训练被中断")
    
    finally:
        logger.info("\n" + "=" * 60)
        logger.info("训练完成")
        logger.info("=" * 60)
        logger.info(f"最佳精度: {best_accuracy:.2f}%")
        logger.info(f"模型已保存至: {SAVE_PATH}")
        
        # 验证文件
        if os.path.exists(SAVE_PATH):
            file_size = os.path.getsize(SAVE_PATH) / (1024 ** 2)  # 转换为 MB
            logger.info(f"✓ 文件大小: {file_size:.2f} MB")
        
        logger.info("\n下一步：")
        logger.info("1. 将 970.t7 放在 ELSM 目录中")
        logger.info("2. 运行: python evolve.py --device 0 --output ./results")

if __name__ == "__main__":
    main()
