import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import numpy as np
import datetime
import os
import tensorflow as tf

from src.utils.logger import ColorLogger
from src.utils.config import Config


def initialize_environment(load_prev_model=True):
    """环境初始化与检测
    
    执行训练前的环境准备工作，包括：
    1. 创建必要的目录（模型、日志、TensorBoard）
    2. 检测依赖库版本
    3. 查找并加载之前训练的模型
    
    参数:
        load_prev_model (bool): 是否加载之前训练的模型
        
    返回:
        Path: 最新模型的路径，如果没有找到则返回None
    """
    ColorLogger.highlight("===== 环境初始化 =====")
    
    # 创建必要目录
    for dir_path in [Config.MODEL_DIR, Config.LOG_DIR, Config.TENSORBOARD_LOG_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)
        ColorLogger.info(f"创建目录: {dir_path}")
    
    # 库版本检测
    ColorLogger.info(f"TensorFlow版本: {tf.__version__} (要求2.10.0+)")
    ColorLogger.info(f"NumPy版本: {np.__version__} (要求1.21.0+)")
    
    # 模型路径检测
    latest_model = None
    if load_prev_model:
        # 收集所有可能的模型文件
        model_files = []
        for pattern in Config.MODEL_PATTERNS:
            model_files.extend(Config.MODEL_DIR.glob(pattern))
        
        # 兼容处理：如果没有找到.keras模型，尝试查找.h5格式
        if not model_files:
            legacy_patterns = [p.replace(Config.MODEL_EXTENSION, ".h5") for p in Config.MODEL_PATTERNS]
            for pattern in legacy_patterns:
                model_files.extend(Config.MODEL_DIR.glob(pattern))
        
        if model_files:
            # 按修改时间排序，确保加载最新模型
            model_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            latest_model = model_files[0]
            
            # 获取模型详细信息
            model_name = latest_model.name
            mod_time = datetime.datetime.fromtimestamp(
                os.path.getmtime(latest_model)
            ).strftime("%Y-%m-%d %H:%M:%S")
            file_size = os.path.getsize(latest_model) / (1024 * 1024)  # MB
            
            ColorLogger.success(
                f"发现最新模型:\n"
                f" 名称: {model_name}\n"
                f" 修改时间: {mod_time}\n"
                f" 大小: {file_size:.2f}MB\n"
                f" 路径: {latest_model}"
            )
        else:
            ColorLogger.warning("未发现上一次训练模型，将从头开始")
    
    ColorLogger.highlight("===== 环境初始化完成 =====")
    return latest_model