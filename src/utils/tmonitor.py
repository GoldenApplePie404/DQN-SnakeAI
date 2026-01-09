import sys
import subprocess
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import threading
import time
import tensorflow as tf
import keyboard
from src.utils.logger import ColorLogger
from src.utils.device import get_training_device


class TrainingMonitor:
    def __init__(self):
        """初始化训练监控器"""
        self.finish_training = False  # 训练结束标志
        self.save_requested = False   # 保存请求标志
        self.lock = threading.Lock()  # 线程锁
        self.event = threading.Event()  # 事件机制，用于线程控制
        self.thread = threading.Thread(target=self._monitor_keyboard, daemon=True)
        self.thread.start()
        self.memory_log = []  # 新增：内存监控日志
    
    def _monitor_keyboard(self):
        ColorLogger.info("\n===== 训练控制 =====\n按 '↑+Q' 键: 立即退出训练\n按 '↑+S' 键: 手动保存当前模型\n===================\n")
        while not self.event.is_set():
            with self.lock:
                # 检测立即退出请求
                if keyboard.is_pressed('up+q') and not self.finish_training:
                    self.finish_training = True
                    ColorLogger.warning("\n检测到立即退出请求，正在终止训练...")
                    self.event.set()  
                    break
                # 检测手动保存请求
                if keyboard.is_pressed('up+s') and not self.save_requested:
                    self.save_requested = True
                    ColorLogger.success("\n检测到手动保存请求，将保存当前模型...")
                    time.sleep(1)  # 防止重复触发
            self.event.wait(0.05)  # 降低CPU占用
    
    def should_end(self):  
        """检查是否应该结束训练
        
        返回:
            bool: True表示应该结束训练，False表示继续训练
        """
        with self.lock:
            return self.finish_training
    
    def should_save(self):
        """检查是否应该保存模型
        
        返回:
            bool: True表示应该保存模型，False表示不需要保存
        """
        with self.lock:
            if self.save_requested:
                self.save_requested = False
                return True
            return False
    
    def record_memory_usage(self, episode,device):
        """记录当前GPU内存使用情况"""
        if not hasattr(self, '_cpu_memory_error_shown'):
            self._cpu_memory_error_shown = False
        
        try:
            
            if 'GPU' in device:
                # GPU模式 - 正常执行内存监控
                tf_mem = tf.config.experimental.get_memory_info('GPU:0')
                smi_output = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]
                ).decode().strip()
                
                self.memory_log.append({
                    "episode": episode,
                    "tf_peak": tf_mem['peak'],
                    "sys_used": int(smi_output)
                })
                
                # 检查内存增长情况
                if len(self.memory_log) >= 3:
                    recent = self.memory_log[-3:]
                    growth_rate = (recent[-1]['sys_used'] - recent[0]['sys_used']) / 3
                    if growth_rate > 5:  # 连续3个周期增长超过5MB
                        ColorLogger.warning(f"内存异常增长: {growth_rate:.2f}MB/episode")
                        
            elif 'CPU' in device and not self._cpu_memory_error_shown:
                # CPU模式 - 只在第一次时显示错误信息
                ColorLogger.error("内存监控失败: 当前运行在CPU模式下，无法获取GPU内存信息")
                self._cpu_memory_error_shown = True
                
        except Exception as e:
            # 只在GPU模式下输出异常（CPU模式已经处理过）
            if 'GPU' in get_training_device():
                ColorLogger.error(f"内存监控失败: {str(e)}")
                