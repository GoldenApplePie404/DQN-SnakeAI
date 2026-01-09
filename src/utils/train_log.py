import csv
import time
import tensorflow as tf
import datetime
import threading
from src.utils.config import Config
from src.utils.logger import ColorLogger

class TrainingLogger:
    """日志与监控模块，处理训练日志与资源监控"""
    
    def __init__(self):
        self.log_file = None
        self.log_writer = None
        self.tensorboard_writer = None
        self.training_start_time = time.time()
        self._init_logging()
        
    def _init_logging(self):
        """初始化日志系统"""
        # 创建日志目录
        Config.LOG_DIR.mkdir(exist_ok=True)
        Config.TENSORBOARD_LOG_DIR.mkdir(exist_ok=True)
        
        # 初始化CSV日志
        log_filename = f"training_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        self.log_path = Config.LOG_DIR / log_filename
        self.log_file = open(self.log_path, 'w', newline='')
        self.log_writer = csv.writer(self.log_file)
        self.log_writer.writerow([
            'episode', 'score', 'total_reward', 'epsilon', 'loss', 
            'steps', 'inference_time', 'episode_time', 'elapsed_time',
            'gpu_memory_used_mb'
        ])
        
        # 初始化TensorBoard
        tensorboard_log_dir = Config.TENSORBOARD_LOG_DIR / datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.tensorboard_writer = tf.summary.create_file_writer(str(tensorboard_log_dir))
        ColorLogger.info(f"TensorBoard日志将保存至: {tensorboard_log_dir}")
        
    def log_episode_metrics(self, episode, metrics):
        threading.Thread(target=self._async_write, args=(episode, metrics)).start()

    def _async_write(self, episode, metrics):
        # 写入CSV日志
        self.log_writer.writerow([
            episode, metrics['score'], metrics['total_reward'], metrics['epsilon'],
            metrics['avg_loss'], metrics['steps'], metrics['avg_inference_time'],
            metrics['episode_time_str'], metrics['elapsed_time_str'], metrics['gpu_memory']
        ])
        self.log_file.flush()
        
        # 写入TensorBoard
        with self.tensorboard_writer.as_default():
            tf.summary.scalar('score', metrics['score'], step=episode)
            tf.summary.scalar('loss', metrics['avg_loss'], step=episode)
            tf.summary.scalar('epsilon', metrics['epsilon'], step=episode)
            tf.summary.scalar('steps', metrics['steps'], step=episode)
            
    def get_gpu_memory_usage(self):
        """获取GPU内存使用情况(MB)"""
        try:
            mem_info = tf.config.experimental.get_memory_info('GPU:0')
            return mem_info['current'] // (1024 * 1024)
        except:
            return 0
            
    def close(self):
        """关闭日志资源"""
        if self.log_file:
            self.log_file.close()
            ColorLogger.success(f"训练日志已保存至: {self.log_path}")
            
        if self.tensorboard_writer:
            self.tensorboard_writer.close()
            ColorLogger.info("TensorBoard写入器已关闭")