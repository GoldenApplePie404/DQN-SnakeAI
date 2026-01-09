import random
from collections import deque
import tensorflow as tf

class ReplayBuffer:
    """经验回放缓冲区
    
    用于存储智能体与环境交互的经验，支持随机采样批量经验进行训练。
    使用双端队列（deque）实现，具有固定容量，当容量满时自动移除最早的经验。
    
    属性:
        buffer (deque): 存储经验的双端队列
    """
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, experience):
        """添加经验到缓冲区
        
        Args:
            experience (tuple): 包含(state, action, reward, next_state, done)的经验元组
        """
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        """从缓冲区采样批量经验
        
        Args:
            batch_size (int): 采样数量
            
        Returns:
            list: 采样的经验列表，若缓冲区大小不足则返回空列表
        """
        return random.sample(self.buffer, batch_size) if len(self.buffer) >= batch_size else []
    
    def __len__(self):
        """返回当前缓冲区大小"""
        return len(self.buffer)
    def sample_batch(self, batch_size):
        """向量化采样批量经验"""
        if len(self.buffer) < batch_size:
            return None
            
        batch = random.sample(self.buffer, batch_size)
        states = tf.convert_to_tensor([exp[0] for exp in batch], dtype=tf.float32)
        actions = tf.convert_to_tensor([exp[1] for exp in batch], dtype=tf.int32)
        rewards = tf.convert_to_tensor([exp[2] for exp in batch], dtype=tf.float32)
        next_states = tf.convert_to_tensor([exp[3] for exp in batch], dtype=tf.float32)
        dones = tf.convert_to_tensor([exp[4] for exp in batch], dtype=tf.float32)
        
        return states, actions, rewards, next_states, dones