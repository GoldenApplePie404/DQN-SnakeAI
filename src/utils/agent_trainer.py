import numpy as np
import random
import time
import tensorflow as tf
import datetime
from src.utils.config import Config
from src.utils.logger import ColorLogger
from src.utils.tmonitor import TrainingMonitor
from src.utils.device import get_training_device

devive = get_training_device()

class AgentTrainer:
    """智能体训练核心模块，实现强化学习训练逻辑"""
    
    def __init__(self, agent, env_handler, replay_buffer, model_manager, logger):
        self.agent = agent  # QNetwork实例
        self.env_handler = env_handler  # EnvironmentHandler实例
        self.replay_buffer = replay_buffer  # ReplayBuffer实例
        self.model_manager = model_manager  # ModelManager实例
        self.logger = logger  # TrainingLogger实例
        self.monitor = TrainingMonitor()
        
        # 训练状态
        self.score_history = []
        self.loss_history = []
        self.episodes_x = []
    def _cleanup_resources(self, episode):
        """资源清理函数"""
        # 每轮清理TensorFlow会话
        tf.keras.backend.clear_session()
        
        # 每10轮强制Python垃圾回收
        if episode % 10 == 0:
            import gc
            gc.collect()
            
        # 清空未使用的变量引用
        if hasattr(self, 'temp_variables'):
            del self.temp_variables
            self.temp_variables = {}
        
    def train(self, start_episode=0):
        """开始训练主循环
        
        Args:
            start_episode (int): 起始训练轮次
            
        Returns:
            tuple: (score_history, loss_history, episodes_x)
        """
        # 初始化训练状态
        training_start_time = time.time()
        start_datetime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ColorLogger.highlight(f"\n===== 训练开始于: {start_datetime} =====\n")
        ColorLogger.info(f"总训练轮次: {Config.EPISODES} | 起始轮次: {start_episode}")
        
        # 进度条配置
        from tqdm import tqdm
        pbar = tqdm(range(start_episode, Config.EPISODES), desc="训练进度", 
                   initial=start_episode, total=Config.EPISODES,
                   bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}, {postfix}]")
        
        try:
            for episode in pbar:
                # 检查退出信号
                if self.monitor.should_end():
                    ColorLogger.warning("\n用户请求退出训练...")
                    self.model_manager.save_model(episode, is_interrupted=True)
                    break
                    
                # 单轮训练
                episode_metrics = self._train_single_episode(episode, pbar)
                self._record_training_history(episode_metrics)
                
                # 定期更新目标网络
                if episode % Config.TARGET_UPDATE_FREQ == 0:
                    self.agent.update_target_network()
                    ColorLogger.info(f"目标网络更新完成，轮次: {episode}\n")
                    
                # 自动保存模型
                if episode > 0 and episode % Config.SAVE_INTERVAL == 0:
                    self.model_manager.save_model(episode)
                    
                # 清理资源
                self._cleanup_resources(episode)
                
            # 训练完成处理
            self.model_manager.save_model(episode, is_final=True)
            self.model_manager.convert_to_tflite(self.env_handler)
            
        except Exception as e:
            ColorLogger.error(f"\n训练过程中发生错误: {str(e)}")
            self.model_manager.save_model(episode, is_interrupted=True)
        finally:
            # 训练总结
            self.logger.close()
            end_datetime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            total_training_time = str(datetime.timedelta(seconds=int(time.time()-training_start_time)))
            ColorLogger.highlight(f"\n===== 训练结束于: {end_datetime} =====\n")
            ColorLogger.info(f"总训练轮次: {episode+1 - start_episode} | 总耗时: {total_training_time}")
            
        return self.score_history, self.loss_history, self.episodes_x
        
    def _train_single_episode(self, episode, pbar):
        """训练单轮episode
        
        Returns:
            dict: 包含轮次指标的字典
        """
        episode_start_time = time.time()
        state = self.env_handler.reset()
        total_reward = 0
        steps = 0
        loss_sum = 0
        inference_time = 0

        global_episode = episode
        
        while True:
            # ε-贪婪策略选择动作
            epsilon = max(Config.EPSILON_MIN, Config.EPSILON_INIT * (Config.EPSILON_DECAY ** global_episode))
            action = self._choose_action(state, epsilon)
            
            # 执行动作
            start_time = time.time()
            next_state, reward, done = self.env_handler.step(action)
            inference_time += (time.time() - start_time) * 1000  # 毫秒
            
            # 存储经验
            self.replay_buffer.add((state, action, reward, next_state, done))
            total_reward += reward
            steps += 1
            
            # 经验回放训练
            loss = self._experience_replay()
            loss_sum += loss
            
            state = next_state
            
            if done:
                # 计算轮次指标
                metrics = self._calculate_episode_metrics(
                    episode, episode_start_time, total_reward, steps, loss_sum, 
                    inference_time, epsilon
                )
                
                # 更新进度条
                pbar.set_postfix({
                    '分数': metrics['score'],
                    'ε': f"{metrics['epsilon']:.3f}",
                    '损失': f"{metrics['avg_loss']:.4f}",
                    '耗时': metrics['elapsed_time_str']
                })
                
                # 记录日志
                self.logger.log_episode_metrics(episode, metrics)
                return metrics
                
    def _choose_action(self, state, epsilon):
        """基于ε-贪婪策略选择动作
        
        Args:
            state: 当前状态
            epsilon: 探索率
            
        Returns:
            int: 选择的动作
        """
        if random.random() < epsilon:
            return random.randint(0, Config.ACTION_SIZE-1)
        else:
            q_values = self.agent.predict_single(state)
            return np.argmax(q_values)
            
    def _experience_replay(self):
        """经验回放训练
        
        Returns:
            float: 训练损失
        """
        if len(self.replay_buffer) < Config.BATCH_SIZE:
            return 0
            
        # 采样并转换为张量
        batch = self.replay_buffer.sample(Config.BATCH_SIZE)
        states = tf.convert_to_tensor([exp[0] for exp in batch], dtype=tf.float32)
        actions = tf.convert_to_tensor([exp[1] for exp in batch], dtype=tf.int32)
        rewards = tf.convert_to_tensor([exp[2] for exp in batch], dtype=tf.float32)
        next_states = tf.convert_to_tensor([exp[3] for exp in batch], dtype=tf.float32)
        dones = tf.convert_to_tensor([exp[4] for exp in batch], dtype=tf.float32)
        
        # 计算目标Q值（双Q学习）
        next_q = self.agent.target_predict_batch(next_states)
        max_next_q = np.max(next_q, axis=1)
        targets = self.agent.predict_batch(states)
        
        actions_one_hot = tf.one_hot(actions, Config.ACTION_SIZE)
        targets = rewards + (Config.GAMMA * max_next_q * (1 - dones))
        targets = tf.expand_dims(targets, 1)
        targets = tf.where(actions_one_hot == 1, targets, self.agent.model.predict(states, verbose=0))
            
        # 训练并返回损失
        loss = self.agent.train(states, targets)
        
        # 显式释放张量
        del states, actions, rewards, next_states, dones
        return loss
            
    def _calculate_episode_metrics(self, episode, start_time, total_reward, steps, loss_sum, inference_time, epsilon):
        """计算单轮训练指标
        
        Returns:
            dict: 包含各类指标的字典
        """
        episode_time = time.time() - start_time
        elapsed_time = time.time() - self.logger.training_start_time
        gpu_memory = self.monitor.record_memory_usage(episode, device=devive)
        
        return {
            'score': self.env_handler.score,
            'total_reward': total_reward,
            'steps': steps,
            'avg_loss': loss_sum / steps if steps > 0 else 0,
            'avg_inference_time': inference_time / steps if steps > 0 else 0,
            'epsilon': epsilon,
            'episode_time': episode_time,
            'episode_time_str': str(datetime.timedelta(seconds=int(episode_time))),
            'elapsed_time': elapsed_time,
            'elapsed_time_str': str(datetime.timedelta(seconds=int(elapsed_time))),
            'gpu_memory': gpu_memory
        }
        
    def _record_training_history(self, metrics):
        """记录训练历史"""
        self.score_history.append(metrics['score'])
        self.loss_history.append(metrics['avg_loss'])
        self.episodes_x.append(len(self.episodes_x) + 1)  