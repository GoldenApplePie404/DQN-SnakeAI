import tensorflow as tf
import numpy as np
from pathlib import Path
from src.utils.config import Config
from src.utils.logger import ColorLogger
from src.utils.t_state import TrainingStateManager

class ModelManager:
    """模型管理模块，处理模型加载、保存与转换"""
    
    def __init__(self, agent):
        self.agent = agent  
        self.latest_model = None
        self.state_manager = TrainingStateManager()
        
    def load_latest_model(self, load_prev_model=True):
        """加载最新模型并返回起始训练轮次
        
        Returns:
            int: 起始训练轮次
        """
        if not load_prev_model:
            return 0
        
        # 优先使用状态文件获取最新模型
        model_path = self.state_manager.get_last_model_path()
        if model_path and Path(model_path).exists():
            self.latest_model = model_path
            self.state_manager.validate_config_compatibility()
            return self.state_manager.get_last_episode() + 1
        
        # 回退到原有逻辑
        from src.utils.ini_env import initialize_environment
        self.latest_model = initialize_environment(load_prev_model)
        
        if not self.latest_model:
            return 0
        
        try:
            self.agent.model = tf.keras.models.load_model(self.latest_model)
            ColorLogger.success(f"成功加载模型: {self.latest_model}")
            # 从文件名提取轮次并更新状态管理器
            start_episode = self._extract_start_episode()
            self.state_manager.save_state(start_episode - 1, self.latest_model)
            return start_episode
        except Exception as e:
            ColorLogger.error(f"模型加载失败: {str(e)}，将从头开始训练")
            return 0
            
    def _extract_start_episode(self):
        """从模型文件名提取起始训练轮次"""
        model_name = Path(self.latest_model).stem
        
        if model_name.startswith(Config.CHECKPOINT_PREFIX):
            return int(model_name.split("_")[-1]) + 1
        elif model_name.startswith(("interrupted_model_", "error_snake_model_")):
            return int(model_name.split("_")[-1]) + 1
        return 0
        
    def save_model(self, episode, is_final=False, is_interrupted=False):
        """保存模型到指定路径
        
        Args:
            episode (int): 当前训练轮次
            is_final (bool): 是否为最终模型
            is_interrupted (bool): 是否因中断保存
            
        Returns:
            str: 保存路径
        """
        if is_final:
            filename = f"final_snake_model{Config.MODEL_EXTENSION}"
        elif is_interrupted:
            filename = f"interrupted_model_{episode}{Config.MODEL_EXTENSION}"
        else:
            filename = f"{Config.CHECKPOINT_PREFIX}{episode}{Config.MODEL_EXTENSION}"
            
        save_path = Config.MODEL_DIR / filename
        self.agent.model.save(save_path)
        ColorLogger.success(f"模型保存至: {save_path}")
        self.state_manager.save_state(episode, save_path)
        return str(save_path)
        
    def convert_to_tflite(self, env_handler):
        """将模型转换为TFLite格式
        
        Args:
            env_handler (EnvironmentHandler): 环境处理器实例
        """
        try:
            converter = tf.lite.TFLiteConverter.from_keras_model(self.agent.model)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            
            def representative_dataset():
                for _ in range(100):
                    state = env_handler.reset()
                    yield [state[np.newaxis, :].astype(np.float32)]
                    
            converter.representative_dataset = representative_dataset
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8
            
            tflite_model = converter.convert()
            tflite_path = Config.MODEL_DIR / "snake_model.tflite"
            with open(tflite_path, "wb") as f:
                f.write(tflite_model)
            ColorLogger.success(f"量化模型转换完成: {tflite_path}")
        except Exception as e:
            ColorLogger.error(f"TFLite模型转换失败: {str(e)}")
            
    def update_target_network(self):
        """更新目标网络"""
        self.agent.update_target_network()
        ColorLogger.info("目标网络更新完成")