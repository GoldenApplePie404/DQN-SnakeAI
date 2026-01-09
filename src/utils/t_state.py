import json
import datetime
from pathlib import Path
from src.utils.config import Config
from src.utils.logger import ColorLogger

class TrainingStateManager:
    """训练状态管理模块，独立跟踪训练进度"""
    
    def __init__(self):
        self.state_file = Config.MODEL_DIR / "training_state.json"
        self.state = self._load_state()
        
    def _load_state(self):
        """加载训练状态"""
        if self.state_file.exists():
            try:
                with open(self.state_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                ColorLogger.warning(f"训练状态文件损坏，将创建新文件: {str(e)}")
        
        # 默认状态
        return {
            "last_episode": 0,
            "last_save_time": None,
            "model_path": None,
            "training_config": {
                "batch_size": Config.BATCH_SIZE,
                "learning_rate": Config.LEARNING_RATE,
                "epsilon_init": Config.EPSILON_INIT
            }
        }
    
    def save_state(self, episode, model_path):
        """保存训练状态"""
        self.state.update({
            "last_episode": episode,
            "last_save_time": datetime.datetime.now().isoformat(),
            "model_path": str(model_path),
            "training_config": {
                "batch_size": Config.BATCH_SIZE,
                "learning_rate": Config.LEARNING_RATE,
                "epsilon_init": Config.EPSILON_INIT
            }
        })
        
        # 确保目录存在
        Config.MODEL_DIR.mkdir(exist_ok=True)
        
        with open(self.state_file, "w") as f:
            json.dump(self.state, f, indent=2)
        
        ColorLogger.success(f"训练状态已保存至: {self.state_file}")
    
    def get_last_episode(self):
        """获取最后训练轮次"""
        return self.state["last_episode"]
    
    def get_last_model_path(self):
        """获取最后模型路径"""
        return self.state.get("model_path")
    
    def validate_config_compatibility(self):
        """验证当前配置与上次训练的兼容性"""
        current_config = {
            "batch_size": Config.BATCH_SIZE,
            "learning_rate": Config.LEARNING_RATE,
            "epsilon_init": Config.EPSILON_INIT
        }
        
        saved_config = self.state.get("training_config", {})
        
        # 检查关键配置是否匹配
        incompatible = []
        for key, value in current_config.items():
            if saved_config.get(key) != value:
                incompatible.append(f"{key}: {saved_config.get(key)} → {value}")
        
        if incompatible:
            ColorLogger.warning("检测到配置变更，可能影响训练连续性:")
            for change in incompatible:
                ColorLogger.warning(f"  {change}")
            return False
        return True