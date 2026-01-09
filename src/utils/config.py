import sys
import json
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

class ConfigLoader:
    def __init__(self, config_file="config.json"):
        self.config_file = Path(config_file)
        self.config_data = self.load_config()
        # 定义每个配置项的预期类型
        self.type_definitions = {
            "game": {
                "GRID_WIDTH": int,
                "GRID_HEIGHT": int,
                "STATE_SIZE": int,
                "ACTION_SIZE": int
            },
            "training": {
                "EPISODES": int,
                "BATCH_SIZE": int,
                "GAMMA": float,
                "LEARNING_RATE": float,
                "EPSILON_INIT": float,
                "EPSILON_MIN": float,
                "EPSILON_DECAY": float,
                "REPLAY_BUFFER_SIZE": int,
                "TARGET_UPDATE_FREQ": int
            },
            "model": {
                "SAVE_INTERVAL": int,
                "MODEL_DIR": str,
                "LOG_DIR": str,
                "CHECKPOINT_PREFIX": str,
                "MODEL_EXTENSION": str,
                "TENSORBOARD_LOG_DIR": str
            },
            "test": {
                "GRID_SIZE": int,
                "TEST_EPISODES": int,
                "MAX_STEPS": int,
                "FPS": int,
                "EXPLORATION_RATE": float,
                "MIN_SCORE_THRESHOLD": int,
                "MIN_LENGTH_THRESHOLD": int,
                "PERFORMANCE_WINDOW": int,
                "MIN_AVG_REWARD": float,
                "MAX_TIME_PER_EPISODE": int,
                "SNAKE_GROWTH_RATE_THRESHOLD": float,
                "CONVERGENCE_THRESHOLD": float,
                "EXPLORATION_EFFICIENCY_THRESHOLD": float,
                "DECISION_QUALITY_THRESHOLD": float,
                "STABILITY_WINDOW": int,
                "MIN_Q_VALUE_DIFFERENCE": float,
                "MODEL_DIR": str,
                "MODEL_EXTENSION": str,
                "RESULT_DIR": str,
                "RESULT_IMG_DIR": str,
                "RESULT_DATA_DIR": str,
                "SAVE_GAMEPLAY_SCREEN": bool,
                "SCREENSHOT_DIR": str,
                "SCREENSHOT_QUALITY": int,
                "SAVE_GAMEPLAY_GIF": bool,
                "GIF_FPS": int,
                "GIF_LOOP": int,
                "GIF_QUALITY": int,
                "GIF_SUBSAMPLE": int
            }
        }
    
    def load_config(self):
        if self.config_file.exists():
            with open(self.config_file, 'r') as f:
                return json.load(f)
        return {}
    
    def get_value(self, section, key, default=None):
        if section in self.config_data and key in self.config_data[section]:
            value = self.config_data[section][key]
            # 尝试转换类型
            if section in self.type_definitions and key in self.type_definitions[section]:
                try:
                    target_type = self.type_definitions[section][key]
                    if target_type == bool:
                        # 特殊处理布尔值
                        return value.lower() == 'true' if isinstance(value, str) else bool(value)
                    else:
                        return target_type(value)
                except (ValueError, TypeError) as e:
                    print(f"警告: 无法将配置值 {section}.{key}={value} 转换为 {target_type.__name__} 类型")
                    return default
            return value
        return default

# 加载配置
config_loader = ConfigLoader()

class Config:
    # ========================
    # 游戏环境参数配置
    # ========================

    # 游戏网格宽度(单位：格子数)
    GRID_WIDTH = config_loader.get_value("game", "GRID_WIDTH", 16)
    # 游戏网格高度(单位：格子数)     
    GRID_HEIGHT = config_loader.get_value("game", "GRID_HEIGHT", 8)
    # 状态向量维度(12维特征)
    STATE_SIZE = config_loader.get_value("game", "STATE_SIZE", 12)
    # 动作空间大小(上、下、左、右)  
    ACTION_SIZE = config_loader.get_value("game", "ACTION_SIZE", 4)
    
    # ========================
    # 强化学习训练参数
    # ========================

    # 总训练轮次
    EPISODES = config_loader.get_value("training", "EPISODES", 10000)
    # 经验回放采样批量大小  
    BATCH_SIZE = config_loader.get_value("training", "BATCH_SIZE", 64)
    # 折扣因子(权衡当前奖励与未来奖励)
    GAMMA = config_loader.get_value("training", "GAMMA", 0.90)
    # 神经网络学习率
    LEARNING_RATE = config_loader.get_value("training", "LEARNING_RATE", 0.0005)
    # ε-贪婪策略初始探索率
    EPSILON_INIT = config_loader.get_value("training", "EPSILON_INIT", 1.0)
    # 最小探索率(保证持续探索)
    EPSILON_MIN = config_loader.get_value("training", "EPSILON_MIN", 0.05)
    # 探索率衰减系数(每轮乘以该值)
    EPSILON_DECAY = config_loader.get_value("training", "EPSILON_DECAY", 0.995)
    # 经验回放缓冲区容量
    REPLAY_BUFFER_SIZE = config_loader.get_value("training", "REPLAY_BUFFER_SIZE", 20000)
    # 目标网络更新频率
    TARGET_UPDATE_FREQ = config_loader.get_value("training", "TARGET_UPDATE_FREQ", 300)
    
    # ========================
    # 模型保存与日志配置
    # ========================

    # 每隔多少轮保存一次模型
    SAVE_INTERVAL = config_loader.get_value("model", "SAVE_INTERVAL", 500)
    # 模型保存目录
    MODEL_DIR = Path(config_loader.get_value("model", "MODEL_DIR", "saved_models"))
    # 训练日志目录
    LOG_DIR = Path(config_loader.get_value("model", "LOG_DIR", "logs"))
    # 模型文件名前缀           
    CHECKPOINT_PREFIX = config_loader.get_value("model", "CHECKPOINT_PREFIX", "snake_agent_")
    # 模型文件扩展名
    MODEL_EXTENSION = config_loader.get_value("model", "MODEL_EXTENSION", ".keras")
    
    # TensorBoard配置
    TENSORBOARD_LOG_DIR = Path(config_loader.get_value("model", "TENSORBOARD_LOG_DIR", "logs/tensorboard"))
    
    # 模型搜索模式
    MODEL_PATTERNS = [
        f"{CHECKPOINT_PREFIX}*" + MODEL_EXTENSION,  # 常规保存模型  
        "interrupted_model_*" + MODEL_EXTENSION,    # 中断保存模型
        "final_snake_model" + MODEL_EXTENSION,      # 最终模型 
        "error_snake_model_*" + MODEL_EXTENSION     # 错误保存模型
    ]

class TestConfig:
    # 游戏参数
    GRID_SIZE = config_loader.get_value("test", "GRID_SIZE", 40)  # 每个格子的像素大小
    GRID_WIDTH = config_loader.get_value("test", "GRID_WIDTH", 16)  # 与训练配置一致
    GRID_HEIGHT = config_loader.get_value("test", "GRID_HEIGHT", 8)  # 与训练配置一致
    
    # 状态大小（与训练一致）
    STATE_SIZE = config_loader.get_value("test", "STATE_SIZE", 12)
    # 动作空间大小（与训练一致）
    ACTION_SIZE = config_loader.get_value("test", "ACTION_SIZE", 4)
    
    # 颜色配置
    BG_COLOR = (50, 50, 50)
    GRID_COLOR = (80, 80, 80)
    SNAKE_COLOR = (0, 255, 0)
    HEAD_COLOR = (255, 0, 0)
    FOOD_COLOR = (0, 0, 255)
    TEXT_COLOR = (255, 255, 255)
    
    # 测试参数
    # 测试轮次
    TEST_EPISODES = config_loader.get_value("test", "TEST_EPISODES", 20)
    # 每轮最大步数
    MAX_STEPS = config_loader.get_value("test", "MAX_STEPS", 1000)
    # 游戏帧率
    FPS = config_loader.get_value("test", "FPS", 10)
    # 测试时的探索率
    EXPLORATION_RATE = config_loader.get_value("test", "EXPLORATION_RATE", 0.1)
    # 最低分数阈值
    MIN_SCORE_THRESHOLD = config_loader.get_value("test", "MIN_SCORE_THRESHOLD", 5)
    # 最低长度阈值
    MIN_LENGTH_THRESHOLD = config_loader.get_value("test", "MIN_LENGTH_THRESHOLD", 5)
    # 性能评估窗口大小
    PERFORMANCE_WINDOW = config_loader.get_value("test", "PERFORMANCE_WINDOW", 5)
    # 最低平均奖励阈值
    MIN_AVG_REWARD = config_loader.get_value("test", "MIN_AVG_REWARD", 0.5)
    # 单轮最大允许时间(秒)
    MAX_TIME_PER_EPISODE = config_loader.get_value("test", "MAX_TIME_PER_EPISODE", 60)
    # 蛇长度增长率阈值
    SNAKE_GROWTH_RATE_THRESHOLD = config_loader.get_value("test", "SNAKE_GROWTH_RATE_THRESHOLD", 0.3)
    
    # 收敛阈值(分数标准差)
    CONVERGENCE_THRESHOLD = config_loader.get_value("test", "CONVERGENCE_THRESHOLD", 0.1)
    # 探索效率阈值
    EXPLORATION_EFFICIENCY_THRESHOLD = config_loader.get_value("test", "EXPLORATION_EFFICIENCY_THRESHOLD", 0.7)
    # 决策质量阈值
    DECISION_QUALITY_THRESHOLD = config_loader.get_value("test", "DECISION_QUALITY_THRESHOLD", 0.8)
    # 稳定性评估窗口大小
    STABILITY_WINDOW = config_loader.get_value("test", "STABILITY_WINDOW", 3)
    # 最小Q值差异阈值
    MIN_Q_VALUE_DIFFERENCE = config_loader.get_value("test", "MIN_Q_VALUE_DIFFERENCE", 0.5)
    
    # 模型路径
    MODEL_DIR = Path(config_loader.get_value("test", "MODEL_DIR", "saved_models"))
    MODEL_EXTENSION = config_loader.get_value("test", "MODEL_EXTENSION", ".keras")
    MODEL_PATTERNS = [
        "final_snake_model" + MODEL_EXTENSION,
        "snake_agent_*" + MODEL_EXTENSION,
        "interrupted_model_*" + MODEL_EXTENSION
    ]
    
    # 结果保存路径
    RESULT_DIR = Path(config_loader.get_value("test", "RESULT_DIR", "test_results"))
    RESULT_IMG_DIR = Path(config_loader.get_value("test", "RESULT_IMG_DIR", "test_results/images"))
    RESULT_DATA_DIR = Path(config_loader.get_value("test", "RESULT_DATA_DIR", "test_results/data"))
    
    # 游戏画面截图参数
    SAVE_GAMEPLAY_SCREEN = config_loader.get_value("test", "SAVE_GAMEPLAY_SCREEN", True)  # 是否保存游戏画面
    SCREENSHOT_DIR = Path(config_loader.get_value("test", "SCREENSHOT_DIR", "game_img"))  # 游戏画面保存目录
    SCREENSHOT_QUALITY = config_loader.get_value("test", "SCREENSHOT_QUALITY", 95)  # 截图质量(1-100)
    
    # GIF生成参数
    SAVE_GAMEPLAY_GIF = config_loader.get_value("test", "SAVE_GAMEPLAY_GIF", True)  # 是否保存游戏画面为GIF
    GIF_FPS = config_loader.get_value("test", "GIF_FPS", 10)  # GIF帧率
    GIF_LOOP = config_loader.get_value("test", "GIF_LOOP", 0)  # GIF循环次数，0表示无限循环
    GIF_QUALITY = config_loader.get_value("test", "GIF_QUALITY", 85)  # GIF质量百分比
    GIF_SUBSAMPLE = config_loader.get_value("test", "GIF_SUBSAMPLE", 1)  # 每隔N帧取一帧，减少GIF文件大小