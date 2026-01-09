from src.game.env import SnakeEnv


class EnvironmentHandler:
    """环境交互模块，封装游戏环境的初始化与状态管理"""
    
    def __init__(self, render_mode=None):
        self.env = SnakeEnv(render_mode=render_mode)
        self.state = None
        
    def reset(self):
        """重置环境并返回初始状态"""
        self.state = self.env.reset()
        return self.state
        
    def step(self, action):
        """执行动作并返回环境反馈
        
        Args:
            action (int): 智能体选择的动作
            
        Returns:
            tuple: (next_state, reward, done)
        """
        next_state, reward, done = self.env.step(action)
        self.state = next_state
        return next_state, reward, done
        
    @property
    def score(self):
        """获取当前游戏得分"""
        return self.env.score
        
    def close(self):
        """关闭环境资源"""
        if hasattr(self.env, 'close'):
            self.env.close()