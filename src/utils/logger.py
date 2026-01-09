# logger.py

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from colorama import init, Fore, Back, Style

# 初始化彩色输出
init(autoreset=True)

class ColorLogger:
    @staticmethod
    def info(message):
        print(f"{Fore.LIGHTMAGENTA_EX}\n[INFO]\n {message}{Style.RESET_ALL}")  # 亮粉紫色
    
    @staticmethod
    def success(message):
        print(f"{Fore.LIGHTGREEN_EX}\n[SUCCESS]\n {message}{Style.RESET_ALL}")  # 亮绿色
    
    @staticmethod
    def warning(message):
        print(f"{Fore.LIGHTYELLOW_EX}\n[WARNING]\n {message}{Style.RESET_ALL}")  # 亮黄色
    
    @staticmethod
    def error(message):
        print(f"{Fore.LIGHTRED_EX}\n[ERROR]\n {message}{Style.RESET_ALL}")  # 亮红色
    
    @staticmethod
    def highlight(message):
        print(f"{Fore.LIGHTCYAN_EX}\n[HIGHLIGHT]\n {message}{Style.RESET_ALL}")  # 亮青色