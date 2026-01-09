import tensorflow as tf

def get_training_device(force_cpu=False):
    """自动选择最优训练设备"""
    if force_cpu:
        print("\n[设备]强制使用CPU模式")
        return '/CPU:0'
    
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"\n[设备]使用GPU: {tf.test.gpu_device_name()}")
            return '/GPU:0'
        except RuntimeError as e:
            print(f"\n[设备]GPU初始化失败: {str(e)}")
            return '/CPU:0'
    else:
        print("\n[设备] 未检测到GPU，使用CPU")
        return '/CPU:0'
    
get_training_device()