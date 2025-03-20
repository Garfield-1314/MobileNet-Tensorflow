import numpy as np
import matplotlib.pyplot as plt

def generate_noise_image(size=(128, 128), save_path=None):
    # 生成随机噪声数据，范围从0到1
    noise = np.random.rand(size[0], size[1], 3)
    
    # 显示噪声图片
    plt.imshow(noise)
    plt.axis('off')  # 隐藏坐标轴
    
    # 保存图片（如果提供了保存路径）
    if save_path:
        plt.imsave(save_path, noise)
    
    plt.show()

# 生成并显示128x128的彩色噪声图片，并保存
generate_noise_image(save_path='./dataset/background/noise_image.png')
