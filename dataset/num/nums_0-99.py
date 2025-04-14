import os
import random
from PIL import Image, ImageDraw, ImageFont

def generate_digit_images(font_dir, output_dir,
                        digits_range=(0, 99),
                        total_samples=1000,
                        image_size=(128, 128),
                        font_scale=0.6,
                        font_color=(0, 0, 0),
                        bg_color=(255, 255, 255)):
    """
    生成与示例图片风格一致的数字图像
    
    参数说明：
    :param font_dir: 字体文件夹路径
    :param output_dir: 输出目录路径
    :param digits_range: 生成数字范围 (默认0-99)
    :param total_samples: 总生成样本数量
    :param image_size: 图像尺寸 (宽, 高)
    :param font_scale: 字体缩放系数（基于图像高度）
    :param font_color: 字体颜色 (RGB)
    :param bg_color: 背景颜色 (RGB)
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载所有有效字体
    font_files = []
    for f in os.listdir(font_dir):
        if f.lower().endswith(('.ttf', '.otf')):
            font_path = os.path.join(font_dir, f)
            try:
                ImageFont.truetype(font_path, 10)
                font_files.append(font_path)
            except:
                continue
                
    if not font_files:
        raise FileNotFoundError("未找到有效字体文件")
    
    # 计算样本分布
    num_digits = digits_range[1] - digits_range[0] + 1
    samples_per_digit = total_samples // num_digits
    remainder = total_samples % num_digits
    
    # 生成图像
    for digit in range(digits_range[0], digits_range[1]+1):
        current_samples = samples_per_digit + (1 if digit < remainder else 0)
        for sample in range(current_samples):
            # 随机选择字体
            font_path = random.choice(font_files)
            
            # 创建画布
            img = Image.new('RGB', image_size, bg_color)
            draw = ImageDraw.Draw(img)
            
            # 计算字体尺寸
            font_size = int(image_size[1] * font_scale)
            font = ImageFont.truetype(font_path, font_size)
            
            # 绘制双位数字
            text = f"{digit:02d}"  # 保持两位数格式
            text_bbox = font.getbbox(text)
            
            # 自动调整字体大小
            while True:
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]
                if text_width < image_size[0]*0.9 and text_height < image_size[1]*0.9:
                    break
                font_size -= 1
                font = ImageFont.truetype(font_path, font_size)
                text_bbox = font.getbbox(text)
            
            # 计算居中位置
            x = (image_size[0] - text_width) / 2 - text_bbox[0]
            y = (image_size[1] - text_height) / 2 - text_bbox[1]
            
            # 绘制文字
            draw.text((x, y), text, font=font, fill=font_color)
            
            # 保存文件
            filename = f"{digit:02d}_{sample:04d}.png"
            img.save(os.path.join(output_dir, filename))

if __name__ == '__main__':
    generate_digit_images(
        font_dir='font',  # 字体文件夹路径
        output_dir='output',  # 输出目录
        digits_range=(0, 99),  # 生成0-99
        total_samples=10000,  # 总样本数
        image_size=(224, 224),  # 图像尺寸
        font_scale=0.55,  # 字体大小系数
        font_color=(0, 0, 0),  # 黑色字体
        bg_color=(255, 255, 255)  # 白色背景
    )