import os
import random
from PIL import Image, ImageDraw, ImageFont

def generate_custom_digits(font_dir, output_root,
                          digits_range=(0, 9),
                          total_samples=1000,
                          image_size=(64, 64),
                          scale_factor=0.7,
                          h_scale_range=(0.8, 1.2),
                          v_scale_range=(0.8, 1.2)):
    """
    自定义数量且分目录存储的生成器
    
    :param font_dir: 字体文件夹路径
    :param output_root: 输出根目录
    :param digits_range: 数字范围 (start, end)
    :param total_samples: 总生成数量（每个数字平均分配）
    :param image_size: 图像尺寸
    :param scale_factor: 基础尺寸比例
    :param h_scale_range: 水平缩放范围
    :param v_scale_range垂直缩放范围
    """
    # 获取所有可用字体
    font_files = _get_valid_fonts(font_dir)
    if not font_files:
        raise File(f"No valid fonts found in {font_dir}")
    
    # 创建输出目录
    for d in range(digits_range[0], digits_range[1] + 1):
        digit_dir = os.path.join(output_root, str(d))
        os.makedirs(digit_dir, exist_ok=True)
    
    # 计算样本分配
    num_digits = digits_range[1] - digits_range[0] + 1
    samples_per_digit = total_samples // num_digits
    remainder = total_samples % num_digits
    
    # 生成样本
    for idx, digit in enumerate(range(digits_range[0], digits_range[1] + 1)):
        current_samples = samples_per_digit + (1 if idx < remainder else 0)
        for sample_idx in range(current_samples):
            _generate_single_digit(
                digit=digit,
                fonts=font_files,  #可变列表以移除无效字体
                output_dir=output_root,
                sample_idx=sample_idx,
                image_size=image_size,
                scale_factor=scale_factor,
                h_scale_range=h_scale_range,
                v_scale_range=v_scale_range
            )

def _get_valid_fonts(font_dir):
    """获取有效字体文件列表（初步筛选）"""
    fonts = []
    for f in os.listdir(font_dir):
        if f.lower().endswith(('.ttf', '.otf', '.ttc')):
            font_path = os.path.join(font_dir, f)
            if _is_valid_font(font_path):
                fonts.append(font_path)
    return fonts

def _is_valid_font(font_path):
    """基础字体有效性检查"""
    try:
        ImageFont.truetype(font_path, 10)
        return True
    except:
        return False

def _generate_single_digit(digit, fonts, output_dir,
                          sample_idx, image_size,
                          scale_factor, h_scale_range,
                          v_scale_range):
    """生成单个数字样本，自动跳过无效字体"""
    # 确保字体列表不为空
    if not fonts:
        raise ValueError("No valid fonts available for generation.")
    
    # 计算基础字号
    base_font_size = int(image_size[1] * scale_factor)
    
    # 多次尝试生成，直到成功
    while True:
        try:
            # 随机选择字体
            font_path = random.choice(fonts)
            font = ImageFont.truetype(font_path, base_font_size)
            break
        except:
            # 移除非法的字体文件
            print(f"移除无效字体: {font_path}")
            fonts.remove(font_path)
            if not fonts:
                raise RuntimeError("所有字体均无效，无法生成样本。")
    
    # 创建白色画布
    img = Image.new('RGB', image_size, (255, 255, 255))
    draw = ImageDraw.Draw(img)
    
    # 动态调整缩放范围，防止溢出
    max_h_scale = min(h_scale_range[1], image_size[0] / (base_font_size * 0.8))  # 估算最大缩放
    max_v_scale = min(v_scale_range[1], image_size[1] / (base_font_size * 1.2))
    h_scale = random.uniform(h_scale_range[0], max_h_scale)
    v_scale = random.uniform(v_scale_range[0], max_v_scale)
    
    temp_size = (
        int(image_size[0] * h_scale),
        int(image_size[1] * v_scale)
    )
    
    # 创建临时层并绘制文字
    temp_img = Image.new('RGBA', temp_size, (255, 255, 2550))
    temp_draw = ImageDraw.Draw(temp_img)
    try:
        temp_draw.text(
            (temp_size[0]//2, temp_size[1]//2),
            str(digit),
            font=font,
            fill=(0, 0, 0, 255),
            anchor='mm'
        )
    except:  # 处理字符渲染错误
        print(f"字体 {font_path} 无法渲染数字 {digit}，已跳过。")
        fonts.remove(font_path)
        return
    
    # 计算合理的位置偏移
    max_x_offset = max(0, (image_size[0] - temp_size[0]) // 2)
    max_y_offset = max(0, (image_size[1] - temp_size[1]) // 2)
    pos = (
        max_x_offset + random.randint(-2, 2),
        max_y_offset + random.randint(-2, 2)
    )
    
    # 合成图像
    img.paste(temp_img, pos, temp_img)
    
    # 保存图像
    save_path = os.path.join(output_dir, str(digit), f"{digit}_{sample_idx:04d}.png")
    img.save(save_path)

if __name__ == '__main__':
    generate_custom_digits(
        font_dir='font',
        output_root='num',
        digits_range=(0, 9),
        total_samples=1000,
        image_size=(96,96),
        scale_factor=1.0,
        h_scale_range=(1, 1.5),
        v_scale_range=(1, 1.5)
    )