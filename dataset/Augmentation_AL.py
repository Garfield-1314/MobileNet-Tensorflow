import os
import cv2
from albumentations import (
    Compose, HorizontalFlip, Rotate,
    RGBShift, RandomBrightnessContrast,MotionBlur,VerticalFlip,HueSaturationValue,ElasticTransform,OpticalDistortion
)
from tqdm import tqdm

# 配置参数
input_dir = r"dataset\224"        # 输入图片根目录（包含子文件夹）
output_dir = r"dataset\224_au"   # 输出图片根目录
num_augments = 10                  # 每张图片生成多少个增强版本

# 创建输出目录
os.makedirs(output_dir, exist_ok=True)

# 定义数据增强管道
augmentation_pipeline = Compose([
    HorizontalFlip(p=0.5),
    VerticalFlip(p=0.5),
    ElasticTransform(),
    OpticalDistortion(),
    Rotate(limit=45, p=0.5),
    RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.2),
    RandomBrightnessContrast(p=0.3,brightness_limit=(-0.25,0.25),contrast_limit=(-0.25,0.25)),
    HueSaturationValue(),
    MotionBlur(p=0.5),

])

# 支持的图片格式
extensions = ['.jpg', '.jpeg', '.png']

# 使用os.walk递归遍历所有子目录
for root, dirs, files in os.walk(input_dir):
    # 计算相对路径以便重建输出目录结构
    relative_path = os.path.relpath(root, input_dir)
    current_output_dir = os.path.join(output_dir, relative_path)
    
    # 创建当前层级的输出目录
    os.makedirs(current_output_dir, exist_ok=True)
    
    # 处理当前目录下的所有文件
    for filename in tqdm(files, desc=f"Processing {relative_path}"):
        # 过滤非图片文件
        if not any(filename.lower().endswith(ext) for ext in extensions):
            continue
        
        # 读取图片
        img_path = os.path.join(root, filename)
        image = cv2.imread(img_path)
        
        # 保存原始图片（可选）
        # original_output = os.path.join(current_output_dir, f"original_{filename}")
        # cv2.imwrite(original_output, image)
        
        # 生成多个增强版本
        for i in range(num_augments):
            augmented = augmentation_pipeline(image=image)
            augmented_img = augmented['image']
            
            # 构建增强后的文件名
            aug_filename = f"{os.path.splitext(filename)[0]}_aug{i+1}{os.path.splitext(filename)[1]}"
            output_path = os.path.join(current_output_dir, aug_filename)
            
            # 保存增强后的图片
            cv2.imwrite(output_path, augmented_img)