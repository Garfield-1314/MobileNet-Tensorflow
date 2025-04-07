import os
import random
import cv2
import numpy as np
from PIL import Image
from datetime import datetime
from albumentations import (
    Compose, HorizontalFlip, Rotate,
    RGBShift, RandomBrightnessContrast, MotionBlur,
    VerticalFlip, HueSaturationValue, ElasticTransform, OpticalDistortion
)

def find_images(root_dir):
    """递归查找所有子目录中的图片文件"""
    img_ext = ('.png', '.jpg', '.jpeg', '.webp')
    for dirpath, _, filenames in os.walk(root_dir):
        for f in filenames:
            if f.lower().endswith(img_ext):
                yield os.path.join(dirpath, f)

# 定义数据增强管道
augmentation_pipeline = Compose([
    HorizontalFlip(p=0.25),
    VerticalFlip(p=0.25),
    ElasticTransform(p=0.25,alpha=1, sigma=20),
    OpticalDistortion(p=0.25,distort_limit=0.1, shift_limit=0.1),
    Rotate(limit=45, p=0.5),
    RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=0.25),
    RandomBrightnessContrast(p=1,brightness_limit=(-0.25,0.25),contrast_limit=(-0.10,0.10)),
    HueSaturationValue( hue_shift_limit = (-10, 10),
                        sat_shift_limit = (-10, 10),
                        val_shift_limit = (-10, 10),
                        p=0.25),
    MotionBlur(p=0.25,blur_limit = 3),
])

def batch_overlay(
    backgrounds_dir=r'dataset\background',
    pics_root=r'dataset\stage2',
    output_root=r'dataset\stage3',
    min_scale=0.3,
    max_scale=1.7,
    min_visible=0.75,
    num_augments=3  # 新增增强次数参数
):
    # 获取所有背景和小图路径
    bg_paths = list(find_images(backgrounds_dir))
    pic_paths = list(find_images(pics_root))

    # 处理每个组合
    for bg_path in bg_paths:
        try:
            base_img = Image.open(bg_path).convert('RGBA')
            bg_w, bg_h = base_img.size
            bg_name = os.path.splitext(os.path.basename(bg_path))[0]
            
            for pic_path in pic_paths:
                # 计算输出路径
                rel_path = os.path.relpath(pic_path, pics_root)
                output_dir = os.path.join(output_root, os.path.dirname(rel_path))
                os.makedirs(output_dir, exist_ok=True)

                try:
                    for aug_idx in range(num_augments):
                        # 加载并处理小图
                        small_img = Image.open(pic_path).convert('RGBA')
                        
                        # 随机缩放
                        scale = random.uniform(min_scale, max_scale)
                        new_size = (int(small_img.width * scale), int(small_img.height * scale))
                        scaled_img = small_img.resize(new_size, Image.LANCZOS)
                        
                        # 随机旋转
                        angle = random.uniform(0, 360)
                        rotated_img = scaled_img.rotate(
                            angle,
                            expand=True,
                            resample=Image.BICUBIC,
                            fillcolor=(0, 0, 0, 0)
                        )
                        rw, rh = rotated_img.size
                        
                        # 智能定位
                        valid_pos = False
                        for _ in range(100):
                            x_min = max(-int(rw * 0.3), -rw + int(bg_w * 0.25))
                            x_max = min(bg_w - int(rw * 0.7), bg_w - int(rw * 0.25))
                            y_min = max(-int(rh * 0.3), -rh + int(bg_h * 0.25))
                            y_max = min(bg_h - int(rh * 0.7), bg_h - int(rh * 0.25))
                            
                            x = random.randint(x_min, x_max)
                            y = random.randint(y_min, y_max)
                            
                            visible_w = min(x+rw, bg_w) - max(x, 0)
                            visible_h = min(y+rh, bg_h) - max(y, 0)
                            if visible_w > 0 and visible_h > 0:
                                visible_area = visible_w * visible_h
                                if visible_area >= min_visible * rw * rh:
                                    valid_pos = True
                                    break

                        # 合成基础图像
                        composite = Image.new('RGBA', (bg_w, bg_h))
                        composite.paste(base_img, (0,0))
                        composite.alpha_composite(rotated_img, (x, y))
                        rgb_composite = composite.convert('RGB')
                        
                        # 转换为OpenCV格式
                        cv_image = cv2.cvtColor(np.array(rgb_composite), cv2.COLOR_RGB2BGR)
                        
                        # 生成多个增强版本
                        
                        # 应用数据增强
                        augmented = augmentation_pipeline(image=cv_image)
                        augmented_img = augmented['image']
                        
                        # 生成唯一文件名
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S%f")
                        pic_name = os.path.splitext(os.path.basename(pic_path))[0]
                        output_name = f"{bg_name}_{pic_name}_{timestamp}_aug{aug_idx}.jpg"
                        output_path = os.path.join(output_dir, output_name)
                        
                        # 保存增强后的图像
                        cv2.imwrite(output_path, augmented_img)
                        print(f"生成成功：{output_path}")

                except Exception as e:
                    print(f"处理失败：{pic_path} | 错误：{str(e)}")
                
        except Exception as e:
            print(f"背景图处理失败：{bg_path} | 错误：{str(e)}")

if __name__ == '__main__':
    batch_overlay(
        backgrounds_dir=r'dataset\background',
        pics_root=r'dataset\96',
        output_root=r'dataset\Au',
        min_scale=0.8,
        max_scale=1.2,
        min_visible=0.5,
        num_augments=100
    )