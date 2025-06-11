# Jupyter Notebook - 代码

# 导入必要的库
import matplotlib.pyplot as plt
import numpy as np
import os,shutil
import tensorflow as tf
import seaborn as sns
from tqdm import tqdm
import datetime

# 设定日志级别
tf.get_logger().setLevel('ERROR')

# 🔹 超参数
IMG_SIZE = (96, 96)
AUTOTUNE = tf.data.AUTOTUNE
IMG_SHAPE = IMG_SIZE + (3,)

# 检查缓存目录是否存在并删除
folder = 'cache'

if os.path.exists(folder) and os.path.isdir(folder):
    try:
        shutil.rmtree(folder)
        print(f"成功删除目录: {folder}")
    except Exception as e:
        print(f"删除失败，错误信息: {e}")
else:
    print(f"目录 '{folder}' 不存在")
    
BATCH_SIZE = 32

# 🔹 数据集路径
cache_dir = os.path.join('cache')
os.makedirs(cache_dir, exist_ok=True)

base_dir = '../data'
train_dir = os.path.join(base_dir, 'object')
valid_dir = os.path.join(base_dir, 'object')

# 🔹 加载数据集
train_dataset_raw = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir, 
    validation_split=0.2, 
    subset="training", 
    seed=12,
    batch_size=BATCH_SIZE, 
    image_size=IMG_SIZE)

validation_dataset_raw = tf.keras.preprocessing.image_dataset_from_directory(
    valid_dir, 
    validation_split=0.2, 
    subset="validation", 
    seed=12,
    batch_size=BATCH_SIZE, 
    image_size=IMG_SIZE)

class_names = train_dataset_raw.class_names
print("Class Names:", class_names)

# 生成 labels.txt 文件
labels_path = os.path.join(cache_dir, 'labels.txt')  # 保存到缓存目录
with open(labels_path, 'w') as f:
    for class_name in class_names:
        f.write(f"{class_name}\n")

print(f"标签文件已生成: {labels_path}")

# 预处理函数（添加增强）
def preprocess_image(image, label):
    return image, label

# 加载数据集
train_dataset = (train_dataset_raw
                 .map(preprocess_image, num_parallel_calls=AUTOTUNE)
                 .cache(os.path.join(cache_dir, 'train_cache1')) # 缓存到文件
                 .shuffle(1000)
                 .prefetch(AUTOTUNE))

validation_dataset = (validation_dataset_raw
                      .map(preprocess_image, num_parallel_calls=AUTOTUNE)
                      .cache(os.path.join(cache_dir, 'val_cache1'))
                      .prefetch(AUTOTUNE))

# 🔹 构建模型

base_model = tf.keras.applications.MobileNetV2(
    input_shape=IMG_SHAPE, 
    include_top=False, 
    pooling = 'avg', 
    alpha=0.35, 
    weights='imagenet')

model = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255),
    base_model,
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(len(class_names), activation='softmax')
])
model.build((None, 96, 96, 3))
model.summary()
# 编译模型

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.00001, decay_steps=len(train_dataset), decay_rate=0.99, staircase=True)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])

early_stopping = tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)

history = model.fit(train_dataset,
                    validation_data=validation_dataset,
                    epochs=100, 
                    callbacks=[early_stopping]
                    ) 

# 🔹 直接导出为TFLite格式 (无需保存H5)
def representative_dataset():
    # 从验证集取500个批次作为量化校准数据
    for images, _ in tqdm(validation_dataset.take(500), desc="Calibration"):
        yield [tf.cast(images, tf.float32)]  # 输入需为浮点型

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8   # 输入为uint8 (0-255)
converter.inference_output_type = tf.uint8  # 输出为uint8类别索引

tflite_model = converter.convert()

output_dir = './model/obj'
os.makedirs(output_dir, exist_ok=True)  # 自动创建目录（如果不存在）
# 保存带时间戳的TFLite模型
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
output_path = f'./model/obj/obj_model_{timestamp}.tflite'
with open(output_path, 'wb') as f:
    f.write(tflite_model)

print(f"TFLite模型已保存至: {output_path}")

target_dir = "model"
# 直接匹配当前目录下的 .h5 文件
for file in os.listdir(target_dir):
    if file.endswith(".h5"):
        file_path = os.path.join(target_dir, file)
        try:
            os.remove(file_path)
            print(f"已删除: {file_path}")
        except Exception as e:
            print(f"删除失败 [{file_path}]: {e}")
from sklearn.metrics import confusion_matrix
# 混淆矩阵
y_pred = np.argmax(model.predict(validation_dataset), axis=1)
y_true = np.concatenate([labels.numpy() for _, labels in validation_dataset])

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, cmap="Blues", fmt="d", 
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()
from lib import polt_improved

stage_names = ["history"]
history_list = [history]
polt_improved.plot_combined_curves_improved(history_list)