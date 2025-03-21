# Jupyter Notebook - 代码

# 导入必要的库
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import tensorflow_model_optimization as tfmot  # 新增剪枝库
import datetime
# 设定日志级别
tf.get_logger().setLevel('ERROR')

# 🔹 超参数
IMG_SIZE = (128, 128)
AUTOTUNE = tf.data.AUTOTUNE
# -------------------------------- 第一阶段训练 ---------------------------------
# 🔹 数据集路径
base_dir = './dataset'
train_dir = os.path.join(base_dir, '80')
valid_dir = os.path.join(base_dir, '80')

BATCH_SIZE = 128
# 🔹 加载数据集
train_dataset_raw = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir, validation_split=0.2, subset="training", seed=12,
    batch_size=BATCH_SIZE, image_size=IMG_SIZE)

validation_dataset_raw = tf.keras.preprocessing.image_dataset_from_directory(
    valid_dir, validation_split=0.2, subset="validation", seed=12,
    batch_size=BATCH_SIZE, image_size=IMG_SIZE)

class_names = train_dataset_raw.class_names
print("Class Names:", class_names)

# 预处理函数
def preprocess_image(image, label):
    return image, label

# 加载数据集
train_dataset = (train_dataset_raw
                 .map(preprocess_image, num_parallel_calls=AUTOTUNE)
                 .cache()
                 .shuffle(1000)
                 .prefetch(AUTOTUNE))

validation_dataset = (validation_dataset_raw
                      .map(preprocess_image, num_parallel_calls=AUTOTUNE)
                      .cache()
                      .prefetch(AUTOTUNE))

# 🔹 构建模型
IMG_SHAPE = IMG_SIZE + (3,)
base_model = tf.keras.applications.MobileNetV2(
    input_shape=IMG_SHAPE, include_top=False, pooling = 'avg', alpha=0.35, weights='imagenet')

# 冻结除最后4层外的所有层
base_model.trainable = True
# for layer in base_model.layers[:-0]:
#     layer.trainable = False

model = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255),
    base_model,
    # tf.keras.layers.GlobalAveragePooling2D(),
    # tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(len(class_names), activation='softmax')
])
model.build((None, 128, 128, 3))
model.summary()
# 编译模型
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.00001, decay_steps=1000, decay_rate=0.90, staircase=True)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])

# 训练第一阶段
early_stopping = tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)

history_stage1 = model.fit(train_dataset,
                    validation_data=validation_dataset,
                    epochs=100, 
                    callbacks=[early_stopping]
                    )

# 保存第一阶段模型
# model.save('./model/stage1_model.h5')
     
# -------------------------------- 第二阶段训练 ---------------------------------
# 🔹 数据集路径
base_dir = './dataset'
train_dir = os.path.join(base_dir, '80')
valid_dir = os.path.join(base_dir, '80')

BATCH_SIZE = 128

# 加载数据集
train_dataset_raw = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir, validation_split=0.2, subset="training", seed=12,
    batch_size=BATCH_SIZE, image_size=IMG_SIZE)

validation_dataset_raw = tf.keras.preprocessing.image_dataset_from_directory(
    valid_dir, validation_split=0.2, subset="validation", seed=12,
    batch_size=BATCH_SIZE, image_size=IMG_SIZE)

# 数据预处理（同第一阶段）
train_dataset = (train_dataset_raw
                 .map(preprocess_image, num_parallel_calls=AUTOTUNE)
                 .cache()
                 .shuffle(1000)
                 .prefetch(AUTOTUNE))

validation_dataset = (validation_dataset_raw
                      .map(preprocess_image, num_parallel_calls=AUTOTUNE)
                      .cache()
                      .prefetch(AUTOTUNE))

early_stopping = tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
# 加载第一阶段模型
# model = tf.keras.models.load_model('./model/stage1_model.h5')
# 编译模型
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.00001, decay_steps=1000, decay_rate=0.90, staircase=True)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])

# 训练第一阶段
early_stopping = tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)

history_stage2 = model.fit(train_dataset, validation_data=validation_dataset,
                    epochs=100, callbacks=[early_stopping])
# 保存第一阶段模型
# model.save('./model/stage2_model.h5')
# -------------------------------- 第三阶段训练 ---------------------------------
# 🔹 剪枝参数配置
PRUNING_PARAMS = {
    'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
        initial_sparsity=0.30,
        final_sparsity=0.60,
        begin_step=0,
        end_step=2000,
        frequency=100
    )
}

# 加载第二阶段模型
# model = tf.keras.models.load_model('./model/stage2_model.h5')

# 🔹 分离 Rescaling 层和基础模型
rescale_layer = model.layers[0]  # 提取 Rescaling 层    
prunable_model = tf.keras.Sequential(model.layers[1:])  # 排除 Rescaling 后的模型

# 🔹 应用剪枝到可剪枝部分
with tfmot.sparsity.keras.prune_scope():  # 确保剪枝作用域正确
    pruned_model = tfmot.sparsity.keras.prune_low_magnitude(
        prunable_model, **PRUNING_PARAMS
    )

# 🔹 重新组合模型
final_model = tf.keras.Sequential([
    rescale_layer,  # 前置 Rescaling
    pruned_model    # 剪枝后的模型部分
])

# 编译模型
final_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy']
)
# 🔹 数据增强
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(factor=(-0.125,0.125),fill_mode="nearest"),
    tf.keras.layers.RandomZoom(0.25,fill_mode="nearest"),
    tf.keras.layers.RandomTranslation(height_factor=0.25, width_factor=0.25),
    tf.keras.layers.RandomBrightness(0.25),
    tf.keras.layers.RandomContrast(0.3)
])

# 预处理函数（添加增强）
def preprocess_image_aug(image, label):
    image = data_augmentation(image)
    return image, label
# 加载数据集（使用新数据集）
base_dir = './dataset'
train_dir = os.path.join(base_dir, '80')
valid_dir = os.path.join(base_dir, '80')

BATCH_SIZE = 64
train_dataset_raw = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir, validation_split=0.2, subset="training", seed=12,
    batch_size=BATCH_SIZE, image_size=IMG_SIZE)

validation_dataset_raw = tf.keras.preprocessing.image_dataset_from_directory(
    valid_dir, validation_split=0.2, subset="validation", seed=12,
    batch_size=BATCH_SIZE, image_size=IMG_SIZE)

# 数据预处理（应用增强）
train_dataset = (train_dataset_raw
                 .map(preprocess_image_aug, num_parallel_calls=AUTOTUNE)
                 .cache()
                 .shuffle(1000)
                 .prefetch(AUTOTUNE))

validation_dataset = (validation_dataset_raw
                      .map(preprocess_image, num_parallel_calls=AUTOTUNE)
                      .cache()
                      .prefetch(AUTOTUNE))
# 🔹 添加剪枝回调
pruning_callbacks = [
    tfmot.sparsity.keras.UpdatePruningStep(),
    tfmot.sparsity.keras.PruningSummaries(log_dir='./logs_pruning')
]
# 训练第三阶段
history_stage3 = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=100,
    callbacks=[early_stopping, pruning_callbacks]
)

# 去除剪枝包装
final_model = tfmot.sparsity.keras.strip_pruning(model)
final_model.save('./model/stage3_pruned_final.h5')
# 加载剪枝后的模型
model = tf.keras.models.load_model('./model/stage3_pruned_final.h5')

def representative_dataset():
    for image_batch, _ in tqdm(validation_dataset_raw.take(500), desc="Processing"):
        yield [tf.cast(image_batch, tf.float32)]

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8
tflite_model_quant = converter.convert()

# ---- 4. 动态生成带时间的文件名 ----
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
output_tflite_path = f'./model/model_{timestamp}.tflite'  # 新文件名格式

with open(output_tflite_path, 'wb') as f:
    f.write(tflite_model_quant)

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
from librariy import polt_improved

stage_names = ["Stage 1", "Stage 2", "Stage 3 (Pruning)"]
history_list = [history_stage1, history_stage2, history_stage3]
polt_improved.plot_combined_curves_improved(history_list)