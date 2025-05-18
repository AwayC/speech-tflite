import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input, Conv1D, MaxPooling1D, GlobalAveragePooling1D, Dense, BatchNormalization, Dropout
)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import os
import shutil # 用于删除旧的示例数据目录
import glob
import matplotlib.pyplot as plt

# --- A. 配置参数 ---
# 数据相关
NUM_MFCC_COEFFS = 12        # MFCC系数数量 (为示例简化)
MAX_FRAMES = 80             # MFCC序列的最大帧数 (为示例简化)
NUM_CLASSES = 5            # 要识别的类别数量
NUM_SAMPLES_PER_CLASS = 100   # 每个类别生成的样本数量 (为示例简化)
TOTAL_SAMPLES = NUM_CLASSES * NUM_SAMPLES_PER_CLASS

# 训练相关
BATCH_SIZE = 20
EPOCHS = 40           # 训练轮数 (为示例简化)
LEARNING_RATE = 0.0005
DROPOUT_RATE = 0.3         # 新增正则化参数
WEIGHT_DECAY = 1e-4        # 新增权重衰减
VALIDATION_SPLIT = 0.15      # 从训练数据中分出验证集的比例

# 路径相关
BASE_DATA_DIR = 'complete_npy_dataset' # 保存.npy文件的根目录
SAVED_KERAS_MODEL_PATH = 'speech_model_complete.keras' # Keras模型保存路径 (新的.keras格式)
SAVED_TFLITE_MODEL_PATH = 'speech_model_complete_float32.tflite'


# --- B. 生成并保存 .npy 样例数据 ---
def generate_and_save_npy_samples(base_dir, num_classes, samples_per_class, num_mfcc, max_frames_limit):
    """生成模拟的MFCC数据并按类别保存为.npy文件"""
    if os.path.exists(base_dir):
        print(f"旧的数据目录 '{base_dir}' 已存在，将删除并重新创建。")
        shutil.rmtree(base_dir) # 删除旧目录
    os.makedirs(base_dir, exist_ok=True)
    print(f"正在生成并保存 .npy 样本到 '{base_dir}'...")

    all_generated_features = []
    all_generated_labels = []

    for class_idx in range(num_classes):
        class_path = os.path.join(base_dir, f"class_{class_idx}")
        os.makedirs(class_path, exist_ok=True)

        for sample_idx in range(samples_per_class):
            # 模拟可变长度的MFCC序列，但通常不超过max_frames_limit太多
            num_actual_frames = np.random.randint(int(max_frames_limit * 0.5), int(max_frames_limit * 1.2))
            # 生成随机MFCC特征
            mfcc_features = np.random.rand(num_actual_frames, num_mfcc).astype(np.float32) * 10 - 5

            # 保存为 .npy 文件 (保存原始长度)
            file_path = os.path.join(class_path, f"sample_{sample_idx:03d}.npy")
            np.save(file_path, mfcc_features)

            # (可选) 收集用于直接演示，但主要目的是展示文件保存和加载
            all_generated_features.append(mfcc_features) # 保存原始特征，后面加载时处理padding
            all_generated_labels.append(class_idx)

    print(f"总共 {num_classes * samples_per_class} 个 .npy 样本已生成并保存。")
    return all_generated_features, all_generated_labels


# --- C. 从 .npy 文件加载数据 ---
def load_data_from_npy_files(base_dir, num_classes, max_frames_model, num_mfcc_model):
    """从按类别组织的.npy文件加载MFCC数据，并进行填充/截断。"""
    all_features = []
    all_labels = []
    class_names = [f"class_{i}" for i in range(num_classes)]

    print(f"正在从 '{base_dir}' 加载 .npy 文件...")
    for class_idx, class_name in enumerate(class_names):
        class_path = os.path.join(base_dir, class_name)
        if not os.path.isdir(class_path):
            print(f"警告: 类别目录 '{class_path}' 未找到。")
            continue

        for npy_file in sorted(glob.glob(os.path.join(class_path, "*.npy"))): # sorted确保顺序一致性
            mfcc = np.load(npy_file)

            if mfcc.ndim != 2 or mfcc.shape[1] != num_mfcc_model:
                print(f"警告: 文件 '{npy_file}' 格式不正确 (shape: {mfcc.shape})，期望shape (*, {num_mfcc_model})。已跳过。")
                continue

            # 填充或截断至 max_frames_model
            if mfcc.shape[0] < max_frames_model: # Padding
                pad_width = max_frames_model - mfcc.shape[0]
                processed_mfcc = np.pad(mfcc, ((0, pad_width), (0, 0)), mode='constant', constant_values=0.0)
            else: # Truncating
                processed_mfcc = mfcc[:max_frames_model, :]

            all_features.append(processed_mfcc)
            all_labels.append(class_idx)

    if not all_features:
        print("错误: 未能从 .npy 文件加载任何数据。请检查数据目录和文件结构。")
        return None, None

    X = np.array(all_features).astype(np.float32)
    y = np.array(all_labels).astype(np.int32)

    print(f"从 .npy 文件加载并处理了 {X.shape[0]} 个样本。")
    print(f"特征数据 X 的形状: {X.shape}") # (num_samples, max_frames_model, num_mfcc_model)
    print(f"标签数据 y 的形状: {y.shape}")   # (num_samples,)
    return X, y


# --- D. 模型定义 ---
def build_model(input_shape, num_classes_model):
    """构建一个简单的CNN模型"""
    model = Sequential([
        Input(shape=input_shape, name='input_mfcc'),
        Conv1D(filters=16, kernel_size=5, activation='relu', padding='same',
               kernel_regularizer=tf.keras.regularizers.l2(WEIGHT_DECAY*2)),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.1),

        Conv1D(filters=32, kernel_size=3, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(DROPOUT_RATE),

        GlobalAveragePooling1D(),
        Dense(16, activation='relu',
              kernel_regularizer=tf.keras.regularizers.l2(WEIGHT_DECAY)),
        Dropout(0.5),
        Dense(num_classes_model, activation='softmax', name='output_dense')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


# --- E. Main 执行块 ---
if __name__ == '__main__':
    # 0. Mac M-series GPU check (可选)
    print("TensorFlow 版本:", tf.__version__)
    gpu_devices = tf.config.list_physical_devices('GPU')
    if gpu_devices:
        print("GPU 设备可用:", gpu_devices)
        try:
            for gpu in gpu_devices: # 针对所有找到的GPU设置内存增长
                 tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print("设置内存增长时出错 (可能已初始化):", e)
    else:
        print("未找到GPU设备，将使用CPU进行训练。")

    # 1. 生成并保存 .npy 样例数据
    # 注意：generate_and_save_npy_samples 返回的 features/labels 在这里仅用于演示，
    # 实际训练将从文件加载。
    # _, _ = generate_and_save_npy_samples(
    #     BASE_DATA_DIR,
    #     NUM_CLASSES,
    #     NUM_SAMPLES_PER_CLASS,
    #     NUM_MFCC_COEFFS,
    #     MAX_FRAMES
    # )

    # 2. 从 .npy 文件加载数据
    X_data, y_labels_int = load_data_from_npy_files(
        BASE_DATA_DIR,
        NUM_CLASSES,
        MAX_FRAMES,
        NUM_MFCC_COEFFS
    )

    if X_data is None or y_labels_int is None:
        print("数据加载失败，脚本终止。")
        exit()

    # 3. 预处理加载的数据
    # 将标签转换为独热编码
    y_data_one_hot = to_categorical(y_labels_int, num_classes=NUM_CLASSES)

    # 打乱数据并分割训练集/验证集 (简单实现)
    indices = np.arange(X_data.shape[0])
    np.random.shuffle(indices)
    X_data_shuffled = X_data[indices]
    y_data_one_hot_shuffled = y_data_one_hot[indices]

    num_validation_samples = int(VALIDATION_SPLIT * X_data.shape[0])
    if num_validation_samples < 1 and X_data.shape[0] > 1 : # 确保至少有一个验证样本（如果总样本>1）
        num_validation_samples = 1
    if num_validation_samples == X_data.shape[0]: # 避免所有数据都做验证
        num_validation_samples = X_data.shape[0] -1 if X_data.shape[0] > 0 else 0


    if num_validation_samples > 0 :
        X_train = X_data_shuffled[:-num_validation_samples]
        y_train = y_data_one_hot_shuffled[:-num_validation_samples]
        X_val = X_data_shuffled[-num_validation_samples:]
        y_val = y_data_one_hot_shuffled[-num_validation_samples:]
        print(f"训练集大小: {X_train.shape[0]}, 验证集大小: {X_val.shape[0]}")
        validation_data_for_fit = (X_val, y_val)
    else: # 如果样本太少，不使用验证集
        X_train = X_data_shuffled
        y_train = y_data_one_hot_shuffled
        print(f"训练集大小: {X_train.shape[0]}, 由于样本过少，不使用验证集。")
        validation_data_for_fit = None


    # 4. 构建并训练模型
    input_shape_model = (MAX_FRAMES, NUM_MFCC_COEFFS)
    model = build_model(input_shape_model, NUM_CLASSES)
    model.summary()

    print("\n开始训练模型...")
    early_stopping = EarlyStopping(monitor='val_loss' if validation_data_for_fit else 'loss', patience=10, restore_best_weights=True, verbose=1)

    history = model.fit(X_train, y_train,
                        epochs=EPOCHS,
                        batch_size=BATCH_SIZE,
                        validation_data=validation_data_for_fit,
                        callbacks=[early_stopping],
                        verbose=1) # verbose=1 显示进度条

    print("\n训练完成!")

    # 5. 评估模型 (可选，如果使用了验证集)
    if validation_data_for_fit:
        val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
        print(f"模型在验证集上的最终损失: {val_loss:.4f}")
        print(f"模型在验证集上的最终准确率: {val_accuracy:.4f}")

    # 6. 绘制训练曲线
    print("\n绘制训练历史曲线...")
    if history is not None and history.history:
        acc = history.history.get('accuracy')
        val_acc = history.history.get('val_accuracy') # .get() 避免KeyError
        loss = history.history.get('loss')
        val_loss = history.history.get('val_loss')
        epochs_range = range(len(acc) if acc else 0)

        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        if acc: plt.plot(epochs_range, acc, label='Training Accuracy')
        if val_acc: plt.plot(epochs_range, val_acc, label='Validation Accuracy')
        if acc or val_acc: plt.legend(loc='lower right')
        plt.title('Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')

        plt.subplot(1, 2, 2)
        if loss: plt.plot(epochs_range, loss, label='Training Loss')
        if val_loss: plt.plot(epochs_range, val_loss, label='Validation Loss')
        if loss or val_loss: plt.legend(loc='upper right')
        plt.title('Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')

        plt.tight_layout()
        plt.show()
    else:
        print("未能获取训练历史，无法绘制曲线。")

    # 7. 保存训练好的 Keras 模型
    print(f"\n保存 Keras 模型到: {SAVED_KERAS_MODEL_PATH}")
    model.save(SAVED_KERAS_MODEL_PATH)
    print("Keras 模型已保存。")

    # 8. 转换为 TensorFlow Lite 模型 (Float32)
    print(f"\n将 Keras 模型转换为 TensorFlow Lite (Float32) 模型...")
    try:
        converter = tf.lite.TFLiteConverter.from_keras_model(model) #直接从模型对象转换
        # 或者从已保存的 .keras 文件加载并转换:
        # loaded_model = tf.keras.models.load_model(SAVED_KERAS_MODEL_PATH)
        # converter = tf.lite.TFLiteConverter.from_keras_model(loaded_model)

        tflite_model_float32 = converter.convert()
        with open(SAVED_TFLITE_MODEL_PATH, 'wb') as f:
            f.write(tflite_model_float32)
        print(f"Float32 TFLite 模型已保存为 {SAVED_TFLITE_MODEL_PATH} (大小: {len(tflite_model_float32)/1024:.2f} KB)")
        print("   - 这是基础的TFLite模型，后续可进行量化以进一步减小体积。")
    except Exception as e:
        print(f"转换为 TFLite 失败: {e}")

    print("\n--- 完整示例脚本执行完毕 ---")