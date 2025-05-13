# TensorFlow Lite 转换 (在训练脚本之后或单独脚本中)
# converter = tf.lite.TFLiteConverter.from_keras_model(model) # 如果从Keras模型对象转换
import tensorflow as tf
import numpy as np
# 加载训练好的模型
SAVED_MODEL_PATH = 'speech_model_complete.keras' # 替换为你的模型保存路径
converter = tf.lite.TFLiteConverter.from_saved_model(SAVED_MODEL_PATH)

# (可选) 开启优化，例如量化 (Post-training quantization)
# converter.optimizations = [tf.lite.Optimize.DEFAULT]
# def representative_dataset_gen(): # 需要一个代表性数据集用于量化校准
#     for i in range(100): # 使用部分训练数据
#         yield [X_train[i:i+1].astype(np.float32)]
# converter.representative_dataset = representative_dataset_gen
# converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8] # INT8量化
# converter.inference_input_type = tf.int8 # or tf.uint8
# converter.inference_output_type = tf.int8 # or tf.uint8

tflite_model = converter.convert()

with open('speech_model.tflite', 'wb') as f:
    f.write(tflite_model)

print("TensorFlow Lite 模型已保存为 speech_model.tflite")