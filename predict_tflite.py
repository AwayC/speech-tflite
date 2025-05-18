import numpy as np
import tensorflow as tf

# 在原有代码基础上添加以下功能
import argparse

class TFLiteClassifier:
    def __init__(self, model_path):
        # 初始化解释器
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        
        # 获取输入输出详细信息
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # 从模型中提取类别数
        self.class_names = [f"class_{i}" for i in range(self.output_details[0]['shape'][-1])]

    def predict(self, input_data):
        processed = input_data.astype(np.float32).reshape(1, 80, 12)
        self.interpreter.set_tensor(self.input_details[0]['index'], processed)
        self.interpreter.invoke()
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        
        confidence = np.max(output_data)
        class_id = np.argmax(output_data)
        all_confidences = {  # 新增字典生成
            self.class_names[i]: float(prob) 
            for i, prob in enumerate(output_data[0])
        }
        
        return self.class_names[class_id], confidence, all_confidences

    def predict_from_file(self, npy_path):
        """从npy文件加载数据进行预测（新增方法）"""
        try:
            arr = np.load(npy_path)
            if arr.shape != (80, 12):
                raise ValueError(f"文件形状应为 (80,12)，实际为 {arr.shape}")
            return self.predict(arr)
        except Exception as e:
            print(f"错误：{str(e)}")
            return None, 0.0, {}

if __name__ == "__main__":
    # 添加命令行参数解析
    parser = argparse.ArgumentParser(description='TFLite模型预测')
    parser.add_argument('--file', type=str, required=True, help='要预测的.npy文件路径')
    args = parser.parse_args()
    
    classifier = TFLiteClassifier("speech_model_complete_float32.tflite")
    class_name, confidence, probs = classifier.predict_from_file(args.file)
    
    print(f"\n预测结果: {class_name}")
    print(f"置信度: {confidence:.4f}")
    print("所有类别概率:")
    for cls, prob in probs.items():
        print(f"  {cls}: {prob:.4f}")
    print(f"\n可信度判断: {'✅ 高可信度' if confidence > 0.8 else '⚠️ 需人工复核'}")