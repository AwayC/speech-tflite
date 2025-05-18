import numpy as np
import tensorflow as tf

# 配置参数（需与训练参数一致）
MAX_FRAMES = 80
NUM_MFCC_COEFFS = 12
MODEL_PATH = 'speech_model_complete.keras'

class AudioClassifier:
    def __init__(self):
        self.model = tf.keras.models.load_model(MODEL_PATH)
        self.class_names = ['class_0', 'class_1', 'class_2']
        
    def predict_from_file(self, npy_path):
        """从npy文件加载80x12数组进行预测"""
        try:
            arr = np.load(npy_path)
            if arr.shape != (80, 12):
                raise ValueError(f"文件 {npy_path} 数据形状应为 (80,12)，实际为 {arr.shape}")
            # print(arr)
            return self._predict(arr)
        except FileNotFoundError:
            print(f"错误：文件 {npy_path} 不存在")
            return None, 0.0
            
    def _predict(self, arr):
        """内部预测方法"""
        processed = arr.astype(np.float32).reshape(1, 80, 12)
        predictions = self.model.predict(processed, verbose=0)
        
        # 修复：生成包含所有类别概率的字典
        all_confidences = {name: float(prob) for name, prob in zip(self.class_names, predictions[0])}
        confidence = np.max(predictions[0])
        class_id = np.argmax(predictions[0])
        return self.class_names[class_id], confidence, all_confidences  # 修正返回顺序

# 使用示例
if __name__ == '__main__':
    import argparse
    
    # 配置命令行解析
    parser = argparse.ArgumentParser(description='语音分类预测工具')
    parser.add_argument('--file', type=str, required=True, 
                       help='需要预测的.npy文件路径')  
    args = parser.parse_args()

    classifier = AudioClassifier()
    class_name, top_confidence, all_confidences = classifier.predict_from_file(args.file)
    
    
    if class_name is not None:
        print(f"\n预测结果类别: {class_name}")
        print(f"最高置信度: {top_confidence:.2%}")
        print("全部分类置信度:")
        for cls, prob in all_confidences.items():
            print(f"  {cls}: {prob:.2%}")
        print(f"\n可信度判断: {'✅ 高可信度' if top_confidence > 0.8 else '⚠️ 需人工复核'}")