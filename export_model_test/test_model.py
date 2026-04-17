import onnxruntime as ort
import tensorflow as tf
import numpy as np
from PIL import Image

# 1. 准备测试输入 (NHWC format for TFLite)
# 创建随机数据或读取真实图片
dummy_input_nhwc = np.random.rand(1, 640, 640, 3).astype(np.float32)
# YOLO 通常需要归一化到 0-1
dummy_input_nhwc = dummy_input_nhwc / 255.0

# 转换为 NCHW for ONNX
dummy_input_nchw = np.transpose(dummy_input_nhwc, (0, 3, 1, 2))

print(f"ONNX input shape (NCHW): {dummy_input_nchw.shape}")
print(f"TFLite input shape (NHWC): {dummy_input_nhwc.shape}")

# 2. ONNX 推理
print("\n--- Testing ONNX ---")
onnx_session = ort.InferenceSession("best_mysuccessfully.onnx")
input_name = onnx_session.get_inputs()[0].name
onnx_out = onnx_session.run(None, {input_name: dummy_input_nchw})[0]
print(f"ONNX output shape: {onnx_out.shape}")

# 3. TFLite 推理
print("\n--- Testing TFLite ---")
interpreter = tf.lite.Interpreter(model_path="best_mysuccessfully_saved_model/best_mysuccessfully_float32.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(f"TFLite input details: {input_details[0]['shape']}")
print(f"TFLite input dtype: {input_details[0]['dtype']}")

interpreter.set_tensor(input_details[0]['index'], dummy_input_nhwc)
interpreter.invoke()
tflite_out = interpreter.get_tensor(output_details[0]['index'])
print(f"TFLite output shape: {tflite_out.shape}")

# 4. 对比输出 (注意输出维度可能也需要转置)
print("\n--- Comparison ---")
# 如果输出维度不同，尝试转置
if onnx_out.shape != tflite_out.shape:
    print(f"Shape mismatch, attempting transpose...")
    # 尝试将 ONNX 输出从 NCHW 转为 NHWC 或反之
    if len(onnx_out.shape) == 4:
        onnx_out_t = np.transpose(onnx_out, (0, 2, 3, 1))
        print(f"ONNX transposed shape: {onnx_out_t.shape}")
        diff = np.abs(onnx_out_t - tflite_out).max()
    else:
        diff = np.abs(onnx_out - tflite_out).max()
else:
    diff = np.abs(onnx_out - tflite_out).max()

print(f"Max absolute difference: {diff:.6f}")
print(f"Mean absolute difference: {np.mean(np.abs(onnx_out - tflite_out)):.6f}")

if diff < 1e-4:
    print("✅ 精度对齐良好，Slice 警告未影响结果")
elif diff < 1e-2:
    print("⚠️  存在微小差异，可能在可接受范围内")
else:
    print("❌ 差异较大，需要检查模型转换")