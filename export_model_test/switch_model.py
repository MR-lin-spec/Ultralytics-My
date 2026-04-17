from ultralytics import RTDETR
import onnx
from onnx_tf.backend import prepare
import tensorflow as tf

# 第一步：导出 ONNX（使用低版本 opset）

# 第二步：ONNX -> TensorFlow SavedModel -> TFLite
onnx_model = onnx.load("/root/Ultralytics-My/best_mysuccessfully.onnx")
tf_rep = prepare(onnx_model)  # 需要 onnx-tf
tf_rep.export_graph("saved_model")

# 第三步：转换为 TFLite
converter = tf.lite.TFLiteConverter.from_saved_model("saved_model")
tflite_model = converter.convert()
with open("best.tflite", "wb") as f:
    f.write(tflite_model)