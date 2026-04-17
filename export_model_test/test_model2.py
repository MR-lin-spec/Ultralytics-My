import tensorflow as tf
import numpy as np
from collections import defaultdict
import json

def analyze_tflite_strided_slice(model_path="/root/Ultralytics-My/best_mysuccessfully_saved_model/best_mysuccessfully_float16.tflite"):
    """
    分析 TFLite 模型中的 StridedSlice 分布，特别是 cross_attn 部分
    """
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    
    # 获取模型图结构
    try:
        # 新版 TF 接口
        graph = interpreter._graph_conversion_info
    except:
        # 通过 model 对象获取
        model = interpreter._model
    
    # 获取张量和算子信息
    tensor_details = interpreter.get_tensor_details()
    op_details = interpreter.get_node_details()
    
    # 构建索引映射
    tensor_idx_to_name = {t['index']: t['name'] for t in tensor_details}
    
    # 统计 StridedSlice
    strided_slice_ops = []
    cross_attn_slices = []
    
    # 遍历所有算子
    for i in range(len(interpreter._nodes)):
        node = interpreter._nodes[i]
        op_name = node.name.decode('utf-8') if isinstance(node.name, bytes) else node.name
        
        # 检查是否为 StridedSlice (BuiltinOptions 是 45)
        if node.builtin_options == 45 or 'strided_slice' in node.name.lower():
            op_info = {
                'index': i,
                'name': op_name,
                'inputs': [tensor_idx_to_name.get(idx, f"tensor_{idx}") for idx in node.inputs],
                'outputs': [tensor_idx_to_name.get(idx, f"tensor_{idx}") for idx in node.outputs],
                'is_cross_attn': 'cross_attn' in op_name.lower() or 'decoder' in op_name.lower()
            }
            strided_slice_ops.append(op_info)
            
            if op_info['is_cross_attn']:
                cross_attn_slices.append(op_info)
    
    # 生成报告
    print("=" * 80)
    print(f"TFLite 模型 StridedSlice 分析报告")
    print("=" * 80)
    print(f"模型路径: {model_path}")
    print(f"总 StridedSlice 数量: {len(strided_slice_ops)}")
    print(f"cross_attn 相关 StridedSlice: {len(cross_attn_slices)}")
    print(f"其他 StridedSlice: {len(strided_slice_ops) - len(cross_attn_slices)}")
    print()
    
    # 详细列出 cross_attn 部分
    if cross_attn_slices:
        print("⚠️  发现 cross_attn 相关 StridedSlice (这些可能受之前警告影响):")
        print("-" * 80)
        for op in cross_attn_slices:
            print(f"算子索引: {op['index']}")
            print(f"  名称: {op['name']}")
            print(f"  输入: {op['inputs']}")
            print(f"  输出: {op['outputs']}")
            print()
    
    # 按层统计 (解析层号)
    layer_stats = defaultdict(int)
    for op in cross_attn_slices:
        # 尝试提取层号 (如 layers.0, layers.1)
        name = op['name']
        if 'layers.' in name:
            try:
                layer_idx = name.split('layers.')[1].split('/')[0]
                layer_stats[f"layer_{layer_idx}"] += 1
            except:
                layer_stats["unknown_layer"] += 1
    
    if layer_stats:
        print("按层分布统计:")
        for layer, count in sorted(layer_stats.items()):
            print(f"  {layer}: {count} 个 StridedSlice")
        print()
    
    # 检查 Flex 算子 (与之前警告相关)
    flex_ops = []
    try:
        for i in range(len(interpreter._nodes)):
            node = interpreter._nodes[i]
            # Flex 算子的特征
            if node.builtin_options == 0 and len(node.custom_options) > 0:
                op_name = node.name.decode('utf-8') if isinstance(node.name, bytes) else node.name
                flex_ops.append({
                    'index': i,
                    'name': op_name,
                    'custom_options': node.custom_options[:50]  # 前50字节
                })
    except:
        pass
    
    if flex_ops:
        print(f"🔴 发现 {len(flex_ops)} 个 Flex 算子 (需 TensorFlow Select 支持):")
        for op in flex_ops[:5]:  # 只显示前5个
            print(f"  - {op['name']}")
        if len(flex_ops) > 5:
            print(f"  ... 还有 {len(flex_ops)-5} 个")
        print()
    
    # 保存详细报告到 JSON
    report = {
        'total_strided_slice': len(strided_slice_ops),
        'cross_attn_strided_slice': len(cross_attn_slices),
        'layer_distribution': dict(layer_stats),
        'cross_attn_details': cross_attn_slices,
        'flex_ops_count': len(flex_ops)
    }
    
    report_path = model_path.replace('.tflite', '_strided_slice_report.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"✅ 详细报告已保存: {report_path}")
    
    # 给出硬件部署建议
    print("\n" + "=" * 80)
    print("硬件部署建议:")
    print("=" * 80)
    
    if len(cross_attn_slices) > 8:  # 你之前有8个警告
        print("⚠️  警告: cross_attn 部分存在大量 StridedSlice (>8个)")
        print("   这对应之前 onnx2tf 的 'Dimensional compression fails' 警告")
        print()
        
    if flex_ops:
        print("🔴 严重: 模型包含 Flex 算子，以下硬件可能不支持:")
        print("   - 纯 TFLite Micro (MCU)")
        print("   - Edge TPU (Coral)")
        print("   - 大多数专用 NPU (Rockchip, Amlogic等)")
        print("   这些算子将回退到 CPU 执行，导致性能下降")
        print()
    
    if len(cross_attn_slices) > 0 and not flex_ops:
        print("🟡 注意: StridedSlice 数量较多，但都是标准 TFLite 算子")
        print("   - 在 GPU/CPU 上运行正常")
        print("   - 部分 NPU 可能对 StridedSlice 优化不佳")
        print("   建议在实际硬件上测试推理延迟")
    
    return report

if __name__ == "__main__":
    # 修改为你的模型路径
    model_path = "/root/Ultralytics-My/best_mysuccessfully_saved_model/best_mysuccessfully_float16.tflite"
    
    try:
        analyze_tflite_strided_slice(model_path)
    except Exception as e:
        print(f"分析出错: {e}")
        print("尝试备用分析方法...")
        
        # 备用方法：使用更简单的方式统计
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        
        # 获取操作详情
        print("\n备用分析结果:")
        print(f"输入张量: {interpreter.get_input_details()}")
        print(f"输出张量: {interpreter.get_output_details()}")
        
        # 尝试打印所有算子名
        try:
            print("\n模型中的所有算子:")
            for i, node in enumerate(interpreter._nodes):
                print(f"  {i}: {node.name}")
        except:
            print("无法获取详细算子列表，请直接用 Netron 打开模型文件查看")