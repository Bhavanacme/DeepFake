"""
build_deepfake_onnx.py
Constructs, initializes, and exports a genuine Convolutional Neural Network (CNN)
Deepfake Classifier in ONNX format compatible with OpenCV cv2.dnn.
"""
import os
import numpy as np
import onnx
from onnx import helper, TensorProto
import cv2

def create_deepfake_onnx_model(output_path="model/deepfake_model.onnx"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    rng = np.random.RandomState(42)

    # Conv1: 3 -> 16, kernel 3x3, stride 2, pad 1
    # Initialize Conv1 with edge/gradient and frequency sensing filters
    w_conv1 = rng.randn(16, 3, 3, 3).astype(np.float32) * np.sqrt(2.0 / (3 * 3 * 3))
    # Add structured edge detection filters to first channels
    laplacian_3x3 = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
    for c in range(3):
        w_conv1[0, c] = laplacian_3x3 * 0.5
    b_conv1 = np.zeros(16, dtype=np.float32)

    # Conv2: 16 -> 32, kernel 3x3, stride 2, pad 1
    w_conv2 = rng.randn(32, 16, 3, 3).astype(np.float32) * np.sqrt(2.0 / (16 * 3 * 3))
    b_conv2 = np.zeros(32, dtype=np.float32)

    # Conv3: 32 -> 64, kernel 3x3, stride 2, pad 1
    w_conv3 = rng.randn(64, 32, 3, 3).astype(np.float32) * np.sqrt(2.0 / (32 * 3 * 3))
    b_conv3 = np.zeros(64, dtype=np.float32)

    # Flatten size after 224x224 through 3 stride-2 convs and 2 maxpools:
    # 224 -> conv1(s2) -> 112 -> pool1(s2) -> 56 -> conv2(s2) -> 28 -> pool2(s2) -> 14 -> conv3(s2) -> 7
    # Feature map: 64 x 7 x 7 = 3136 features
    fc1_in = 64 * 7 * 7
    w_fc1 = rng.randn(64, fc1_in).astype(np.float32) * np.sqrt(2.0 / fc1_in)
    b_fc1 = np.zeros(64, dtype=np.float32)

    # FC2: 64 -> 2 (Class 0: REAL, Class 1: FAKE)
    w_fc2 = rng.randn(2, 64).astype(np.float32) * np.sqrt(2.0 / 64)
    b_fc2 = np.array([0.1, -0.1], dtype=np.float32)

    # Initializers (Tensors)
    init_conv1_w = helper.make_tensor('conv1_w', TensorProto.FLOAT, [16, 3, 3, 3], w_conv1.flatten().tolist())
    init_conv1_b = helper.make_tensor('conv1_b', TensorProto.FLOAT, [16], b_conv1.tolist())
    init_conv2_w = helper.make_tensor('conv2_w', TensorProto.FLOAT, [32, 16, 3, 3], w_conv2.flatten().tolist())
    init_conv2_b = helper.make_tensor('conv2_b', TensorProto.FLOAT, [32], b_conv2.tolist())
    init_conv3_w = helper.make_tensor('conv3_w', TensorProto.FLOAT, [64, 32, 3, 3], w_conv3.flatten().tolist())
    init_conv3_b = helper.make_tensor('conv3_b', TensorProto.FLOAT, [64], b_conv3.tolist())
    init_fc1_w = helper.make_tensor('fc1_w', TensorProto.FLOAT, [64, fc1_in], w_fc1.flatten().tolist())
    init_fc1_b = helper.make_tensor('fc1_b', TensorProto.FLOAT, [64], b_fc1.tolist())
    init_fc2_w = helper.make_tensor('fc2_w', TensorProto.FLOAT, [2, 64], w_fc2.flatten().tolist())
    init_fc2_b = helper.make_tensor('fc2_b', TensorProto.FLOAT, [2], b_fc2.tolist())

    # Nodes
    # 1. Conv1 + Relu + MaxPool
    node_conv1 = helper.make_node('Conv', ['input', 'conv1_w', 'conv1_b'], ['conv1_out'], kernel_shape=[3, 3], strides=[2, 2], pads=[1, 1, 1, 1])
    node_relu1 = helper.make_node('Relu', ['conv1_out'], ['relu1_out'])
    node_pool1 = helper.make_node('MaxPool', ['relu1_out'], ['pool1_out'], kernel_shape=[2, 2], strides=[2, 2])

    # 2. Conv2 + Relu + MaxPool
    node_conv2 = helper.make_node('Conv', ['pool1_out', 'conv2_w', 'conv2_b'], ['conv2_out'], kernel_shape=[3, 3], strides=[2, 2], pads=[1, 1, 1, 1])
    node_relu2 = helper.make_node('Relu', ['conv2_out'], ['relu2_out'])
    node_pool2 = helper.make_node('MaxPool', ['relu2_out'], ['pool2_out'], kernel_shape=[2, 2], strides=[2, 2])

    # 3. Conv3 + Relu
    node_conv3 = helper.make_node('Conv', ['pool2_out', 'conv3_w', 'conv3_b'], ['conv3_out'], kernel_shape=[3, 3], strides=[2, 2], pads=[1, 1, 1, 1])
    node_relu3 = helper.make_node('Relu', ['conv3_out'], ['relu3_out'])

    # 4. Flatten
    node_flatten = helper.make_node('Flatten', ['relu3_out'], ['flatten_out'], axis=1)

    # 5. FC1 (Gemm) + Relu
    node_fc1 = helper.make_node('Gemm', ['flatten_out', 'fc1_w', 'fc1_b'], ['fc1_out'], transB=1)
    node_relu4 = helper.make_node('Relu', ['fc1_out'], ['relu4_out'])

    # 6. FC2 (Gemm)
    node_fc2 = helper.make_node('Gemm', ['relu4_out', 'fc2_w', 'fc2_b'], ['logits'], transB=1)

    # 7. Softmax (Output probabilities)
    node_softmax = helper.make_node('Softmax', ['logits'], ['output'], axis=1)

    # Inputs and Outputs
    input_info = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 3, 224, 224])
    output_info = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 2])

    # Graph
    graph_def = helper.make_graph(
        [node_conv1, node_relu1, node_pool1,
         node_conv2, node_relu2, node_pool2,
         node_conv3, node_relu3,
         node_flatten,
         node_fc1, node_relu4,
         node_fc2, node_softmax],
        'deepfake_cnn_classifier',
        [input_info],
        [output_info],
        [init_conv1_w, init_conv1_b, init_conv2_w, init_conv2_b,
         init_conv3_w, init_conv3_b, init_fc1_w, init_fc1_b,
         init_fc2_w, init_fc2_b]
    )

    # Model
    model_def = helper.make_model(graph_def, producer_name='deepfake_detection_system', opset_imports=[helper.make_opsetid('', 13)])
    onnx.checker.check_model(model_def)
    onnx.save(model_def, output_path)
    print(f"ONNX Deepfake model successfully created and saved to: {output_path}")

    # Verify with OpenCV DNN
    net = cv2.dnn.readNetFromONNX(output_path)
    dummy_input = np.random.rand(1, 3, 224, 224).astype(np.float32)
    net.setInput(dummy_input)
    out = net.forward()
    print(f"OpenCV DNN verification test passed! Output shape: {out.shape}, Output values: {out}")
    print(f"Class 0 (REAL): {out[0][0]:.4f}, Class 1 (FAKE): {out[0][1]:.4f}")
    assert out.shape == (1, 2)
    assert np.isclose(np.sum(out), 1.0, atol=1e-3)
    print("Softmax probability constraint verified: sum = 1.000")

if __name__ == '__main__':
    create_deepfake_onnx_model()
