import re
import subprocess

# model path
new_models = [
    (
        "../../../objectDetector_Yolo/DeepStream-Yolo/yolov5/yolov5s.pt.onnx",
        "../../../objectDetector_Yolo/DeepStream-Yolo/yolov5/yolov5s.pt.onnx_b1_gpu0_fp32.engine"
    ),
    (
        "../../../objectDetector_Yolo/DeepStream-Yolo/yolov5/yolov5m.pt.onnx",
        "../../../objectDetector_Yolo/DeepStream-Yolo/yolov5/yolov5m.pt.onnx_b1_gpu0_fp32.engine"
    ),
    (
        "../../../objectDetector_Yolo/DeepStream-Yolo/yolov6/yolov6s.pt.onnx",
        "../../../objectDetector_Yolo/DeepStream-Yolo/yolov6/yolov6s.pt.onnx_b1_gpu0_fp32.engine"
    ),
    (
        "../../../objectDetector_Yolo/DeepStream-Yolo/yolov6/yolov6m.pt.onnx",
        "../../../objectDetector_Yolo/DeepStream-Yolo/yolov6/yolov6m.pt.onnx_b1_gpu0_fp32.engine"
    ),
    (
        "../../../objectDetector_Yolo/DeepStream-Yolo/yolov7/yolov7.pt.onnx",
        "../../../objectDetector_Yolo/DeepStream-Yolo/yolov7/yolov7.pt.onnx_b1_gpu0_fp32.engine"
    ),
    (
        "../../../objectDetector_Yolo/DeepStream-Yolo/yolov8/yolov8s.pt.onnx",
        "../../../objectDetector_Yolo/DeepStream-Yolo/yolov8/yolov8s.pt.onnx_b1_gpu0_fp32.engine"
    ),
    (
        "../../../objectDetector_Yolo/DeepStream-Yolo/yolov8/yolov8m.pt.onnx",
        "../../../objectDetector_Yolo/DeepStream-Yolo/yolov8/yolov8m.pt.onnx_b1_gpu0_fp32.engine"
    )
]

# YAML file path
file_path = "diva_test_pgie_config.yml"

# read YAML file
with open(file_path, "r") as file:
    lines = file.readlines()

# update YAML file
for i, model in enumerate(new_models):
    onnx_pattern = re.compile(r"^\s*onnx-file:")
    engine_pattern = re.compile(r"^\s*model-engine-file:")
    model_replaced = False

    for j, line in enumerate(lines):

        if onnx_pattern.match(line):  # update onnx-file
            lines[j] = f"  onnx-file: {model[0]}\n"

        if engine_pattern.match(line):  # update model-engine-file
            lines[j] = f"  model-engine-file: {model[1]}\n"
            model_replaced = True

        # if update onnx-file and model-engine-file pass to next model
        if model_replaced:
            break
        
    # output YAML file
    with open(file_path, "w") as file:
        file.writelines(lines)

    print("YAML updated！")

    # test all videos map
    try:
        subprocess.run(["bash", "map_test.sh"], check=True)
        print("Bash command success！")
    except subprocess.CalledProcessError as e:
        print(f"Bash command failed：{e}")

