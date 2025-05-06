# mAP_auto_test

This project provides a simple pipeline to automate the evaluation of NVIDIA DeepStream object detection results using mAP (mean Average Precision). The workflow consists of updating configuration files, running inference to generate predictions, and calculating mAP against ground truth annotations.

## Directory Structure

```
mAP_auto_test/
├── map_test.sh                 ← Step 1: Start the testing process
├── update_config.py            ← Step 2: Update config.yml
├── config.yml
├── deepstream_test_app.cpp               ← Step 3: Execute and generate bbox_predic.txt
├── bbox_predic.txt             ← Output of predicted bounding boxes
└── auto_test/
    └── cal_map.py              ← Step 4: Evaluate mAP using two files
        ├── ← bbox_predic.txt   (from the upper directory)
        └── ← bbox_gt.txt       (from the Video_gt folder)
```

## Steps to Run

### 1. Start the Testing Process

```bash
bash map_test.sh
```

This script will sequentially run all necessary steps: update the config, execute the model inference, and compute the mAP.

### 2. Update the Configuration

```bash
python update_config.py
```

This script modifies `config.yml` with parameters needed for the test (e.g., input video path, resolution).

### 3. Run Inference to Generate Predictions

Compile and run the C++ executable:

```bash
g++ deepstream_test_app.cpp -o deepstream_test_app `pkg-config --cflags --libs opencv`
./deepstream_test_app
```

This step will generate `bbox_predic.txt` containing bounding box predictions for the video frames.

### 4. Calculate mAP

```bash
cd auto_test
python cal_map.py
```

This script compares `bbox_predic.txt` (predictions) and `bbox_gt.txt` (ground truth) to calculate the mAP score.

> Note: Ensure `bbox_gt.txt` is present in the `Video_gt/` folder.

## Requirements

- Python 3.x
- OpenCV (for C++ inference)
- NumPy (for `cal_map.py`)

## Output

- `bbox_predic.txt`: Prediction results.
- Console output: Final mAP score.

---



