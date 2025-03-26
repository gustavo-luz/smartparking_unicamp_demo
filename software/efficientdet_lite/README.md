# Efficientdet TFLite Inference

## Overview

This repository provides a script for running inference using a YOLO model in TensorFlow Lite format. The model is pre-trained on the COCO dataset with an input shape of 448X448. The environment setup follows the same requirements as the YOLO TFLite models.

## TODOs
- Add conversion steps for TFLite models following the provided link.
- Provide a script to download demo images instead of distributing third-party images.
- Cite the source of images in the associated research paper.

---

## Requirements
### Python & Virtual Environment
Ensure that Python is installed. The script has been tested with Python 3.11.7 and pip 23.2.1. Using a virtual environment is recommended to manage dependencies and prevent conflicts.

### Setting Up a Virtual Environment

1. **Create a virtual environment**:
   ```bash
   python -m venv tfliteenv
   ```

2. **Activate the virtual environment**:
   - **Windows**:
     ```bash
     tfliteenv\Scripts\activate
     ```
   - **macOS/Linux**:
     ```bash
     source tfliteenv/bin/activate
     ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Deactivate the virtual environment**:
   ```bash
   deactivate
   ```

---

## Required Files

Before running the script, ensure you have:
- **`mask_original_img_768_1024_bw.png`**: This mask file defines regions where object counting is performed. The default mask is all black, meaning every vehicle will be counted. You can modify it using image editing tools (e.g., GIMP) by selecting and painting the desired regions in black.

---

## Script Details

### Main Script: `inference_edet_lite.py`
This script processes images using a YOLO TFLite model and outputs results in a CSV file.

**Example demo images**: Available in the [docs folder](../../assets/demo_images), extracted from [CNRPark](http://cnrpark.it/). The dataset includes images in shape 728x1024, but other shapes should also be compatible.

### Running the Script

Run the script using the following command:
```bash
python3 inference_edet_lite.py
```

#### Manual Configuration Required Dependng on User Needs
Before running the script, ensure that the following variables in the script are correctly set according to your needs:

```python
MODEL = 'lite-model_efficientdet_lite2_detection_default_1.tflite'
IMAGE_DIR = '../../assets/demo_images'
OUTPUT_DIR = '../../assets/results/results_efficientdet_tflite/efficientdetd2lite'
savefigs = 'debug'  
```

Make sure `MODEL`, `IMAGE_DIR`, and `OUTPUT_DIR` are correctly defined before executing the script. The `savefigs` variable should be set to `'debug'` if you want to save images or `'no'` if you prefer not to save them.



## Output Structure

```
output_path/
│
├── batch_<timestamp>/  # New batch directory for each run
│   ├── df_individual_metrics_<batch_number>_<timestamp>.csv  # Per-image detection results
│   ├── annotated_image.jpg/  # Processed images with bounding boxes (if enabled)
```
---

## Benchmarking

To evaluate model performance, use the benchmarking script in [software/benchmarks](../benchmarks/README.md).

---
