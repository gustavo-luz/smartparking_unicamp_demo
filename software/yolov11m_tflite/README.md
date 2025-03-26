todo: colocar conversão pra tflite e passos, seguimos o link..
no entanto modelo está disponível, pre trained at COCO with input shape 640x640

this is the same environment of efficientdet models




## Requirements
### Python & Virtual Environment

Before running the script, ensure that Python is installed on your system. The script has been tested with Python 3.11.7 and pip 23.2.1. It is recommended to use a virtual environment to manage dependencies and avoid conflicts with other Python packages on your system.

### Creating a Virtual Environment

To create and activate a virtual environment, follow these steps:

1. **Create the virtual environment**:  
   In the terminal, navigate to the project directory and run:
   ```bash
   python -m venv tfliteenv
   ```
   This will create a directory named `tfliteenv` in your project directory, which will contain the isolated Python environment.

2. **Activate the virtual environment**:
   - **macOS/Linux**:
     ```bash
     source tfliteenv/bin/activate
     ```

   Once activated, the terminal prompt should change to indicate that the virtual environment is active, e.g., `(tfliteenv)`.

3. **Install required libraries**:  
   With the virtual environment active, run the following command to install all necessary dependencies:
   ```bash
   pip install -r requirements.txt
   ```

   This will install the required libraries

4. **Deactivate the virtual environment**:  
   After you're done working, you can deactivate the virtual environment by running:
   ```bash
   deactivate
   ```

### Required File

Ensure that you have the following file before running the script:
- **`maskmask_original_img_768_1024_bw.png`**: This file serves as a guideline to select regions where to perform counting. We provide an all-black mask, so every vehicle will be counted. To create a custom mask, you can a base image as an example. You can select and paint the desired counting region as black using image editing tools like GIMP with the free selection tool. The rest of the image should be left white.

---

## Files

1. **inference_yolo_tflite.py**  
   This script processes images, runs inference using a YOLO model, and returns results in a csv.

   Example test images can be found on [docs folder](../../assets/demo_images) extracted from [CNRPark](http://cnrpark.it/) named CNR-EXT_FULL_IMAGE_1000x750.tar (1.1 GB) and are from the under CNR-EXT_FULL_IMAGE_1000x750/FULL_IMAGE_1000x750/SUNNY/2015-11-12/camera1. The images used for the reference paper are in shape 728x1024 but other shapes should also be compatible.

TODO: criar um script que baixa as fotos de demonstração e colocar em todos os readmes, pra não distribuir fotos de terceiros. add citação no artigo

## Running the Script

To execute the script, you need to provide some command-line arguments. Here’s a summary of the required arguments:

- `input_path`: Path to the directory containing the images to be processed.
- `output_path`: Path to the directory where results (such as processed images or result files) will be saved.
- `--savefigs`: Option to save figures. Available options:
  - `debug`: Saves all figures generated during the process.
  - `no`: Does not save any figures.


### Example usage:


```
python3 inference_yolo_tflite.py --input_path ../../assets/demo_images --output_path ../../assets/results/results_yolo_ultralytics/yolov8n --savefigs debug
```

---
  

### Output Directory Structure

When you run the script, it creates an output directory (specified by `--output_path`). Inside this directory, you will find:

```
output_path/
│
├── batch_<timestamp>/  # A new batch directory is created for each run
│   ├── df_individual_metrics_<batch_number>_<timestamp>.csv  # Metrics for each processed image
│   ├── df_full_metrics.csv  # Overall metrics for the batch
│   ├── results/  # Processed images with bounding boxes (if savefigs is enabled)
│   ├── images/  # Original images (if configured)
│
```

### Description of Output Files

`batch_<timestamp>/`

Each time the script is executed, a new batch directory is created. The `<timestamp>` is based on the execution time to ensure uniqueness.

- **`df_individual_metrics_<batch_number>_<timestamp>.csv`**: This file contains per-image detection results, including:
  - Image name
  - Timestamp
  - Predicted number of cars
  - Inference time
  - Blank fields for metrics


- **`df_full_metrics.csv`**: This file aggregates the results from all processed images in the batch, including:
  - Number of images processed
  - Average processing time per image

### `results/`
If saving figures is enabled (`--savefigs debug` or `--savefigs partial`), this directory contains processed images with bounding boxes drawn over detected objects.

---

2. **bench.sh.py**  
   This shell script is just and easy way to run multiple tests at once.

### Metrics: 

To compute the result you will need the standardized script that can be found at [software/benchmarks](../benchmarks/README.md)

---
