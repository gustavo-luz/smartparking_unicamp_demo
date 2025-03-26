# Smart Parking System with Deep Learning at Unicamp


pegar dados de algum lugar e testar,talvez pklot
todo: entrar no pi e descobrir versão certinha do python e das libs que estão rodando e colocar aqui e no requirements
colocar imagens base exemplo
executar do 0 e ver se funciona com python3.11 ai podemos colocar python 3.8 and 3.11 tested
opção de não usar a máscara, ou coloca uma máscara inteira preta só de referência

The files here correspond to performing benchmarks with different deep learning models and measure inference time, as long as machine learning metrics for YOLOv8 to YOLOv11, as reported in the paper available at [arxiv](https://arxiv.org/abs/2412.01983) cited by: 

```
@article{da2024smart,
  title={Smart Parking with Pixel-Wise ROI Selection for Vehicle Detection Using YOLOv8, YOLOv9, YOLOv10, and YOLOv11},
  author={da Luz, Gustavo PCP and Sato, Gabriel Massuyoshi and Gonzalez, Luis Fernando Gomez and Borin, Juliana Freitag},
  journal={arXiv preprint arXiv:2412.01983},
  year={2024}
}

```

---

## Prerequisites

Ensure that Python is installed on your system. This script has been tested with Python 3.11.7 and Python. Required libraries can be installed using:

```
pip install -r requirements.txt
```

Another prerequisite is a file named maskmask_original_img_768_1024_bw.png that will serve as a guideline to select regions where to perform counting. We provide an all black mask, so every vehicle is counted. To create a custom mask you can use the base image as example and select and paint the desire counting region as black only, leaving the rest white. This can be done with image editing tools like GIMP using the free selection tool.

---

## Files

1. **inference_yolo_ultralytics.py**  
   This script processes images, runs inference using a YOLO model, and returns results in a csv.

   Example test images can be found on [docs folder](../../assets/demo_images). The images used for the paper are in shape 728x1024 but other shapes should also be compatible.



## Running the Script

To execute the script, you need to provide some command-line arguments. Here’s a summary of the required arguments:

- `input_path`: Path to the directory containing the images to be processed.
- `output_path`: Path to the directory where results (such as processed images or result files) will be saved.
- `--savefigs`: Option to save figures. Available options:
  - `debug`: Saves all figures generated during the process.
  - `no`: Does not save any figures.
  - `partial`: Saves only essential figures.
- `--model`: YOLO model to be used, e.g., `yolov8n`, `yolov8x`, `yolov9t`, `yolov9x`, `yolov10n`, `yolov10x`,`yolo11n`,`yolo11x`. The model will be automatically downloaded.


### Example usage:


```
python3 inference_yolo_ultralytics.py --input_path ../../assets/demo_images --output_path ../../assets/results/results_yolo_ultralytics/yolov8n --savefigs debug --model yolov8n
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

To compute the result you will need the standardized script that can be found at [software/benchmarks_accuracy](../benchmarks_accuracy/compute_metrics.py)

---
