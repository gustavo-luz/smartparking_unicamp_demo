# Smart Parking System with Deep Learning at Unicamp

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

## Files

1. **inference_yolo_ultralytics.py**  
   This script processes images, runs inference using a YOLO model, and returns results in a csv.

   Example test images can be found on [docs folder](../../assets/demo_images).

---

## Prerequisites

Ensure that Python is installed on your system. This script has been tested with Python 3.11.7 and Python. Required libraries can be installed using:

```
pip install -r requirements.txt
```

---

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
python3 inference_yolo_ultralytics.py --input_dir ../../assets/demo_images --output_dir ../../assets/results/ --savefigs no --model yolov8x
```

---
  
### Output:

Output is stored in the output_dir chosen and contains an csv with predicted cars and inference time. If selected, the output images will also be saved.

---


### Metrics: 

To compute the result you will need: script ipynb, json file configuração.

the output will be , pensar no diretório

---
