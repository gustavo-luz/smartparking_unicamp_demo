#!/bin/bash

# to run it:
# ./bench.sh

# if issues with permission
# chmod +x bench.sh

# to use nohup
# nohup ./bench.sh &

python3 inference_yolo_ultralytics.py --input_path ../../assets/demo_images --output_path ../../assets/results/results_yolo_ultralytics/yolov8n --savefigs debug --model yolov8n
python3 inference_yolo_ultralytics.py --input_path ../../assets/demo_images --output_path ../../assets/results/results_yolo_ultralytics/yolov10n --savefigs debug --model yolov10n
# add desired models