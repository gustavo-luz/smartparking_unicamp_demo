#!/bin/bash

# python inference_mask_768_1024.py --model yolov8n --input_path ../assets/data/bench_fotos --output_path ../assets/results/bench_fotos_yolov8n --savefigs no --num_splits 1 --mask_type pre
# python inference_mask_768_1024.py --model yolov8x --input_path ../assets/data/bench_fotos --output_path ../assets/results/bench_fotos_yolov8x --savefigs no --num_splits 1 --mask_type pre
# python inference_mask_768_1024.py --model yolov9t --input_path ../assets/data/bench_fotos --output_path ../assets/results/bench_fotos_yolov9t --savefigs no --num_splits 1 --mask_type pre
# python inference_mask_768_1024.py --model yolov9e --input_path ../assets/data/bench_fotos --output_path ../assets/results/bench_fotos_yolov9e --savefigs no --num_splits 1 --mask_type pre
# python inference_mask_768_1024.py --model yolov10n --input_path ../assets/data/bench_fotos --output_path ../assets/results/bench_fotos_yolov10n --savefigs no --num_splits 1 --mask_type pre
# python inference_mask_768_1024.py --model yolov10x --input_path ../assets/data/bench_fotos --output_path ../assets/results/bench_fotos_yolov10x --savefigs no --num_splits 1 --mask_type pre
# chmod +x bench.sh
# ./bench.sh

# nohup ./bench.sh &
#nohup ./bench_post.sh &

# tail -f nohup.out
# python inference_mask_768_1024.py --model yolov8n --input_path ../assets/data/bench_original --output_path ../assets/results/bench_fotos_yolo8n_original --savefigs no --num_splits 1 --mask_type post
# python inference_mask_768_1024.py --model yolov9t --input_path ../assets/data/bench_original --output_path ../assets/results/bench_fotos_yolo9t_original --savefigs no --num_splits 1 --mask_type post
# python inference_mask_768_1024.py --model yolov10n --input_path ../assets/data/bench_original --output_path ../assets/results/bench_fotos_yolo10n_original --savefigs no --num_splits 1 --mask_type post
# python inference_mask_768_1024.py --model yolov10x --input_path ../assets/data/bench_original --output_path ../assets/results/bench_fotos_yolo10x_original --savefigs no --num_splits 1 --mask_type post
# python inference_mask_768_1024.py --model yolov9e --input_path ../assets/data/bench_original --output_path ../assets/results/bench_fotos_yolo9e_original --savefigs no --num_splits 1 --mask_type post
# python inference_mask_768_1024.py --model yolov8x --input_path ../assets/data/bench_original --output_path ../assets/results/bench_fotos_yolo8x_original --savefigs no --num_splits 1 --mask_type post


# python inference_mask_768_1024_detr.py --model yolo11n --input_path ../assets/data/bench_original --output_path ../assets/results/bench_fotos_yolo11n_original_cpu --savefigs no --num_splits 1 --mask_type post
# python inference_mask_768_1024.py --model yolo11n --input_path bench_original --output_path results/bench_fotos_yolo11n_pi3_5 --savefigs no --num_splits 1 --mask_type post

# python inference_mask_768_1024.py --model yolo11x --input_path bench_original_5 --output_path results/bench_fotos_yolo11x_pi3_5 --savefigs no --num_splits 1 --mask_type post

# python inference_mask_768_1024.py --model yolo11x --input_path bench_original_100 --output_path results/bench_fotos_yolo11x_pi3_100 --savefigs no --num_splits 1 --mask_type post


#python inference_mask_768_1024.py --model yolov8n --input_path bench_original_100 --output_path results/bench_fotos_yolo8n_original_cpu_clean_all --savefigs no --num_splits 1 --mask_type post
#python inference_mask_768_1024.py --model yolov9t --input_path bench_original_100 --output_path results/bench_fotos_yolo9t_original_cpu_clean_all --savefigs no --num_splits 1 --mask_type post
#python inference_mask_768_1024.py --model yolov10n --input_path bench_original_100 --output_path results/bench_fotos_yolo10n_original_cpu_clean_all --savefigs no --num_splits 1 --mask_type post
#python inference_mask_768_1024.py --model yolov10x --input_path bench_original_100 --output_path results/bench_fotos_yolo10x_original_cpu_clean_all --savefigs no --num_splits 1 --mask_type post
python inference_mask_768_1024.py --model yolov9e --input_path bench_original_100 --output_path results/bench_fotos_yolo9e_original_cpu_clean_all_rerun --savefigs no --num_splits 1 --mask_type post
#python inference_mask_768_1024.py --model yolov8x --input_path bench_original_100 --output_path results/bench_fotos_yolo8x_original_cpu_clean_all --savefigs no --num_splits 1 --mask_type post
#python inference_mask_768_1024.py --model yolo11n --input_path bench_original_100 --output_path results/bench_fotos_yolo11n_original_cpu_clean_all --savefigs no --num_splits 1 --mask_type post
#python inference_mask_768_1024.py --model yolo11x --input_path bench_original_100 --output_path results/bench_fotos_yolo11x_original_cpu_clean_all --savefigs no --num_splits 1 --mask_type post

