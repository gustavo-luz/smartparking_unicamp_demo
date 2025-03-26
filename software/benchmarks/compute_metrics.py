import os
import glob
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description="Compute evaluation metrics for model predictions.")
    parser.add_argument("--model", required=True, help="Model name")
    parser.add_argument("--base_dir_labeled_data", required=True, help="Base directory for labeled data")
    parser.add_argument("--base_dir_inference_csvs", required=True, help="Base directory for inference CSV files")
    parser.add_argument("--base_dir_results", required=True, help="Base directory to save results")
    return parser.parse_args()

def load_data(base_dir_labeled_data, base_dir_inference_csvs):
    """Load labeled data and model predictions."""
    base_labeled_df = pd.read_csv(f'{base_dir_labeled_data}/combined_metrics.csv')
    dfs = []
    csv_files = glob.glob(os.path.join(base_dir_inference_csvs, '**', 'df_individual_metrics*.csv'), recursive=True)
    
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        dfs.append(df)
    
    combined_df = pd.concat(dfs, ignore_index=True)
    return base_labeled_df, combined_df

def process_data(combined_df, base_labeled_df):
    """Process data and compute evaluation metrics."""
    print(base_labeled_df['image_name'].iloc[0])
    if 'croped_' in base_labeled_df['image_name'].iloc[0]:
        combined_df['image_name'] = 'croped_' + combined_df['image_name']
        max_spots = 16
    else:
        max_spots = max(combined_df['predicted_cars'])
        print(f'\n Max spots: {max_spots}\n')
    combined_df.rename(columns={'predicted_persons': 'predicted_cars', 'inference_time':'processing_time'}, inplace=True)
    print(combined_df,combined_df.columns)
    print(base_labeled_df,base_labeled_df.columns)
    combined_df['image_name'] = combined_df['image_name'].str.replace('.jpg', '')
    merged_df = pd.merge(combined_df, base_labeled_df[['image_name', 'real_cars']], on='image_name', how='inner')
    
    merged_df.to_csv(os.path.join('./', 'merged_metrics.csv'), index=False)
    merged_df['predicted_background'] = max_spots - merged_df['predicted_cars']
    if 'real_cars_x' in merged_df.columns or 'real_cars_y' in merged_df.columns:
        merged_df['real_cars'] = merged_df['real_cars_y']

    merged_df['real_background'] = max_spots - merged_df['real_cars']
    merged_df.to_csv(os.path.join('./', 'merged_metrics.csv'), index=False)
    print(merged_df)
    merged_df = calculate_metrics(merged_df)


    return merged_df

def calculate_metrics(df):
    """Calculate classification metrics."""
    # True Positives: Correctly predicted cars
    df['TP'] = df.apply(lambda row: min(row['predicted_cars'], row['real_cars']), axis=1)
    
    # False Negatives: Actual cars but model predicted none
    df['FN'] = df.apply(lambda row: max(row['real_cars'] - row['predicted_cars'], 0), axis=1)
    
    # False Positives: Model predicted cars but there were none
    df['FP'] = df.apply(lambda row: max(row['predicted_cars'] - row['real_cars'], 0), axis=1)
    
    # True Negatives: Correctly predicted no cars (background)
    df['TN'] = df.apply(lambda row: min(row['predicted_background'], row['real_background']), axis=1)
    
    # Accuracy, recall, precision, F1 score
    df['accuracy'] = (df['TP'] + df['TN']) / (df['TP'] + df['TN'] + df['FP'] + df['FN'])
    df['recall'] = df['TP'] / (df['TP'] + df['FN'])
    df['precision'] = df['TP'] / (df['TP'] + df['FP'])
    df['f1_score'] = 2 * ((df['precision'] * df['recall']) / (df['precision'] + df['recall']))
    
    # Sensitivity and specificity
    df['sensitivity'] = df['TP'] / (df['TP'] + df['FN'])
    df['specificity'] = df['TN'] / (df['TN'] + df['FP'])
    
    # Balanced accuracy
    df['bal_acc'] = (df['sensitivity'] + df['specificity']) / 2
    
    return df

def summarize_metrics(df, base_dir_results, model, suffix=""):
    """Generate summary metrics and save as CSV."""
    filename_suffix = f"_{suffix}" if suffix else ""
    
    summary_data = {
        'number_images': [len(df)],
        'average_accuracy': [df['accuracy'].mean()],
        'average_accuracy_balanced': [df['bal_acc'].mean()],
        'average_precision': [df['precision'].mean()],
        'average_recall': [df['recall'].mean()],
        'average_f1': [df['f1_score'].mean()],
    }
    summary_df = pd.DataFrame(summary_data)
    print(summary_df)
    summary_df.to_csv(os.path.join(base_dir_results, f'summary_metrics{filename_suffix}_{model}.csv'), index=False)

def save_confusion_matrix(df, base_dir_results, model, suffix=""):
    """Generate and save confusion matrix."""
    filename_suffix = f"_{suffix}" if suffix else ""
    total_TP = df['TP'].sum().astype(int)
    total_TN = df['TN'].sum().astype(int)
    total_FP = df['FP'].sum().astype(int)
    total_FN = df['FN'].sum().astype(int)
    total_samples = len(df)
    
    confusion_matrix = np.array([[total_TP, total_FP], [total_FN, total_TN]])
    print(confusion_matrix)
    group_names = ['TP', 'FP', 'FN', 'TN']
    group_counts = [f"{value}" for value in confusion_matrix.flatten()]
    group_percentages = [f"{value:.2%}" for value in confusion_matrix.flatten() / np.sum(confusion_matrix)]
    labels = np.asarray([f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names, group_counts, group_percentages)]).reshape(2, 2)
    
    plt.figure(figsize=(12, 10))
    heatmap = sns.heatmap(confusion_matrix, annot=labels, fmt='', cmap='Blues', xticklabels=['Background', 'Vehicles'], yticklabels=['Background', 'Vehicles'], annot_kws={"fontsize": 40})
    plt.xlabel('Actual', fontsize=40)
    plt.ylabel('Predicted', fontsize=40)
    plt.title(f'Confusion Matrix - {model} (Samples: {total_samples})', fontsize=42)
    
    heatmap.xaxis.set_tick_params(labelsize=38)
    heatmap.yaxis.set_tick_params(labelsize=38)
    
    cbar = heatmap.collections[0].colorbar
    cbar.ax.tick_params(labelsize=24)
    
    plt.savefig(os.path.join(base_dir_results, f'confusion_matrix{filename_suffix}_{model}.png'), bbox_inches='tight')

def main():
    args = parse_arguments()
    base_labeled_df, combined_df = load_data(args.base_dir_labeled_data, args.base_dir_inference_csvs)
    merged_df = process_data(combined_df, base_labeled_df)
    
    # Save results for all images
    summarize_metrics(merged_df, args.base_dir_results, args.model, suffix="all")
    save_confusion_matrix(merged_df, args.base_dir_results, args.model, suffix="all")
    
    # Filter and save results for images with at least one car
    filtered_df = merged_df[merged_df['real_cars'] > 0]
    summarize_metrics(filtered_df, args.base_dir_results, args.model, suffix="cars_only")
    save_confusion_matrix(filtered_df, args.base_dir_results, args.model, suffix="cars_only")
    
    print("Evaluation metrics computed and saved successfully.")

if __name__ == "__main__":
    main()
