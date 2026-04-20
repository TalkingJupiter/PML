import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import json
import os
import glob
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def load_distillation_data(results_dir="experiments"):
    """
    Loads history.json from all subdirectories in results_dir.
    Expected structure: experiments/run_name/history.json
    """
    all_runs = []
    base_path = Path(results_dir)
    
    for history_path in base_path.glob("**/history.json"):
        with open(history_path, 'r') as f:
            data = json.load(f)
            run_name = history_path.parent.name
            
            # Metadata might be stored in a separate file or in history.json
            # For simplicity, we assume metadata is in history.json
            teacher_name = data.get('teacher_name', 'Unknown')
            teacher_acc = data.get('teacher_accuracy', 100.0)
            student_params = data.get('student_parameters', 0)
            
            for entry in data['history']:
                entry['run_name'] = run_name
                entry['teacher_name'] = teacher_name
                entry['teacher_accuracy'] = teacher_acc
                entry['student_parameters'] = student_params
                # Calculate % of parent accuracy
                entry['accuracy_percentage_of_parent'] = (entry['test_accuracy'] / teacher_acc) * 100
                all_runs.append(entry)
                
    return pd.DataFrame(all_runs)

def plot_distillation_efficiency(df, output_path="plots/distillation_efficiency.png"):
    """
    Plots accuracy as % of parent model vs epoch.
    """
    plt.figure(figsize=(12, 7))
    ax = sns.lineplot(
        data=df,
        x='epoch',
        y='accuracy_percentage_of_parent',
        hue='teacher_name',
        linewidth=2.5,
        marker='o',
        markersize=4
    )
    
    plt.title('Distillation Efficiency: Student Accuracy as % of Teacher', fontsize=16, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Percentage of Teacher Accuracy (%)', fontsize=12)
    plt.axhline(100, color='red', linestyle='--', alpha=0.5, label='Teacher Baseline')
    plt.legend(title='Teacher Model')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Saved efficiency plot to {output_path}")

def plot_param_count_vs_accuracy(df, output_path="plots/param_vs_accuracy.png"):
    """
    Plots Parameter Count vs. Teacher Model performance.
    """
    # Get final accuracy per run
    final_df = df.groupby('run_name').last().reset_index()
    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=final_df,
        x='student_parameters',
        y='test_accuracy',
        hue='teacher_name',
        size='teacher_accuracy',
        sizes=(50, 200),
        alpha=0.7
    )
    
    plt.title('Student Parameter Count vs. Final Test Accuracy', fontsize=16, fontweight='bold')
    plt.xlabel('Number of Trainable Parameters', fontsize=12)
    plt.ylabel('Final Test Accuracy (%)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Saved parameter analysis plot to {output_path}")

if __name__ == "__main__":
    # Example usage (will need actual data to produce plots)
    try:
        df = load_distillation_data()
        if not df.empty:
            plot_distillation_efficiency(df)
            plot_param_count_vs_accuracy(df)
        else:
            print("No experimental data found in 'experiments/' directory.")
    except Exception as e:
        print(f"Error generating plots: {e}")
