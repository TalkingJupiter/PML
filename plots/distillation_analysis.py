import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import json
import os
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def load_distillation_data(results_dir="experiments"):
    """
    Loads history.json from all subdirectories in results_dir.
    """
    all_runs = []
    base_path = Path(results_dir)
    
    for history_path in base_path.glob("**/history.json"):
        try:
            with open(history_path, 'r') as f:
                data = json.load(f)
                run_name = data.get('run_name', history_path.parent.name)
                
                teacher_name = data.get('teacher_name', 'Unknown')
                teacher_acc = data.get('teacher_accuracy', 100.0)
                student_params = data.get('student_parameters', 0)
                teacher_params = data.get('teacher_parameters', 0)
                student_width = data.get('student_width', 1.0)
                
                for entry in data['history']:
                    # Extract metrics and handle potential naming mismatches
                    t_acc = entry.get('test_acc') or entry.get('test_accuracy')
                    if t_acc is None:
                        continue
                        
                    row = {
                        'run_name': run_name,
                        'epoch': entry['epoch'],
                        'test_accuracy': t_acc,
                        'train_loss': entry.get('train_loss'),
                        'teacher_name': teacher_name,
                        'teacher_accuracy': teacher_acc,
                        'student_parameters': student_params,
                        'teacher_parameters': teacher_params,
                        'student_width': student_width,
                        'accuracy_percentage_of_parent': (t_acc / teacher_acc) * 100 if teacher_acc > 0 else 0
                    }
                    all_runs.append(row)
        except Exception as e:
            print(f"Warning: Could not process {history_path}: {e}")
                
    return pd.DataFrame(all_runs)

def plot_distillation_efficiency(df, output_path="plots/distillation_efficiency.png"):
    if df.empty: return
    plt.figure(figsize=(12, 7))
    sns.lineplot(data=df, x='epoch', y='accuracy_percentage_of_parent', hue='run_name', linewidth=2.5, marker='o', markersize=4)
    plt.title('Distillation Efficiency: Student Accuracy as % of Teacher', fontsize=16, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Percentage of Teacher Accuracy (%)', fontsize=12)
    plt.axhline(100, color='red', linestyle='--', alpha=0.5, label='Teacher Baseline')
    plt.legend(title='Experiment Run', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Saved efficiency plot to {output_path}")

def plot_compression_summary(df, output_path="plots/compression_summary.png"):
    if df.empty: return
    # Get the latest epoch for each run
    summary_df = df.sort_values('epoch').groupby('run_name').last().reset_index()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 1. Parameter Comparison
    param_data = []
    for _, row in summary_df.iterrows():
        param_data.append({'Model': 'Teacher', 'Run': row['run_name'], 'Params': row['teacher_parameters']})
        param_data.append({'Model': 'Student', 'Run': row['run_name'], 'Params': row['student_parameters']})
    
    param_df = pd.DataFrame(param_data)
    sns.barplot(data=param_df, x='Run', y='Params', hue='Model', ax=ax1)
    ax1.set_title('Parameter Count Comparison', fontweight='bold')
    ax1.set_yscale('log')
    ax1.set_ylabel('Parameters (log scale)')
    ax1.tick_params(axis='x', rotation=45)

    # 2. Final Accuracy
    sns.barplot(data=summary_df, x='run_name', y='test_accuracy', ax=ax2, palette='viridis')
    ax2.set_title('Final Student Accuracy', fontweight='bold')
    ax2.set_ylabel('Accuracy (%)')
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Saved compression summary to {output_path}")

def plot_scaling_analysis(df, output_path="plots/scaling_analysis.png"):
    if df.empty: return
    # Get the latest epoch for each combination of teacher and width
    scaling_df = df.sort_values('epoch').groupby(['teacher_name', 'student_width']).last().reset_index()
    
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=scaling_df, x='student_width', y='test_accuracy', hue='teacher_name', marker='s', markersize=8, linewidth=2)
    
    plt.title('Scaling Analysis: Accuracy vs. Student Width', fontsize=16, fontweight='bold')
    plt.xlabel('Student Width Multiplier', fontsize=12)
    plt.ylabel('Final Test Accuracy (%)', fontsize=12)
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Saved scaling analysis plot to {output_path}")

if __name__ == "__main__":
    os.makedirs("plots", exist_ok=True)
    try:
        df = load_distillation_data()
        if not df.empty:
            plot_distillation_efficiency(df)
            plot_compression_summary(df)
            plot_scaling_analysis(df)
        else:
            print("No experimental data found in 'experiments/' directory.")
    except Exception as e:
        print(f"Error generating plots: {e}")
