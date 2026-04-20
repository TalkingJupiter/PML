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
    all_runs = []
    base_path = Path(results_dir)
    
    for history_path in base_path.glob("**/history.json"):
        with open(history_path, 'r') as f:
            data = json.load(f)
            run_name = history_path.parent.name
            
            teacher_name = data.get('teacher_name', 'Unknown')
            teacher_acc = data.get('teacher_accuracy', 100.0)
            student_params = data.get('student_parameters', 0)
            teacher_params = data.get('teacher_parameters', 0)
            
            for entry in data['history']:
                entry['run_name'] = run_name
                entry['teacher_name'] = teacher_name
                entry['teacher_accuracy'] = teacher_acc
                entry['student_parameters'] = student_params
                entry['teacher_parameters'] = teacher_params
                entry['accuracy_percentage_of_parent'] = (entry['test_accuracy'] / teacher_acc) * 100
                all_runs.append(entry)
                
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
    # Get one row per run
    summary_df = df.groupby('run_name').last().reset_index()
    
    # Create two subplots: one for params, one for final accuracy
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 1. Parameter Comparison
    param_data = []
    for _, row in summary_df.iterrows():
        param_data.append({'Model': 'Teacher', 'Run': row['run_name'], 'Params': row['teacher_parameters']})
        param_data.append({'Model': 'Student', 'Run': row['run_name'], 'Params': row['student_parameters']})
    
    param_df = pd.DataFrame(param_data)
    sns.barplot(data=param_df, x='Run', y='Params', hue='Model', ax=ax1)
    ax1.set_title('Parameter Count Comparison', fontweight='bold')
    ax1.set_yscale('log') # Log scale for better visibility of small students
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

if __name__ == "__main__":
    os.makedirs("plots", exist_ok=True)
    try:
        df = load_distillation_data()
        if not df.empty:
            plot_distillation_efficiency(df)
            plot_compression_summary(df)
        else:
            print("No experimental data found.")
    except Exception as e:
        print(f"Error: {e}")
