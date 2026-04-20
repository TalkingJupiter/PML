import subprocess
import os
import argparse
from setup_teachers import get_available_teachers

def run_experiment(exp, epochs=2, batch_size=64, strict=False):
    run_name = f"FeatureKD_from_{exp['name']}"
    if strict:
        run_name += "_Strict"
        
    cmd = [
        "uv", "run", "training_feature_kd_resnet50_small.py",
        "--run_name", run_name,
        "--epochs", str(epochs),
        "--batch_size", str(batch_size),
    ]
    
    if strict:
        cmd.append("--strict")
    
    if exp["teacher_run"]:
        cmd.extend(["--teacher_run", exp["teacher_run"]])
    elif exp["teacher_model"]:
        cmd.extend(["--teacher_model", exp["teacher_model"]])
        
    print(f"\n>>> Running Experiment: {run_name} (Source: {exp['source']})")
    print(f">>> Command: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"!!! Experiment {run_name} failed with error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict", action="store_true", help="Strictly internal feature distillation")
    args_loop = parser.parse_args()

    # Check if we are in a SLURM environment or local
    is_slurm = "SLURM_JOB_ID" in os.environ
    
    default_epochs = 200 if is_slurm else 2
    default_batch_size = 256 if is_slurm else 64
    
    # Dynamically find teachers (Local or HF Fallback)
    available_teachers = get_available_teachers()
    
    if not available_teachers:
        print("!!! No teachers available. Exiting.")
        exit(1)
        
    for exp in available_teachers:
        run_experiment(exp, epochs=default_epochs, batch_size=default_batch_size, strict=args_loop.strict)
        
    # After all experiments, generate plots
    print("\n>>> Generating Comparison Plots...")
    subprocess.run(["uv", "run", "plots/distillation_analysis.py"])
