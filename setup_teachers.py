import os
from pathlib import Path

def get_available_teachers(outdir="experiments"):
    """
    Scans the experiments directory for available teacher models
    and provides HF fallbacks for standard architectures.
    """
    # Mapping of logical names to fallback HF model IDs (timm-compatible)
    FALLBACKS = {
        "ResNet50": "hf_hub:edadaltocg/resnet50_cifar10",
    }
    
    # Mapping of logical names to expected local paths (under outdir)
    LOCAL_PATHS = {
        "ResNet50": "resnet/resnet50_seed0",
        "DenseNet161": "densenet/densenet161_seed0",
        "VGG13": "vgg13/vgg13_bn_seed0",
    }
    
    teachers = []
    base_path = Path(outdir)
    
    print("=== Checking Teacher Availability ===")
    for name, local_rel_path in LOCAL_PATHS.items():
        local_path = base_path / local_rel_path
        best_model = local_path / "best_model.pth"
        
        config = {
            "name": name,
            "teacher_run": None,
            "teacher_model": None,
            "source": "None"
        }
        
        if best_model.exists():
            print(f"[LOCAL] Found {name} at {local_path}")
            config["teacher_run"] = local_rel_path
            config["source"] = "local"
        elif name in FALLBACKS:
            print(f"[HF HUB] {name} will be pulled from {FALLBACKS[name]}")
            config["teacher_model"] = FALLBACKS[name]
            config["source"] = "huggingface"
        else:
            print(f"[MISSING] {name} - No local checkpoint or HF fallback found.")
            continue
            
        teachers.append(config)
    
    return teachers

if __name__ == "__main__":
    available = get_available_teachers()
    print(f"\nTotal available teachers: {len(available)}")
