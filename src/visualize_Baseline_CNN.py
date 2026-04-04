import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from src.model_CNN import build_baseline_cnn

model = build_baseline_cnn()

output_dir = os.path.join(project_root, "outputs")
os.makedirs(output_dir, exist_ok=True)

save_path = os.path.join(output_dir, "CNN.keras")
model.save(save_path)

print(f"Saved model to: {save_path}")