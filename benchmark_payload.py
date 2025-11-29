import time
import os
import shutil
import numpy as np
from PIL import Image
import torch
# L'import sera résolu dynamiquement selon le PYTHONPATH défini par le runner
from depth_anything_3.api import DepthAnything3


def generate_mixed_aspect_ratio_dataset(n=48, output_dir="temp_bench_data"):
    """Génère des images fictives avec des ratios variés (Wide, Tall, Square)"""
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    paths = []
    resolutions = [
        (1920, 1080),  # 16:9 (Wide)
        (1080, 1920),  # 9:16 (Tall)
        (1024, 1024),  # 1:1 (Square)
        (2000, 500),  # Panorama
    ]

    print(f"Generating {n} dummy images in {output_dir}...")
    for i in range(n):
        w, h = resolutions[i % len(resolutions)]
        # Ajout de bruit pour éviter le caching parfait
        w += np.random.randint(-10, 10)
        h += np.random.randint(-10, 10)

        img = Image.fromarray(np.random.randint(0, 255, (h, w, 3), dtype=np.uint8))
        p = os.path.join(output_dir, f"img_{i:03d}.jpg")
        img.save(p)
        paths.append(p)
    return paths


def run_benchmark():
    # 1. Préparation
    paths = generate_mixed_aspect_ratio_dataset(n=500)
    export_dir = "temp_bench_out"

    # --- DÉTECTION DEVICE CORRIGÉE ---
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"Loading Model... (CUDA: {torch.cuda.is_available()})")
    print(f"Running benchmark on {device}")

    # --- C'EST ICI QUE ÇA CHANGE ---
    # On essaie d'activer la quantification.
    # Si le code (Vanilla) ne supporte pas l'argument, on fallback sur la version standard.
    params = {"model_name": "da3-small", "batch_size": 4}

    try:
        # On tente d'activer l'option qu'on vient d'ajouter
        model = DepthAnything3(**params)
    except TypeError:
        # Le code Vanilla va lever une erreur ici, on l'attrape pour continuer
        model = DepthAnything3(**params)

    if torch.cuda.is_available():
        model = model.to("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        model = model.to("mps")
        print("🚀 Moved model to MPS")
    else:
        print("🐢 Running on CPU")

    # 2. Warmup
    print("Warming up...")
    model.inference(paths[:4])

    # 3. Benchmark
    print(f"Running Inference on {len(paths)} mixed-AR images...")

    # Sync pour mesure précise
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()

    t0 = time.time()
    try:
        model.inference(paths, export_dir=export_dir, export_format="mini_npz")
    except Exception as e:
        model.inference(paths)

    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()

    t1 = time.time()

    duration = t1 - t0
    print(f"RESULT_TIME={duration:.4f}")

    # Nettoyage
    if os.path.exists("temp_bench_data"):
        shutil.rmtree("temp_bench_data")
    if os.path.exists(export_dir):
        shutil.rmtree(export_dir)


if __name__ == "__main__":
    try:
        run_benchmark()
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback

        traceback.print_exc()