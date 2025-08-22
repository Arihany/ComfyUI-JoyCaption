import subprocess
import sys
import os

def run(cmd):
    print(f"\n>> {cmd}")
    subprocess.check_call(cmd, shell=True)

def install_llama():
    print("Installing llama-cpp-python for ComfyUI (with CUDA support if available)...")

    run(f"{sys.executable} -m pip install --upgrade pip")
    run(f"{sys.executable} -m pip cache purge")
    run(f"{sys.executable} -m pip install scikit-build-core cmake")

    try:
        import torch
        has_gpu = torch.cuda.is_available()
    except ImportError:
        has_gpu = False

    if has_gpu:
        os.environ["CMAKE_ARGS"] = "-DGGML_CUDA=on"
        print("✅ GPU detected — building with CUDA support")
    else:
        print("🖥️ No GPU detected — building CPU-only version")

    run(f"{sys.executable} -m pip install llama-cpp-python --no-cache-dir")
    os.environ.pop("CMAKE_ARGS", None)

    print("\n✅ Installation complete. Please restart ComfyUI.")

if __name__ == "__main__":
    install_llama()
