import os
from pathlib import Path
import torch
import numpy as np
from PIL import Image, ImageOps

class ImageBatchPath:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {"image_dir": ("STRING", {"default": "", "multiline": True, "placeholder": "Input directory containing images"})},
            "optional": {
                "batch_size": ("INT", {"default": 0, "min": 0, "step": 1, "tooltip": "Number of images to load (0 = all images)"}),
                "start_from": ("INT", {"default": 1, "min": 1, "step": 1, "tooltip": "Start from Nth image (1 = first image)"}),
                "sort_method": (["sequential", "reverse", "random"], {"default": "sequential", "tooltip": "Image loading order: sequential/reverse/random"})
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("IMAGE", "IMAGE_PATH")
    OUTPUT_IS_LIST = (True, True)
    FUNCTION = "load_image_batch"
    CATEGORY = "🧪AILab/📝JoyCaption"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        if 'sort_method' in kwargs and kwargs['sort_method'] == "random":
            return float("NaN")
        return hash(frozenset(kwargs))

    def load_image_batch(self, image_dir: str, batch_size: int = 0, start_from: int = 1, sort_method: str = "sequential"):
        if not os.path.isdir(image_dir):
            raise FileNotFoundError(f"Directory '{image_dir}' cannot be found.")
            
        valid_extensions = ['.jpg', '.jpeg', '.png', '.webp']
        image_files = [f for f in os.listdir(image_dir) if any(f.lower().endswith(ext) for ext in valid_extensions)]
        
        if not image_files:
            raise FileNotFoundError(f"No valid images found in '{image_dir}'.")

        if sort_method == "sequential":
            image_files.sort()
        elif sort_method == "reverse":
            image_files.sort(reverse=True)
        elif sort_method == "random":
            import random
            random.shuffle(image_files)

        start_index = min(start_from - 1, len(image_files) - 1)
        image_files = image_files[start_index:]
        if batch_size > 0:
            image_files = image_files[:batch_size]
        
        images = []
        image_paths = []
        for filename in image_files:
            img_path = os.path.join(image_dir, filename)
            try:
                image = ImageOps.exif_transpose(Image.open(img_path)).convert("RGB")
                image = torch.from_numpy(np.array(image).astype(np.float32) / 255.0)[None,]
                images.append(image)
                image_paths.append(img_path)
            except Exception as e:
                print(f'\033[91mError loading {filename}: {str(e)}\033[0m')
                continue

        return (images, image_paths)

class CaptionSaver:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "string": ("STRING", {"forceInput": True}),
                "image_path": ("STRING", {"forceInput": True}),
            },
            "optional": {
                "image": ("IMAGE",),
                "custom_output_path": ("STRING", {"default": "", "multiline": True, "placeholder": "Custom output directory path. If empty, will use the directory of image_path"}),
                "custom_file_name": ("STRING", {"default": "", "multiline": False, "placeholder": "Custom filename (without extension)"}),
                "overwrite": ("BOOLEAN", {"default": True, "tooltip": "if true, will overwrite the existing txt file, if false, will add a number to the filename to make it unique"})
            }
        }
    
    RETURN_TYPES = ()
    FUNCTION = "save_caption"
    CATEGORY = "🧪AILab/📝JoyCaption"
    OUTPUT_NODE = True

    def get_unique_filename(self, base_path: Path) -> Path:
        """Get a unique filename by adding numbers if file exists."""
        if not base_path.exists():
            return base_path
        
        counter = 1
        while True:
            new_path = base_path.parent / f"{base_path.stem}_{counter:02d}{base_path.suffix}"
            if not new_path.exists():
                return new_path
            counter += 1

    def save_caption(self, string, image_path, image=None, custom_output_path="", custom_file_name="", overwrite=True):
        try:
            image_path = Path(image_path)
            save_dir = Path(custom_output_path.strip()) if custom_output_path.strip() else image_path.parent
            save_dir.mkdir(parents=True, exist_ok=True)
            
            try:
                base_name = custom_file_name.strip() if custom_file_name.strip() else image_path.stem
                txt_path = save_dir / f"{base_name}.txt"
                
                if not overwrite and txt_path.exists():
                    txt_path = self.get_unique_filename(txt_path)
                    base_name = txt_path.stem
                
                with open(txt_path, 'w', encoding='utf-8') as f:
                    f.write(string)
                print(f'\033[93m[{txt_path.name}]:\033[0m {string}')
                
                if image is not None and custom_output_path.strip():
                    try:
                        if isinstance(image, torch.Tensor):
                            while len(image.shape) > 3:
                                image = image.squeeze(0)
                            
                            image = (image.cpu().numpy() * 255).astype(np.uint8)
                            if image.shape[-1] != 3:
                                image = np.transpose(image, (1, 2, 0))
                            image = Image.fromarray(image)
                        
                        img_out_path = save_dir / f"{base_name}{image_path.suffix}"
                        image.save(img_out_path)
                        
                    except Exception as e:
                        print(f'\033[91mFailed to copy image: {str(e)}\033[0m')
                
            except Exception as e:
                print(f'\033[91mError saving caption: {str(e)}\033[0m')
            
        except Exception as e:
            print(f'\033[91mCritical error: {str(e)}\033[0m')
        
        return ()

NODE_CLASS_MAPPINGS = {
    "ImageBatchPath": ImageBatchPath,
    "CaptionSaver": CaptionSaver
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageBatchPath": "Image Batch Path 🖼️",
    "CaptionSaver": "Caption Saver 📝"
} 