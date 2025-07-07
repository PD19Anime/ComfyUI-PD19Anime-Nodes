#
# Author: PD19 Anime (X: @PD19_Anime)
# Version: 1.2
# Description: A powerful suite of nodes for ComfyUI to dynamically load prompts and images. This pack is especially focused on   batch processing from a directory and includes two main nodes: Advanced Prompt & Image Loader (Multiple) and   Advanced Prompt & Image Loader (single)..
#

import os
import glob
import random
import hashlib
import json
import torch
import numpy as np
from PIL import Image, ImageOps
import folder_paths

# --- Common Helper Functions ---
def get_file_info(file_path):
    try:
        stats = os.stat(file_path)
        return f"{stats.st_mtime}-{stats.st_size}"
    except (OSError, FileNotFoundError):
        return float("NaN")

def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

# ======================================================================
# == Single Image Loader Node
# ======================================================================
class AdvancedLoaderSingle:
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        return {
            "required": {
                "image": (sorted(files), {"image_upload": True}),
                "swap_prompts": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "IMAGE",)
    RETURN_NAMES = ("positive_prompt", "negative_prompt", "image",)
    FUNCTION = "load_and_read"
    CATEGORY = "PD19🪐Anime-Nodes"

    def load_and_read(self, image, swap_prompts):
        image_path = folder_paths.get_annotated_filepath(image)
        positive_prompt, negative_prompt, image_tensor = "", "", None
        try:
            pil_image = Image.open(image_path)
            pil_image = ImageOps.exif_transpose(pil_image)
            image_tensor = pil2tensor(pil_image.convert("RGB"))
            if image_path.lower().endswith('.png'):
                info = pil_image.info or {}
                prompt_text, workflow_text = info.get('prompt'), info.get('workflow')
                if workflow_text:
                    workflow = json.loads(workflow_text)
                    nodes = workflow.get('nodes', [])
                    prompts = [n['widgets_values'][0] for n in nodes if n.get('type') == 'CLIPTextEncode' and n.get('widgets_values')]
                    if len(prompts) > 0: positive_prompt = prompts[0]
                    if len(prompts) > 1: negative_prompt = prompts[1]
                elif prompt_text:
                    prompt_data = json.loads(prompt_text)
                    for node in prompt_data.values():
                        if node.get('class_type') == 'CLIPTextEncode':
                            positive_prompt = node.get('inputs', {}).get('text', '')
                            break
        except Exception as e:
            print(f"AdvancedLoaderSingle: Error processing file '{image_path}': {e}")
            return ("", "", None)
        
        return (positive_prompt, negative_prompt, image_tensor) if swap_prompts else (negative_prompt, positive_prompt, image_tensor)

    @classmethod
    def IS_CHANGED(s, image, swap_prompts):
        return get_file_info(folder_paths.get_annotated_filepath(image))

# ======================================================================
# == Multiple Image Loader Node
# ======================================================================
class AdvancedLoaderMultiple:
    incremental_counters = {}
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "path": ("STRING", {"default": ''}),
                "pattern": ("STRING", {"default": '*'}),
                "mode": (["single_image", "incremental_image", "random"],),
                "index": ("INT", {"default": 0, "min": 0, "step": 1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                # Core Fix: Move swap_prompts to the end of the dictionary
                "swap_prompts": ("BOOLEAN", {"default": False}),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING", "IMAGE",)
    RETURN_NAMES = ("positive_prompt", "negative_prompt", "image",)
    FUNCTION = "read_prompt_and_image"
    CATEGORY = "PD19🪐Anime-Nodes"

    def read_prompt_and_image(self, path, pattern, mode, index, seed, swap_prompts):
        # Note: The order of function parameters must also be adjusted to match INPUT_TYPES!
        default_return = ("", "", None)
        if not path or not isinstance(path, str): return default_return
        abs_path = os.path.abspath(os.path.normpath(path.strip()))
        if not os.path.isdir(abs_path): return default_return
        search_path = os.path.join(abs_path, pattern)
        all_files = sorted(glob.glob(search_path, recursive=True))
        image_formats = ['.png', '.jpg', '.jpeg', '.webp', '.bmp', '.gif']
        image_paths = [p for p in all_files if os.path.splitext(p)[1].lower() in image_formats and os.path.isfile(p)]
        if not image_paths: return default_return
        
        selected_path = None
        if mode == 'single_image':
            selected_path = image_paths[index] if index < len(image_paths) else image_paths[0]
        elif mode == 'incremental_image':
            counter_key = f"{abs_path}_{pattern}"
            counter = self.incremental_counters.get(counter_key, 0)
            if counter >= len(image_paths): counter = 0
            selected_path = image_paths[counter]
            self.incremental_counters[counter_key] = counter + 1
        elif mode == 'random':
            random.seed(seed)
            selected_path = random.choice(image_paths)
        if not selected_path: return default_return

        positive_prompt, negative_prompt, image_tensor = "", "", None
        try:
            pil_image = Image.open(selected_path)
            pil_image = ImageOps.exif_transpose(pil_image)
            image_tensor = pil2tensor(pil_image.convert("RGB"))
            if selected_path.lower().endswith('.png'):
                info = pil_image.info or {}
                prompt_text, workflow_text = info.get('prompt'), info.get('workflow')
                if workflow_text:
                    workflow = json.loads(workflow_text)
                    nodes = workflow.get('nodes', [])
                    prompts = [n['widgets_values'][0] for n in nodes if n.get('type') == 'CLIPTextEncode' and n.get('widgets_values')]
                    if len(prompts) > 0: positive_prompt = prompts[0]
                    if len(prompts) > 1: negative_prompt = prompts[1]
                elif prompt_text:
                    prompt_data = json.loads(prompt_text)
                    for node in prompt_data.values():
                        if node.get('class_type') == 'CLIPTextEncode':
                            positive_prompt = node.get('inputs', {}).get('text', '')
                            break
        except Exception as e:
            print(f"AdvancedLoaderMultiple: Error processing file '{selected_path}': {e}")
            return default_return
        
        return (positive_prompt, negative_prompt, image_tensor) if swap_prompts else (negative_prompt, positive_prompt, image_tensor)

    @classmethod
    def IS_CHANGED(s, path, pattern, mode, index, seed, swap_prompts):
        # The order of function parameters must also be adjusted here
        if mode in ['incremental_image', 'random']: return float("NaN")
        if not path or not isinstance(path, str): return float("NaN")
        abs_path = os.path.abspath(os.path.normpath(path.strip()))
        if not os.path.isdir(abs_path): return float("NaN")
        search_path = os.path.join(abs_path, pattern)
        all_files = sorted(glob.glob(search_path, recursive=True))
        image_formats = ['.png', '.jpg', '.jpeg', '.webp', '.bmp', '.gif']
        image_paths = [p for p in all_files if os.path.splitext(p)[1].lower() in image_formats and os.path.isfile(p)]
        if not image_paths or index >= len(image_paths): return float("NaN")
        return get_file_info(image_paths[index])

# --- Node Registration ---
NODE_CLASS_MAPPINGS = {
    "AdvancedLoaderSingle": AdvancedLoaderSingle,
    "AdvancedLoaderMultiple": AdvancedLoaderMultiple
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AdvancedLoaderSingle": "Advanced Prompt & Image Loader (single) 🪐",
    "AdvancedLoaderMultiple": "Advanced Prompt & Image Loader (Multiple) 🪐"
}
