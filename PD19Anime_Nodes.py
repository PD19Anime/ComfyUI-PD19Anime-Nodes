#
# Author: PD19 Anime (X: @PD19_Anime)
# Version: 1.2.2
# Description: A powerful ComfyUI node suite featuring the Advanced Prompt & Image Loader (Multiple) node, designed for batch extracting and saving image prompts. Automatically reads prompt information from multiple images and passes it to other nodes for processing, plus includes other workflow utility nodes.
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

def extract_prompts_from_workflow(workflow):
    """
    Extract prompts from workflow by analyzing node connections
    Returns (positive_prompt, negative_prompt)
    """
    nodes = workflow.get('nodes', [])
    
    # Find all CLIPTextEncode nodes
    clip_nodes = {}
    for node in nodes:
        if node.get('type') == 'CLIPTextEncode' and node.get('widgets_values'):
            node_id = node.get('id')
            clip_nodes[node_id] = node.get('widgets_values')[0]
    
    if not clip_nodes:
        return "", ""
    
    # Find sampler nodes and their connections
    positive_prompt, negative_prompt = "", ""
    
    # Look for any node with positive/negative inputs
    for node in nodes:
        inputs = node.get('inputs', [])
        for inp in inputs:
            if inp.get('name') == 'positive' and inp.get('link') is not None:
                link_id = inp['link']
                source_node = find_source_node(nodes, link_id)
                if source_node and source_node.get('id') in clip_nodes:
                    positive_prompt = clip_nodes[source_node.get('id')]
            elif inp.get('name') == 'negative' and inp.get('link') is not None:
                link_id = inp['link']
                source_node = find_source_node(nodes, link_id)
                if source_node and source_node.get('id') in clip_nodes:
                    negative_prompt = clip_nodes[source_node.get('id')]
        
        # If we found both prompts, we can break
        if positive_prompt and negative_prompt:
            break
    
    # Fallback: if we couldn't find connections, use the old method
    if not positive_prompt and not negative_prompt:
        clip_values = list(clip_nodes.values())
        if len(clip_values) > 0:
            positive_prompt = clip_values[0]
        if len(clip_values) > 1:
            negative_prompt = clip_values[1]
    
    return positive_prompt, negative_prompt

def find_source_node(nodes, link_id):
    """Find the source node for a given link ID"""
    # We need to find the node that has this link_id in its outputs
    for node in nodes:
        outputs = node.get('outputs', [])
        for output in outputs:
            links = output.get('links', [])
            if isinstance(links, list) and link_id in links:
                return node
    return None

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

    RETURN_TYPES = ("STRING", "STRING", "IMAGE", "STRING")
    RETURN_NAMES = ("positive_prompt", "negative_prompt", "image", "filename_prefix")
    FUNCTION = "load_and_read"
    CATEGORY = "PD19🪐Anime-Nodes"

    def load_and_read(self, image, swap_prompts):
        image_path = folder_paths.get_annotated_filepath(image)
        positive_prompt, negative_prompt, image_tensor = "", "", None
        
        # Extract filename prefix (without extension)
        filename_prefix = os.path.splitext(os.path.basename(image_path))[0]
        
        try:
            pil_image = Image.open(image_path)
            pil_image = ImageOps.exif_transpose(pil_image)
            image_tensor = pil2tensor(pil_image.convert("RGB"))
            if image_path.lower().endswith('.png'):
                info = pil_image.info or {}
                prompt_text, workflow_text = info.get('prompt'), info.get('workflow')
                if workflow_text:
                    workflow = json.loads(workflow_text)
                    positive_prompt, negative_prompt = extract_prompts_from_workflow(workflow)
                elif prompt_text:
                    prompt_data = json.loads(prompt_text)
                    for node in prompt_data.values():
                        if node.get('class_type') == 'CLIPTextEncode':
                            positive_prompt = node.get('inputs', {}).get('text', '')
                            break
        except Exception as e:
            print(f"AdvancedLoaderSingle: Error processing file '{image_path}': {e}")
            return ("", "", None, filename_prefix)
        
        if swap_prompts:
            return (negative_prompt, positive_prompt, image_tensor, filename_prefix)
        else:
            return (positive_prompt, negative_prompt, image_tensor, filename_prefix)

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
                "mode": (["single_image", "incremental_image", "random"], {"default": "incremental_image"}),
                "index": ("INT", {"default": 0, "min": 0, "step": 1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                # Core Fix: Move swap_prompts to the end of the dictionary
                "swap_prompts": ("BOOLEAN", {"default": False}),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING", "IMAGE", "STRING")
    RETURN_NAMES = ("positive_prompt", "negative_prompt", "image", "filename_prefix")
    FUNCTION = "read_prompt_and_image"
    CATEGORY = "PD19🪐Anime-Nodes"

    def read_prompt_and_image(self, path, pattern, mode, index, seed, swap_prompts):
        # Note: The order of function parameters must also be adjusted to match INPUT_TYPES!
        default_return = ("", "", None, "")
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

        # Extract filename prefix (without extension)
        filename_prefix = os.path.splitext(os.path.basename(selected_path))[0]
        
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
                    positive_prompt, negative_prompt = extract_prompts_from_workflow(workflow)
                elif prompt_text:
                    prompt_data = json.loads(prompt_text)
                    for node in prompt_data.values():
                        if node.get('class_type') == 'CLIPTextEncode':
                            positive_prompt = node.get('inputs', {}).get('text', '')
                            break
        except Exception as e:
            print(f"AdvancedLoaderMultiple: Error processing file '{selected_path}': {e}")
            return default_return
        
        if swap_prompts:
            return (negative_prompt, positive_prompt, image_tensor, filename_prefix)
        else:
            return (positive_prompt, negative_prompt, image_tensor, filename_prefix)

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

# ======================================================================
# == Advanced Empty Latent Image Node
# ======================================================================
class AdvancedEmptyLatentImage:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "preset": (["Portrait (832x1216)", "Landscape (1216x832)", "Square (1024x1024)", "Manual"], {"default": "Portrait (832x1216)"}),
                "width": ("INT", {"default": 832, "min": 64, "max": 8192, "step": 8}),
                "height": ("INT", {"default": 1216, "min": 64, "max": 8192, "step": 8}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 64})
            }
        }

    RETURN_TYPES = ("LATENT", "INT", "INT")
    RETURN_NAMES = ("latent", "width", "height")
    FUNCTION = "generate"
    CATEGORY = "PD19🪐Anime-Nodes"

    def generate(self, preset, width, height, batch_size):
        # Apply preset dimensions
        if preset == "Portrait (832x1216)":
            actual_width, actual_height = 832, 1216
        elif preset == "Landscape (1216x832)":
            actual_width, actual_height = 1216, 832
        elif preset == "Square (1024x1024)":
            actual_width, actual_height = 1024, 1024
        else:  # Manual
            actual_width, actual_height = width, height
        
        # Generate latent tensor
        latent = torch.zeros([batch_size, 4, actual_height // 8, actual_width // 8], device="cpu")
        
        return ({"samples": latent}, actual_width, actual_height)

# ======================================================================
# == Advanced Prompt Saver Node
# ======================================================================
class AdvancedPromptSaver:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "positive_prompt": ("STRING", {"forceInput": True}),
                "negative_prompt": ("STRING", {"forceInput": True}),
                "filename_prefix": ("STRING", {"default": ""}),
                "output_path": ("STRING", {"default": ""}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "save_prompts"
    CATEGORY = "PD19🪐Anime-Nodes"
    OUTPUT_NODE = True

    @classmethod
    def VALIDATE_INPUTS(s, **kwargs):
        return True

    def save_prompts(self, positive_prompt, negative_prompt, filename_prefix, output_path, unique_id=None, extra_pnginfo=None):
        try:
            # Handle output path
            if not output_path or not isinstance(output_path, str):
                output_path = os.getcwd()  # Use current working directory if not specified
            
            # Ensure output directory exists
            output_path = os.path.abspath(os.path.normpath(output_path.strip()))
            os.makedirs(output_path, exist_ok=True)
            
            # Handle filename prefix
            if not filename_prefix or not isinstance(filename_prefix, str) or filename_prefix.strip() == "":
                # Generate default filename with timestamp
                from datetime import datetime
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename_prefix = timestamp
            
            # Clean filename prefix (remove invalid characters)
            import re
            clean_prefix = re.sub(r'[<>:"/\\|?*]', '_', filename_prefix.strip())
            
            # Generate filename
            filename = f"{clean_prefix}_prompts.txt"
            filepath = os.path.join(output_path, filename)
            
            # Handle duplicate filenames
            counter = 1
            original_filepath = filepath
            while os.path.exists(filepath):
                base_name = f"{clean_prefix}_prompts_{counter}.txt"
                filepath = os.path.join(output_path, base_name)
                counter += 1
            
            # Format content
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            content = f"""=== POSITIVE PROMPT ===
{positive_prompt}

=== NEGATIVE PROMPT ===
{negative_prompt}

=== METADATA ===
Generated: {timestamp}
Original Filename: {filename_prefix}
"""
            
            # Write file
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            
            # Prepare status messages for display
            filename_only = os.path.basename(filepath)
            status_messages = [
                f"✅ Prompts Saved Successfully",
                f"File: {filename_only}",
                f"Path: {output_path}"
            ]
            
            success_message = f"✅ Prompts saved successfully to: {filepath}"
            print(success_message)
            
            return {
                "ui": {"text": (status_messages, success_message)},
                "result": (success_message,)
            }
            
        except Exception as e:
            error_messages = [
                "❌ Save Failed",
                f"Error: {str(e)}"
            ]
            error_message = f"❌ Error saving prompts: {str(e)}"
            print(error_message)
            
            return {
                "ui": {"text": (error_messages, error_message)},
                "result": (error_message,)
            }

# --- Node Registration ---
NODE_CLASS_MAPPINGS = {
    "AdvancedLoaderSingle": AdvancedLoaderSingle,
    "AdvancedLoaderMultiple": AdvancedLoaderMultiple,
    "AdvancedEmptyLatentImage": AdvancedEmptyLatentImage,
    "AdvancedPromptSaver": AdvancedPromptSaver
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AdvancedLoaderSingle": "Advanced Prompt & Image Loader (single) 🪐",
    "AdvancedLoaderMultiple": "Advanced Prompt & Image Loader (Multiple) 🪐",
    "AdvancedEmptyLatentImage": "Advanced Empty Latent Image 🪐",
    "AdvancedPromptSaver": "Advanced Prompt Saver 🪐"
}
