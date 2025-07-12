#
# Author: PD19 Anime (X: @PD19_Anime)
# Version: 1.2.3
# Description: A powerful ComfyUI node suite featuring the Advanced Prompt & Image Loader (Multiple) node, designed for batch extracting and saving image prompts. Automatically reads prompt information from multiple images and passes it to other nodes for processing, plus includes other workflow utility nodes. Now includes the Advanced Model Loader with directory-based model selection and integrated LoRA support, and Advanced Random Noise node with current seed display and copy functionality.
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
import comfy.sd
import comfy.utils
import comfy.sample

# --- Common Helper Functions ---
def get_file_info(file_path):
    try:
        stats = os.stat(file_path)
        return f"{stats.st_mtime}-{stats.st_size}"
    except (OSError, FileNotFoundError):
        return float("NaN")

def get_image_files_with_folders(directory):
    """Get all image files in directory with folder structure preserved"""
    if not os.path.exists(directory):
        return []
    
    image_files = []
    image_formats = ['.png', '.jpg', '.jpeg', '.webp', '.bmp', '.gif']
    
    for root, dirs, files in os.walk(directory):
        # Sort directories and files for consistent ordering
        dirs.sort()
        files.sort()
        
        for file in files:
            if any(file.lower().endswith(ext) for ext in image_formats):
                full_path = os.path.join(root, file)
                # Get relative path from input directory
                rel_path = os.path.relpath(full_path, directory)
                # Normalize path separators for cross-platform compatibility
                rel_path = rel_path.replace('\\', '/')
                image_files.append(rel_path)
    
    return sorted(image_files)

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

def get_folder_structure(folder_name):
    """Get hierarchical folder structure for model selection"""
    try:
        folders = folder_paths.get_folder_paths(folder_name)
        if not folders:
            return ["None"]
        
        files = []
        for folder in folders:
            if os.path.exists(folder):
                for root, dirs, file_list in os.walk(folder):
                    for file in file_list:
                        if any(file.lower().endswith(ext) for ext in ['.ckpt', '.pt', '.pth', '.safetensors']):
                            file_path = os.path.join(root, file)
                            rel_path = os.path.relpath(file_path, folder)
                            files.append(rel_path)
        
        if not files:
            return ["None"]
        
        return sorted(files)
    except Exception as e:
        print(f"Error getting folder structure for {folder_name}: {e}")
        return ["None"]

def get_hierarchical_structure(folder_name):
    """Get hierarchical folder structure for tree widget"""
    try:
        folders = folder_paths.get_folder_paths(folder_name)
        if not folders:
            return {}
        
        structure = {}
        
        for folder in folders:
            if os.path.exists(folder):
                for root, dirs, file_list in os.walk(folder):
                    for file in file_list:
                        if any(file.lower().endswith(ext) for ext in ['.ckpt', '.pt', '.pth', '.safetensors']):
                            file_path = os.path.join(root, file)
                            rel_path = os.path.relpath(file_path, folder)
                            
                            # Get directory path
                            dir_path = os.path.dirname(rel_path)
                            
                            if dir_path == '':
                                # Root level file
                                if 'root' not in structure:
                                    structure['root'] = []
                                structure['root'].append(file)
                            else:
                                # File in subdirectory
                                dir_parts = dir_path.split(os.sep)
                                current_level = structure
                                
                                # Navigate through directory structure
                                for i, part in enumerate(dir_parts):
                                    if part not in current_level:
                                        current_level[part] = {}
                                    
                                    if i == len(dir_parts) - 1:
                                        # Last part, add files
                                        if '_files' not in current_level[part]:
                                            current_level[part]['_files'] = []
                                        current_level[part]['_files'].append(file)
                                    else:
                                        current_level = current_level[part]
        
        return structure
    except Exception as e:
        print(f"Error getting hierarchical structure for {folder_name}: {e}")
        return {}

def get_categorized_models(folder_name):
    """Get models organized by subdirectory"""
    models = {}
    folders = folder_paths.get_folder_paths(folder_name)
    
    if not folders:
        return models
    
    for folder in folders:
        if os.path.exists(folder):
            for root, dirs, files in os.walk(folder):
                for file in files:
                    if any(file.lower().endswith(ext) for ext in ['.ckpt', '.pt', '.pth', '.safetensors']):
                        file_path = os.path.join(root, file)
                        rel_path = os.path.relpath(file_path, folder)
                        
                        # Determine category from subfolder
                        if os.path.dirname(rel_path):
                            category = os.path.dirname(rel_path).split(os.sep)[0]
                        else:
                            category = "Root"
                        
                        if category not in models:
                            models[category] = []
                        models[category].append(rel_path)
    
    # Sort files within each category
    for category in models:
        models[category].sort()
    
    return models

# ======================================================================
# == Single Image Loader Node
# ======================================================================
class AdvancedLoaderSingle:
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = get_image_files_with_folders(input_dir)
        if not files:
            files = ["None"]
        return {
            "required": {
                "image": (files, {"image_upload": True}),
                "swap_prompts": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "IMAGE", "STRING")
    RETURN_NAMES = ("positive_prompt", "negative_prompt", "image", "filename_prefix")
    FUNCTION = "load_and_read"
    CATEGORY = "PD19🪐Anime-Nodes"

    def load_and_read(self, image, swap_prompts):
        # Handle folder-structured paths
        if image == "None":
            return ("", "", None, "")
        
        input_dir = folder_paths.get_input_directory()
        image_path = os.path.join(input_dir, image)
        
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
        if image == "None":
            return float("NaN")
        input_dir = folder_paths.get_input_directory()
        image_path = os.path.join(input_dir, image)
        return get_file_info(image_path)

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

# ======================================================================
# == Advanced Model Loader Node
# ======================================================================
class AdvancedModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ckpt_name": (get_folder_structure("checkpoints"), {}),
                "lora1": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "lora1_name": (get_folder_structure("loras"), {}),
                "lora1_strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "lora2": ("BOOLEAN", {"default": False}),
                "lora2_name": (get_folder_structure("loras"), {}),
                "lora2_strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
            }
        }
    
    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    FUNCTION = "load_models"
    CATEGORY = "PD19🪐Anime-Nodes"
    
    def load_models(self, ckpt_name, lora1, lora1_name=None, lora1_strength=1.0, lora2=False, lora2_name=None, lora2_strength=1.0):
        # Validate checkpoint
        if not ckpt_name or ckpt_name == "None":
            raise ValueError("No checkpoint selected")
        
        # Load checkpoint
        try:
            ckpt_path = folder_paths.get_full_path_or_raise("checkpoints", ckpt_name)
            out = comfy.sd.load_checkpoint_guess_config(ckpt_path, output_vae=True, output_clip=True, 
                                                        embedding_directory=folder_paths.get_folder_paths("embeddings"))
            model, clip, vae = out[:3]
            print(f"AdvancedModelLoader: Successfully loaded checkpoint '{ckpt_name}'")
        except Exception as e:
            print(f"AdvancedModelLoader: Error loading checkpoint '{ckpt_name}': {e}")
            raise
        
        # Apply LoRA1 if enabled
        if lora1 and lora1_name and lora1_name.strip() and lora1_name != "None":
            try:
                lora1_path = folder_paths.get_full_path_or_raise("loras", lora1_name)
                lora1_data = comfy.utils.load_torch_file(lora1_path, safe_load=True)
                model, clip = comfy.sd.load_lora_for_models(model, clip, lora1_data, lora1_strength, lora1_strength)
                print(f"AdvancedModelLoader: Successfully applied LoRA1 '{lora1_name}' with strength {lora1_strength}")
            except Exception as e:
                print(f"AdvancedModelLoader: Error loading LoRA1 '{lora1_name}': {e}")
        
        # Apply LoRA2 if enabled
        if lora2 and lora2_name and lora2_name.strip() and lora2_name != "None":
            try:
                lora2_path = folder_paths.get_full_path_or_raise("loras", lora2_name)
                lora2_data = comfy.utils.load_torch_file(lora2_path, safe_load=True)
                model, clip = comfy.sd.load_lora_for_models(model, clip, lora2_data, lora2_strength, lora2_strength)
                print(f"AdvancedModelLoader: Successfully applied LoRA2 '{lora2_name}' with strength {lora2_strength}")
            except Exception as e:
                print(f"AdvancedModelLoader: Error loading LoRA2 '{lora2_name}': {e}")
        
        return (model, clip, vae)
    
    @classmethod
    def IS_CHANGED(s, ckpt_name, lora1, lora2, lora1_name=None, lora1_strength=1.0, lora2_name=None, lora2_strength=1.0):
        hash_input = f"{ckpt_name}_{lora1}_{lora2}_{lora1_name}_{lora1_strength}_{lora2_name}_{lora2_strength}"
        return hashlib.md5(hash_input.encode()).hexdigest()

# ======================================================================
# == Advanced Random Noise Node
# ======================================================================
class AdvancedRandomNoise:
    """Advanced Random Noise node with current seed display and copy functionality"""
    
    # Class variable to track current seed per instance
    _current_seeds = {}
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "noise_seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffffffffffff,
                    "control_after_generate": True
                }),
            }
        }
    
    RETURN_TYPES = ("NOISE", "INT")
    RETURN_NAMES = ("noise", "current_seed")
    FUNCTION = "generate_noise"
    CATEGORY = "PD19🪐Anime-Nodes"
    
    def generate_noise(self, noise_seed):
        """Generate random noise and track the current seed"""
        try:
            # Store current seed for this instance
            instance_id = id(self)
            AdvancedRandomNoise._current_seeds[instance_id] = noise_seed
            
            # Create noise object
            noise = AdvancedNoise(noise_seed)
            
            print(f"AdvancedRandomNoise: Generated noise with seed {noise_seed}")
            
            return (noise, noise_seed)
            
        except Exception as e:
            print(f"AdvancedRandomNoise: Error generating noise: {e}")
            # Return default values on error
            return (AdvancedNoise(0), 0)
    
    @classmethod
    def IS_CHANGED(s, noise_seed):
        """Handle change detection for the node"""
        # Return the seed value - cache will be invalidated when seed changes
        # This allows fixed mode to work properly while randomize mode still functions
        return noise_seed

class AdvancedNoise:
    """Advanced Noise object that generates random noise from a seed"""
    
    def __init__(self, seed):
        self.seed = seed
    
    def generate_noise(self, input_latent):
        """Generate noise for the given latent"""
        try:
            latent_image = input_latent["samples"]
            batch_inds = input_latent.get("batch_index", None)
            
            # Use ComfyUI's built-in noise generation
            noise = comfy.sample.prepare_noise(latent_image, self.seed, batch_inds)
            
            return noise
            
        except Exception as e:
            print(f"AdvancedNoise: Error generating noise: {e}")
            # Return zero noise on error
            return torch.zeros_like(input_latent["samples"])
    
    def __call__(self, input_latent):
        """Make the noise object callable"""
        return self.generate_noise(input_latent)

# --- Node Registration ---
NODE_CLASS_MAPPINGS = {
    "AdvancedLoaderSingle": AdvancedLoaderSingle,
    "AdvancedLoaderMultiple": AdvancedLoaderMultiple,
    "AdvancedEmptyLatentImage": AdvancedEmptyLatentImage,
    "AdvancedPromptSaver": AdvancedPromptSaver,
    "AdvancedModelLoader": AdvancedModelLoader,
    "AdvancedRandomNoise": AdvancedRandomNoise
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AdvancedLoaderSingle": "Advanced Prompt & Image Loader (single) 🪐",
    "AdvancedLoaderMultiple": "Advanced Prompt & Image Loader (Multiple) 🪐",
    "AdvancedEmptyLatentImage": "Advanced Empty Latent Image 🪐",
    "AdvancedPromptSaver": "Advanced Prompt Saver 🪐",
    "AdvancedModelLoader": "Advanced Model Loader 🪐",
    "AdvancedRandomNoise": "Advanced Random Noise 🪐"
}
