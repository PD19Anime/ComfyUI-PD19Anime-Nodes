# ComfyUI-PD19Anime-Nodes
A suite of nodes for dynamically reading prompts and images.

This node is specifically designed for batch high-resolution image upscaling, saving you a significant amount of time and effort. The original purpose of creating this node was to eliminate the need to manually copy and paste the prompt for each image when processing a large number of files. With this node, you can set up batch operations and free your hands for other tasks, greatly improving your workflow efficiency.

In addition to high-resolution upscaling, this node can be flexibly integrated into your custom ComfyUI workflows to automate a variety of image processing tasks.

This node combines the functionality of the Load Image Batch node from the was-node-suit package and parts of the SD Prompt Reader from the comfyui-prompt-reader-node package. The original MIT licenses have been incorporated into this project's LICENSE file.


![CleanShot 2025-07-07 at 19 28 50@2x](https://github.com/user-attachments/assets/c4b17582-7f5e-4c4e-aabf-4c12d5dcb07d)

Install

cd to the custom_node folder
Clone this repo
`git clone --recursive https://github.com/receyuki/comfyui-prompt-reader-node.git`


Features

Advanced Prompt & Image Loader (Multiple) provides the following features:
Input Parameters

    path:
    The absolute or relative path to the folder containing your images.

    pattern:
    Filename filter (e.g., *.png, prefix_*.jpg). * matches all files.

    mode:

        single_image: Loads the image at the specified index.

        incremental_image: Loads the next image in the folder on each run.

        random: Loads a random image based on the provided seed.

    index / seed:
    Numeric parameter for the corresponding mode.

    swap_prompts:
    A toggle switch. If your positive and negative prompts are swapped, enable this to correct the output.

Output Parameters

    positive_prompt:
    The positive prompt read from the image metadata.

    negative_prompt:
    The negative prompt read from the image metadata.

    image:
    The image data itself.


How to Use

    1.Set the path to the image folder.

    2.Check the number of images in the folder and adjust the Batch Count value in ComfyUI to match the number of images.

    3.Configure the loading mode and other parameters as needed.

    4.Start the workflow.

![CleanShot 2025-07-07 at 23 42 04@2x](https://github.com/user-attachments/assets/10211f62-4cb3-4673-bade-f9da96789d16)


Notes

  This node may not be fully compatible with all complex workflows that use large numbers of custom nodes. If your images were generated using many custom nodes, pipe-type nodes, or third-party Ksampler nodes, it may not correctly read prompts or may swap positive and negative prompts (which can be corrected with the swap_prompts option).


Usage Suggestions

Simply add this node to your workflow, configure the folder path and batch processing rules, and it will automatically use the prompt from each image for your operations, greatly reducing repetitive manual work. Feel free to expand or integrate it with other automation processes as needed!
