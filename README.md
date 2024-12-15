# PaliGemma: A Multi-Modal Generative Model for Image and Text

## Overview

PaliGemma is a state-of-the-art multi-modal generative model designed to process both text and image inputs. It leverages deep learning techniques to generate meaningful text based on a given image and a text prompt. The core components of this repository include model training, inference, and a seamless way to deploy the trained model for generating outputs from multi-modal inputs. This repository contains the necessary code to load the model, run inference, and generate text conditioned on both image and text data.

The paper "PaliGemma: A Versatile 3B Vision-Language Model (VLM) for Transfer" introduces an open-source VLM designed to excel in a wide range of transfer learning tasks. The model combines the SigLIP-So400m vision encoder with the Gemma-2B language model to create a robust system capable of handling various open-world applications, such as segmentation, remote sensing, and OCR-related tasks. It has been evaluated on nearly 40 tasks and achieves strong performance on diverse benchmarks, including those outside traditional vision-language use cases, such as radiography report generation and fine-grained captioning.

The architecture focuses on efficient transfer across tasks, leveraging large-scale training across multiple resolutions (224px to 896px) and incorporating a wide variety of datasets. It represents a step forward in creating flexible VLMs for generalist applications.

## Features

- **Text and Image Processing**: The model can take a textual prompt and an image as input and generate coherent text based on both modalities.
- **Configurable Sampling Parameters**: Customize generation behavior with parameters like temperature, top-p, and token count.
- **Efficient Model Inference**: Supports both CPU and GPU execution, optimizing performance based on available hardware.
- **Pre-trained Weights Support**: The model supports loading pre-trained weights from `safetensors` files, ensuring easy deployment and fast inference.

## Directory Structure

- `modeling_gemma.py`: Contains the definition of the PaliGemma model architecture (`PaliGemmaForConditionalGeneration` and `PaliGemmaConfig`) for handling both text and image inputs.
- `inference.py`: The main script for running inference. This file takes in user parameters, loads the model, processes inputs, and generates text from the given image and prompt.
- `utils.py`: Utility functions for loading the model and associated tokenizer from the specified path, handling safetensors files, and managing device placement (CPU/GPU).
- `launch_inference.sh`: A bash script to automate the process of running inference, setting up required arguments such as the model path, prompt, image file, and sampling parameters.
- `test_images/`: Folder containing example images for running inference.

## Prerequisites

Before you begin, ensure you have the following dependencies installed:

- Python 3.x
- PyTorch (with CUDA support for GPU acceleration, if needed)
- `transformers` library for model and tokenizer handling
- `safetensors` for safe tensor loading
- PIL (Python Imaging Library) for image handling
- `fire` for creating CLI-based programs

To install the necessary Python packages, you can use the following:

```bash
pip install torch transformers safetensors fire pillow
```

## Installation

### Step 1: Clone the Repository

Clone this repository to your local machine using:

```bash
git clone https://github.com/yourusername/paligemma.git
cd paligemma
```

### Step 2: Set Up Your Model

Download the pre-trained model weights and place them in the directory specified in the `MODEL_PATH` variable (e.g., `"$HOME/projects/paligemma-weights/paligemma-3b-pt-224"`). You should have the model weights as `.safetensors` files and the `config.json` for the model configuration.

### Step 3: Run Inference

You can run the model inference by either using the `inference.py` script directly or the provided bash script `launch_inference.sh`.

#### Running with the Bash Script

The simplest way to run the inference is by using the `launch_inference.sh` script, which automatically sets up the parameters. To run the inference, simply execute the following:

```bash
bash launch_inference.sh
```

This will trigger the inference with the specified prompt, image file, and generation settings.

#### Running Directly with Python

You can also run inference manually by invoking the `inference.py` script directly:

```bash
python inference.py --model_path "$MODEL_PATH" \
                    --prompt "your prompt here" \
                    --image_file_path "path/to/image.jpeg" \
                    --max_tokens_to_generate 100 \
                    --temperature 0.8 \
                    --top_p 0.9 \
                    --do_sample True \
                    --only_cpu False
```

This will load the model, process the input image, and prompt, and generate text output.

### Available Parameters for Inference

- `model_path`: Path to the pre-trained model weights directory.
- `prompt`: A string prompt to guide the model's text generation.
- `image_file_path`: Path to the input image.
- `max_tokens_to_generate`: Maximum number of tokens the model should generate (default: 100).
- `temperature`: Sampling temperature for generation. A lower value makes the model's output more deterministic (default: 0.8).
- `top_p`: Nucleus sampling parameter (top-p). A value between 0 and 1 (default: 0.9).
- `do_sample`: Whether to sample (True) or take the most likely next token (False).
- `only_cpu`: Use CPU for inference. Set to `True` if no GPU support is available.

### Example Output

Given the input prompt `"this building is "` and an image of a building, the model may generate descriptive text based on both the visual input and the prompt. For example:

```
this building is a modern architectural structure with large glass windows and a steel frame. It stands tall against the skyline, with a sleek and minimalist design.
```

## Key Components

### 1. `modeling_gemma.py`

This file defines the PaliGemma model architecture, including the configuration and model class. It handles both the text and image inputs through a shared transformer architecture, where the image features are processed as tokens along with text tokens.

### 2. `inference.py`

The script orchestrates the inference process by loading the model and tokenizer, handling the input preprocessing, and generating output tokens using the model. It can work on both CPU and GPU, depending on the system's available hardware.

### 3. `utils.py`

The utility functions help load the model from the specified path and ensure that the weights are loaded correctly from `.safetensors` files. It also handles moving model inputs to the correct device (CPU/GPU).

### 4. `launch_inference.sh`

This script provides an easy-to-use interface to run inference without manually setting parameters. It ensures that all necessary arguments are passed to `inference.py` automatically.

## Contributing

Contributions are welcome! Feel free to submit issues or pull requests. If you'd like to contribute a feature or fix a bug, please follow these steps:

1. Fork the repository.
2. Create a new branch for your changes.
3. Commit your changes with a descriptive message.
4. Push your changes to your forked repository.
5. Submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### Conclusion

The PaliGemma repository provides an easy-to-use framework for running multi-modal inference tasks using text and image data. With configurable parameters for text generation and seamless integration with GPU/CPU resources, this model is highly adaptable for various creative and practical applications.

## Additional Resources

https://www.youtube.com/watch?v=vAmKB7iPkWw
