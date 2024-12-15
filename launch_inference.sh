#!/bin/bash

MODEL_PATH="paligemma-3b-pt-224"
PROMPT="this building is " # The model will generate text based on this prompt and the image provided in IMAGE_FILE_PATH
IMAGE_FILE_PATH="./test_images/image.jpg"
MAX_TOKENS_TO_GENERATE=100 # Controls the maximum number of tokens the model will generate. It's set to 100 in this case
TEMPERATURE=0.8 # Controls the randomness of token generation. A value of 0.8 will give more diverse but still coherent text generation
TOP_P=0.9 # A sampling parameter for top-p (nucleus) sampling, set to 0.9. This means the model will sample from the top 90% of the probability distribution
DO_SAMPLE="False" # This parameter (False) indicates that deterministic sampling is used, meaning the model will always pick the most probable next token rather than sampling randomly
ONLY_CPU="False" # Set to "False", which means that if CUDA or MPS is available, the model will use the GPU or Apple’s Metal API for inference


# Inference Workflow:
# - The user specifies an image and a prompt.
# - The model processes the image and prompt together and generates text based on the image content.
# - The text is sampled using techniques like temperature scaling and top-p sampling.
# - The final output is printed.
python ./src/inference.py \
    --model_path "$MODEL_PATH" \
    --prompt "$PROMPT" \
    --image_file_path "$IMAGE_FILE_PATH" \
    --max_tokens_to_generate $MAX_TOKENS_TO_GENERATE \
    --temperature $TEMPERATURE \
    --top_p $TOP_P \
    --do_sample $DO_SAMPLE \
    --only_cpu $ONLY_CPU \

