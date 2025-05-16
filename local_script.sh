#!/bin/bash -l

source ./create_env_vars.sh

in_model="google/gemma-3-4b-it"
in_datasets="DollyDataset ShareGPT Alpaca Mbpp Apps DS1000 BigCodeBench" # DollyDataset ShareGPT Alpaca Mbpp Apps DS1000 BigCodeBench
in_loglevel="DEBUG"

eval python3 -m scripts.run_llm_response_generator --in_model="${in_model}" --in_device "auto" --datasets "${in_datasets}" --log_level "${in_loglevel}"


# HF_API_TOKEN_MODELS = [
#     "mistralai/Mistral-7B-Instruct-v0.3",
#     "mistralai/Mixtral-8x7B-Instruct-v0.1",
#     "mistralai/Ministral-8B-Instruct-2410",
#     "mistralai/Mistral-Nemo-Instruct-2407",

#     "meta-llama/Llama-3.2-1B-Instruct",
#     "meta-llama/Llama-3.2-3B-Instruct",
#     "meta-llama/Llama-3.1-8B-Instruct",
#     "meta-llama/Llama-3.3-70B-Instruct",
    
#     "google/gemma-2-2b-it",
#     "google/gemma-2-9b-it",
#     "google/gemma-3-4b-it",
#     "google/gemma-3-12b-it",
#     "google/gemma-3-27b-it",
# ]

# HF_SUPPORTED_MODELS = \
#     HF_API_TOKEN_MODELS + \
#         ["Qwen/Qwen2.5-7B-Instruct-1M",
#          "Qwen/Qwen2.5-14B-Instruct",
#          "Qwen/Qwen2.5-32B-Instruct",

#          "microsoft/Phi-4-mini-instruct"
#          ]
