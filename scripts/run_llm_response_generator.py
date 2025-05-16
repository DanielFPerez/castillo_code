#!/usr/bin/env python3
"""
Takes the raw data, a model and generates the response length dataset.
"""
import os
import logging
import argparse
import torch
import json
import signal
import sys
import numpy as np
from typing import List, Dict

sys.path.append("../")
from src.utils.logger import setup_logger
import src.utils.config as utils_config 
import src.data.rawdata_processor as rawdataproc
from src.models.ModelHandler import ModelHandler

from transformers import logging as transf_logging
transf_logging.disable_progress_bar()


SAVE_PATH_EXCEPTION = str()
PROGRESS_DATA = list()


# Function to save progress
def save_progress():
    global SAVE_PATH_EXCEPTION
    global PROGRESS_DATA
    logger = logging.getLogger(__name__)
    with open(SAVE_PATH_EXCEPTION, "w") as f:
        json.dump(PROGRESS_DATA, f)
    logger.info(f"Saved preprocessed data to {SAVE_PATH_EXCEPTION}")


# Signal handler function
def handle_exit(signum, frame):
    global SAVE_PATH_EXCEPTION
    logger = logging.getLogger(__name__)
    logger.info(f"Received signal {signum}")
    if SAVE_PATH_EXCEPTION is not None: 
        save_progress()
    logger.info(f"Exiting...")
    sys.exit(0)


def update_save_path(save_path: str):
    global SAVE_PATH_EXCEPTION
    SAVE_PATH_EXCEPTION = save_path


def clear_progress_data():
    global PROGRESS_DATA
    PROGRESS_DATA.clear()


def update_progress_data(data: List[Dict]):
    global PROGRESS_DATA
    PROGRESS_DATA = data



def load_preprocessed_data(src_path: str, verbose: bool = False):
    """
    Loads the preprocessed data from the specified source path.
    """
    logger = logging.getLogger(__name__)
    if not os.path.exists(src_path):
        logger.error(f"Preprocessed data file does not exist: {src_path}")
        raise FileNotFoundError(f"Preprocessed data file does not exist: {src_path}")
    else:
        with open(src_path, 'r') as file:
            data = json.load(file)
        logger.info(f"Loaded preprocessed data from {src_path}")
        if verbose: print(f"Loaded preprocessed data from {src_path}")

        existing_samples = set([elem['sample_id'] for elem in data])

        if len(data) != len(existing_samples):
            logger.error(f"Samples in the saved dataset are not unique! Check them")
            raise 
    return data, existing_samples


def main(datasets: List[str], in_dataset_types: List[str], in_model: str, in_device: str, save_folder_src: str, include_hiddenstates: bool = False, 
         include_logits: bool = False, include_attnheads: bool = False,
         in_top_k: str = str(), in_top_p: str = str(), in_temp: str = str(), 
         batch_size: int = 10, verbose: bool = False, retries: int = 6):
    """
    Main execution function that initializes logging and runs the pipeline.
    """
    global PROGRESS_DATA

    # Register signal handlers
    signal.signal(signal.SIGINT, handle_exit)
    signal.signal(signal.SIGTERM, handle_exit)

    logger = logging.getLogger(__name__)
    logger.info("Application started with custom logging settings.")

    if torch.cuda.is_available():
        num_devices = torch.cuda.device_count()
        logger.info(f"Number of CUDA devices: {num_devices}")
        for i in range(num_devices):
            logger.info(f"Device {i}: {torch.cuda.get_device_name(i)}")
    else:
        logger.warning("CUDA is not available.")

    logger.info(f"Using dataset(s): {datasets}")
    logger.info(f"Using dataset type(s): {in_dataset_types}")
    logger.info(f"Using model: {in_model}")
    logger.info(f"Using device: {in_device}")

    # Check if the save folder exists
    if save_folder_src: 
        utils_config.check_save_folder(save_folder_src)
        logger.info(f"Using save folder: {save_folder_src}")
    else:
        save_folder_src = os.path.join(utils_config.get_project_root(), f"data/preprocessed")
        logger.info(f"Using default save folder: {save_folder_src}")

    model_name: str = in_model
    device_str: str = in_device
    padding_side: str = 'left'

    modelhandler = ModelHandler(model_name=model_name, device_str=device_str, padding_side=padding_side, verbose=verbose)

    modelhandler.load_model_and_tokenizer()

    modelhandler.load_generation_config(out_hidden_states=include_hiddenstates, out_logits=include_logits, out_attentions=include_attnheads,
                                        in_top_k=in_top_k, in_top_p=in_top_p, in_temp=in_temp)

    # Model string to save results
    model_str = in_model.replace("/", "--")

    for dataset_name in datasets:
        
        for dataset_type in in_dataset_types:
            logger.info(f"\n")
            logger.info(f"# ######## Processing dataset: {dataset_name} - {dataset_type}")
            
            save_folder = os.path.join(save_folder_src, dataset_type) 
            os.makedirs(save_folder, exist_ok=True)

            save_path = os.path.join(save_folder, f"preproc_{dataset_name}_{dataset_type}_{model_str}.json")
            update_save_path(save_path)

            if os.path.exists(save_path):
                logger.info(f"Preprocessed file already exists. Loading information from: {save_path}")
                tmp_data, samples_id_set = load_preprocessed_data(save_path)
                update_progress_data(tmp_data)

                if not len(PROGRESS_DATA): 
                    raise ValueError("Preprocessed data is empty. Check the file.")
                tmp_data = None
            else: 
                clear_progress_data()
                if len(PROGRESS_DATA):
                    raise ValueError("Preprocessed data is not empty. Check the file.")
                samples_id_set = set()

            # Load dataset
            dataset = rawdataproc.load_rawdata_file(dataset_name, dataset_type, verbose=verbose)
            # dataset = dataset[0:5]

            logger.info(f"# ### LOADED dataset. The dataset has {len(dataset)} samples.")

            # Load sample processor for the dataset
            prompt_func = rawdataproc.RAWDATA_SAMPLE_PROCESSORS[dataset_name]

            for i, sample in enumerate(dataset):
        
                # logger.info(f"# ####### Processing sample {i}")
                
                prompt_text = prompt_func(sample, modelhandler.tokenizer, verbose=verbose)

                sample_id = rawdataproc.generate_sample_id(dataset_name, model_str, prompt_text, 
                                                           in_top_k=modelhandler.gen_config.top_k,
                                                           in_top_p=modelhandler.gen_config.top_p,
                                                           in_temp=modelhandler.gen_config.temperature,
                                                           verbose=verbose)
                prompt_id = rawdataproc.generate_prompt_id(dataset_name, prompt_text)

                if sample_id in samples_id_set:
                    logger.info(f"Sample-ID for sample {i} already exists in the dataset. Skipping: {sample_id}")
                    continue
                else: 
                    samples_id_set.add(sample_id)
                
                logger.debug(f"- Sample ID: {sample_id}")
                logger.debug(f"- Prompt ID: {prompt_id}")

                input_size, output, gen_time = modelhandler.generate_with_retry(prompt_text=prompt_text)

                if output == None: 
                    logger.warning(f"# ## FAILED to generate data for sample {i}")
                    continue
                    
                # Get output data and its metrics
                output_lengths = rawdataproc.get_len_out_tokens_batch(input_size, output.sequences, modelhandler.tokenizer)

                if max(output_lengths) > modelhandler.max_new_tokens_cached:
                    modelhandler.max_new_tokens_cached = max(output_lengths)
                
                longest_response = modelhandler.tokenizer.decode(output['sequences'][np.argmax(output_lengths)][input_size:], 
                                                                 skip_special_tokens=True)
                shortest_response = modelhandler.tokenizer.decode(output['sequences'][np.argmin(output_lengths)][input_size:], 
                                                                 skip_special_tokens=True)
                
                out_lengths_mean = round(np.mean(output_lengths).item(),2)
                out_lengths_std = round(np.std(output_lengths).item(),2)
                out_percentiles = {elem: round(np.percentile(output_lengths, int(elem[1:])).item(), 3) 
                                   for elem in ['p25', 'p50', 'p75', 'p99'] }
                
                category = sample['category'] if dataset_name == "DollyDataset" else "undefined" 
                
                logger.debug(f"- Input size (tokens): {input_size}")
                outlim = len(longest_response) if len(longest_response) < 80 else 80 
                logger.debug(f"- Longest response: {longest_response[0:outlim].replace("\n", " ")}...")
                outlim = len(shortest_response) if len(shortest_response) < 80 else 80 
                logger.debug(f"- Shortest respose: {shortest_response[0:outlim].replace("\n", " ")}")
                logger.debug(f"- Generation time: {gen_time}")
                logger.debug(f"- MAX output size (tokens): {max(output_lengths)}")
                logger.debug(f"- Output lengths mean: {out_lengths_mean}, std: {out_lengths_std}")
                logger.debug(f"- Output size  batch (tokens): {output_lengths}")
                logger.debug(f"- Output lengths percentiles: {out_percentiles}")
                logger.debug(f"- Top-k: {modelhandler.gen_config.top_k}, top-p: {modelhandler.gen_config.top_p}, temperature: {modelhandler.gen_config.temperature}")

                # Generated sample to cache
                tmp_sample = {'sample_id': sample_id, 'prompt_id': prompt_id, 'model': model_str,
                              'dataset': dataset_name, 'prompt_text': prompt_text, 
                              'longest_response': longest_response, 'shortest_response': shortest_response,
                              'input_size': input_size, 'output_sizes' : output_lengths, 'output_mean' : out_lengths_mean,
                              'output_std' : out_lengths_std, 'output_percentiles': out_percentiles,
                              'top_k': modelhandler.gen_config.top_k, 'top_p': modelhandler.gen_config.top_p,
                              'temp': modelhandler.gen_config.temperature,
                              'category': category,'gen_time': gen_time}
                
                if include_hiddenstates:
                    tmp_sample['hidden_states'] = output.hidden_states
                if include_logits:
                    tmp_sample['logits'] = output.logits
                
                PROGRESS_DATA.append(tmp_sample)
                
                output = None
                
                # logger.info(f"# ## Processed sample {i} - {sample_id}")
                if i % 20 == 0:
                    logger.info(f"Processed {i} samples. Saving data...")
                    save_progress()
            
            logger.info(f"########## DONE processing {dataset_name}-{dataset_type}. TOTAL: {len(dataset)} samples. Saving data.")
            save_progress()
    
    logger.info(f"")
    logger.info(f"Finished processing all datasets. Exiting...")    


def get_parser():
    """Defines and returns the argument parser for the script."""
    parser = argparse.ArgumentParser(description="Main application script for data processing and model training.")

    parser.add_argument("--log_format", type=str, default="", help="Format for log messages.")
    # Select logging level
    parser.add_argument("--log_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Logging level for the logger.")

    parser.add_argument("--in_device", default="auto", type=str, choices=["auto", "mps", "cuda:0", "cuda:1", "cuda:2", "cuda:3", "cpu"],
                        help="GPU/CUDA device to run the model. 'auto' for multiple GPUs.")
    parser.add_argument("--datasets", default=["DollyDataset"], type=str, choices=utils_config.DATASETS_SUPPORTED, nargs='+',
                        help="Raw dataset to use for generating the prompts.")
    parser.add_argument("--dataset_types", default=["train", "validate", "test"], type=str, choices=['train', 'validate', 'test'], nargs='+',
                        help="Data subset(s) to use.")
    parser.add_argument("--in_model", type=str, default="meta-llama/Llama-3.2-3B-Instruct", choices=utils_config.HF_SUPPORTED_MODELS, 
                        help="Huggingface model name to use for generation.")
    parser.add_argument("--save_folder", type=str, default=str(), help="Folder to save the processed data. A 'train' and 'validation' folder will be created.")
    parser.add_argument("--batch_size", type=int, default=10, help="Batch size for generating the responses.")
    parser.add_argument("--include_hiddenstates", action="store_true", help="Include hidden states in the model output.")
    parser.add_argument("--include_logits", action="store_true", help="Include logits in the model output.")
    parser.add_argument("--in_top_k", type=str, default=str(), help="Top-k parameter for controlling the model's generation_config.")
    parser.add_argument("--in_top_p", type=str, default=str(), help="Top-p parameter for controlling the model's generation_config.")
    parser.add_argument("--in_temp", type=str, default=str(), help="Temperature parameter for controlling the model's generation_config.")
    parser.add_argument("--log_suffix", type=str, default=str(), help="Suffix to append to the log file.")
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    log_path = os.path.join(utils_config.get_project_root(), f"outputs/logs/log_{args.in_model.replace("/", "--")}{args.log_suffix}.log")

    # log level from arguments
    log_level = getattr(logging, args.log_level, logging.INFO)

    # Setup logging with arguments
    setup_logger(log_file_path=log_path, log_format=args.log_format,
                 console_level=log_level, 
                 file_level=log_level)

    # Remove DEBUG logging from huggingface urllib
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    if "gemma-3" in args.in_model and args.in_device != "mps":
        logger = logging.getLogger(__name__)
        logger.warning("GEMMA-3 model family detected. Disabling FlashAttention, Enabling math sdp.")
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)

    # Run the main function
    main(datasets=args.datasets, in_dataset_types=args.dataset_types, 
         in_model=args.in_model, in_device=args.in_device, save_folder_src=args.save_folder, 
         include_hiddenstates=args.include_hiddenstates, 
         include_logits=args.include_logits, include_attnheads=False,
         in_top_k=args.in_top_k, in_top_p=args.in_top_p, in_temp=args.in_temp,
         batch_size=args.batch_size, verbose=False, 
         retries=6)