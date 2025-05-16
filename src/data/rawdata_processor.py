import os
import logging
import json
import torch
from typing import List, Dict
from enum import Enum
from hashlib import sha256

from transformers import PreTrainedTokenizer
from transformers.generation.utils import GenerateDecoderOnlyOutput

import src.utils.config as utils_config 

logger = logging.getLogger(__name__) 


DATASET_TYPES = Enum("DatasetTypes", "train validate test")


MAX_PROMTP_LEN = 2500   # Maximum length of the prompt text in number of tokens (default: 2500, or ~1830 words, or ~5 pages)

# ####
# # Functions for reading datasets from disk
# ####  

def check_dataset_name(dataset_name: str) -> bool:
    # Check if the dataset exists
    if dataset_name not in utils_config.DATASETS_SUPPORTED:
        logger.error(f"Dataset {dataset_name} is not supported.")
        raise ValueError(f"Dataset {dataset_name} is not supported. Supported Types: {utils_config.DATASETS_SUPPORTED}")
    else:
        return True


def get_rawdata_src_path(dataset_name: str) -> str:
    """
    Returns the path to the raw data source for the specified dataset.
    """
    # Check if the dataset exists
    check_dataset_name(dataset_name)

    tmp_path = os.path.join(utils_config.get_project_root(), "data", "raw_instruct", dataset_name)
    if not os.path.exists(tmp_path):
        logger.error(f"Dataset {dataset_name} does not exist in {tmp_path}")
        raise FileNotFoundError(f"Dataset {dataset_name} does not exist in {tmp_path}")
    logger.debug(f"Reading raw data from {tmp_path}")
    return tmp_path


def load_rawdata_file(dataset_name: str, data_type: str, verbose: bool = False) -> List[str]:
    """
    Loads the raw data file based on the specified data type.
    """
    data_src_path = get_rawdata_src_path(dataset_name)
    if verbose: print("Data Source Path: ", data_src_path)

    if not DATASET_TYPES[data_type]:
        logger.error(f"Dataset type {data_type} is not supported.")
        raise ValueError(f"Dataset {data_type} is not supported.")
    else:
        data_path = os.path.join(data_src_path, f"raw_{dataset_name}_{data_type}.json")
        if verbose: print("Data Path: ", data_path)
        if not os.path.exists(data_path):
            logger.error(f"Data file for {data_type} does not exist at {data_path}")
            raise FileNotFoundError(f"Data file for {data_type} does not exist at {data_path}")
        else:
            with open(data_path, 'r') as file:
                data = json.load(file)  
            logger.debug(f"Loaded raw data file for {data_type} from {data_path}")
            return data

# #################################################################################
# # Raw Data Sample Processors
# #################################################################################


def build_rawprompt_text_chat(question: str) -> List[Dict]:
    """
    Builds a basic chat template with a question.
    """
    return [{"role": "user", "content": question}]


def build_rawprompt_text_batch(prompt: str, batch_size: int) -> List[List[Dict]]:
    """
    Builds a batch of raw prompt text from a single prompt.
    """
    return [build_rawprompt_text_chat(prompt)]*batch_size


def check_rawprompt_text_length(rawprompt: str, tokenizer: PreTrainedTokenizer) -> str:
    """
    Checks the length of the raw prompt text and truncates it if it exceeds the maximum length.
    """
    logger = logging.getLogger(__name__)

    rawprompt_tokenized = tokenizer(rawprompt)['input_ids']
    if len(rawprompt_tokenized) > MAX_PROMTP_LEN:
        logger.warning(f"Prompt text length exceeds maximum limit of {MAX_PROMTP_LEN} tokens. Truncarting prompt.")
        return tokenizer.decode(rawprompt_tokenized[0:MAX_PROMTP_LEN], skip_special_tokens=True)
    else:
        return rawprompt


def print_rawprompt_text(rawprompt: str, verbose: bool = False):
    """
    Prints the raw prompt text.
    """
    logger = logging.getLogger(__name__)
    # Print prompt
    tmp_threshold = 100
    lim = len(rawprompt) if len(rawprompt) < tmp_threshold else tmp_threshold
    logger.debug(f"Raw Prompt: {rawprompt[0:lim].replace("\n", " ")}...")
    if verbose: 
        print(f"Raw Prompt: {rawprompt[0:lim].replace("\n", " ")}...")


# #### DOLLY PREPROC FUNC
def dolly_generate_rawprompt_text(datasample: Dict, tokenizer: PreTrainedTokenizer, truncate: bool = True, verbose: bool = False) -> str:
    """
    Generates the raw prompt text for the Dolly dataset.   
    """
    logger = logging.getLogger(__name__)

    if datasample['context']:
        rawprompt = datasample['instruction']+"\nContext: "+datasample['context']
    else:
        rawprompt = datasample['instruction']

    # Check Length of the tokenized prompt (some datasets have prompts longer than model's context length)
    if truncate:
        rawprompt = check_rawprompt_text_length(rawprompt, tokenizer)

    print_rawprompt_text(rawprompt, verbose)    

    return rawprompt


# #### ALPACA PREPROC FUNC
def alpaca_generate_rawprompt_text(datasample: Dict, tokenizer: PreTrainedTokenizer, truncate: bool = True, verbose: bool = False) -> str:
    """
    Generates the raw prompt text for the Alpaca dataset.   
    """
    logger = logging.getLogger(__name__)
    if datasample['input']:
        rawprompt = datasample['instruction']+"\nInput: "+ datasample['input']
    else:
        rawprompt = datasample['instruction']

    if truncate:
        rawprompt = check_rawprompt_text_length(rawprompt, tokenizer)

    print_rawprompt_text(rawprompt, verbose)    

    return rawprompt


# #### SHAREGPT PREPROC FUNC
def sharegpt_generate_simple_rawprompt_text(datasample: Dict, tokenizer: PreTrainedTokenizer, truncate: bool = True, verbose: bool = False):
    logger = logging.getLogger(__name__)
    if datasample['conversations'][0]['from'] == "human":
        rawprompt = datasample['conversations'][0]['value']
    else:
        logger.error(f"First conversation in ShareGPT datasample {datasample['id']} is not from human.")
        raise ValueError(f"First conversation in ShareGPT datasample {datasample['id']} is not from human.")
    if truncate:
        rawprompt = check_rawprompt_text_length(rawprompt, tokenizer)
    print_rawprompt_text(rawprompt, verbose)
    return rawprompt


# #### PYTHONCODE PREPROC FUNC
def pythoncode_generate_rawprompt_text(datasample: Dict, tokenizer: PreTrainedTokenizer, truncate: bool = True, verbose: bool = False):
    raise NotImplementedError("PythonCode raw data sample processor not implemented!")

# ### MBPP PREPROC FUNC
def mbpp_generate_rawprompt_text(datasample: Dict, tokenizer: PreTrainedTokenizer, truncate: bool = True, verbose: bool = False, prompt_key="text"):
    logger = logging.getLogger(__name__)
    if datasample[prompt_key] and datasample['test_list']:
        rawprompt = "You are an expert Python programmer, and here is your task: "
        rawprompt += datasample[prompt_key]
        rawprompt += " Your code should pass these tests:\n\n"
        rawprompt += "\n".join(datasample['test_list']) + "\n"
    else:
        logger.error(f"MBPP data sample does not have any prompt and/or test_list fields: {datasample}")
        raise ValueError(f"MBPP data sample does not have any prompt and/or test_list fields")
    
    if truncate:
        rawprompt = check_rawprompt_text_length(rawprompt, tokenizer)

    print_rawprompt_text(rawprompt, verbose)

    return rawprompt


# ### APPS PREPROC FUNC
def apps_generate_rawprompt_text(datasample: Dict, tokenizer: PreTrainedTokenizer, truncate: bool = True, verbose: bool = False, prompt_key="question"):
    logger = logging.getLogger(__name__)
    if datasample[prompt_key]:
        rawprompt = "QUESTION: "
        rawprompt += datasample[prompt_key] + "\n\n"
    else:
        logger.error(f"MBPP data sample does not have any prompt and/or test_list fields: {datasample}")
        raise ValueError(f"MBPP data sample does not have any prompt and/or test_list fields")

    if datasample['starter_code']:
        rawprompt += "Your code should start with the following: \n"
        rawprompt += datasample['starter_code'] + "\n"

    if datasample['input_output']:
        mydict = json.loads(datasample['input_output'])
        rawprompt += "The following is the standard input and output format: \n"
        rawprompt += f"Inputs:\n"
        rawprompt += "\n".join([str(elem) for elem in mydict['inputs']])
        rawprompt += f"\nOutputs:\n"
        rawprompt += "\n".join([str(elem) for elem in mydict['outputs']])
    
    if truncate:
        rawprompt = check_rawprompt_text_length(rawprompt, tokenizer)

    print_rawprompt_text(rawprompt, verbose)

    return rawprompt


# #### DS-1000 PREPROC FUNC
def ds1000_generate_rawprompt_text(datasample: Dict, tokenizer: PreTrainedTokenizer, truncate: bool = True, verbose: bool = False) -> str:
    """
    Generates the raw prompt text for the DS-1000 dataset.   
    """
    logger = logging.getLogger(__name__)
    if datasample['prompt']:
        rawprompt = datasample['prompt']
    else:
        logger.error(f"DS-1000 data sample does not have any prompt and/or test_list fields: {datasample}")
        raise ValueError(f"DS-1000 data sample does not have any prompt and/or test_list fields")

    if truncate:
        rawprompt = check_rawprompt_text_length(rawprompt, tokenizer)

    print_rawprompt_text(rawprompt, verbose)    

    return rawprompt


# #### BigCodeBench PREPROC FUNC
def bigcodebench_generate_rawprompt_text(datasample: Dict, tokenizer: PreTrainedTokenizer, truncate: bool = True, verbose: bool = False) -> str:
    """
    Generates the raw prompt text for the BigCodeBench dataset.   
    """
    logger = logging.getLogger(__name__)
    if datasample['instruct_prompt']:
        rawprompt = datasample['instruct_prompt']
    else:
        logger.error(f"BigCodeBench data sample does not have any prompt and/or test_list fields: {datasample}")
        raise ValueError(f"BigCodeBench data sample does not have any prompt and/or test_list fields")

    if truncate:
        rawprompt = check_rawprompt_text_length(rawprompt, tokenizer)

    print_rawprompt_text(rawprompt, verbose)    

    return rawprompt


# ########### Dict with function callbacks
RAWDATA_SAMPLE_PROCESSORS = {
    'DollyDataset': dolly_generate_rawprompt_text, 
    'Alpaca': alpaca_generate_rawprompt_text, 
    'ShareGPT': sharegpt_generate_simple_rawprompt_text, 
    'PythonCode': sharegpt_generate_simple_rawprompt_text,
    'Mbpp': mbpp_generate_rawprompt_text,
    'Apps': apps_generate_rawprompt_text,
    'DS1000': ds1000_generate_rawprompt_text,
    'BigCodeBench': bigcodebench_generate_rawprompt_text,
}


def generate_sample_id(dataset_name: str, model_str : str, rawprompt: str,
                       in_top_p: float, in_top_k: int, in_temp: float, 
                       resolution: int = 16, verbose : bool = False) -> str:
    """
    Generates a hash value for a string to store the sample in the dataset.
    The string is as long as the 'resolution' argument. 
    NOTE: THIS SAMPLE IS UNIQUE, as it depends on the model and the prompt text, as well as generation paramters.
    """
    if "/" in model_str: 
        model_str = model_str.replace("/","--")
    
    unique_str = f"{dataset_name}_{model_str}_topk_{in_top_k}_topp_{in_top_p}_temp_{in_temp}_{rawprompt}"
    if verbose: print(f"str for sample ID hash: {unique_str[0:60]}[...]")
    logger.debug(f"str for sample ID hash: {unique_str[0:80]}[...]")
    hash_key = sha256(unique_str.encode()).hexdigest()[:resolution]
    return str(hash_key)


def generate_prompt_id(dataset_name: str, rawprompt: str, resolution: int = 16) -> str:
    """
    Generates a hash value for a string that contains the dataset namne and the prompt text.
    Helpful later to compare results from different models on the same prompt.
    The string is as long as the 'resolution' argument. 
    NOTE: there CAN be multiple id's of this type in the dataset.
    """
    # ID SHOULD DEPEND ON DATASET NANE AND STRING, NOT ON MODEL! 
    # We later would like to compare the same sample across different models 
    unique_str = f"{dataset_name}_{rawprompt}"
    hash_key = sha256(unique_str.encode()).hexdigest()[:resolution]
    return str(hash_key)



# ###############################################################################
# # Functions for processing output of the model
# ###############################################################################

def print_model_output_batch(input_batch: torch.Tensor, output_batch: torch.Tensor, tokenizer: PreTrainedTokenizer,
                             n_elems: int = -1, ):
    """
    Prints the output of the model for a batch of inputs.
    """
    if len(input_batch.shape) != 2:
        logger.error(f"Input batch shape is not 2D [batch_size, input_prompt_len]. Shape: {input_batch.shape}")
        raise ValueError(f"Input batch must be 2D: [batch_size, input_prompt_len].")
    
    # Get tensor of output lengths
    output_lengths = get_len_out_tokens_batch(input_batch.shape[1], output_batch, tokenizer) # returns a list
    # get corrensponding strings
    output_strs = tokenizer.batch_decode([output_batch[i, input_batch.shape[1]:output_lengths[i]+input_batch.shape[1]] 
                                          for i in range(len(output_lengths))], 
                                          skip_special_tokens=False)
    for i, elem in enumerate(output_strs[0:n_elems]):
        print(f"ELEMENT {i}:\n{elem}")
        print(f"NUM. OUT TOKENS: {output_lengths[i]}")
        print("============================================================\n")
    return True


def get_len_out_tokens_batch(input_size: int, output_seq: torch.Tensor, 
                             tokenizer: PreTrainedTokenizer, verbose: bool = False) -> List[int]:
    """
    Returns the length of the output sequence in tokens for each sample in the batch
    Arguments: 
    - output_seq (torch.Tensor): output of an inference pass. size: [batch_size, len_of_longest_response].
    """
    tmp_output = output_seq[:,input_size:]

    if not tokenizer.pad_token_id:
        logger.error("Tokenizer does not have a pad token id")  
        raise ValueError("Tokenizer does not have a pad token id")

    # Create a tensor with same size as output filled with the pad_token_id
    eos_mask = torch.ones(tmp_output.shape, dtype=tmp_output.dtype).to(tmp_output.device)*tokenizer.pad_token_id
    
    if verbose: 
        logger.debug(f"EOS MASK SHAPE {eos_mask.shape}\n{eos_mask}")
        print(f"EOS MASK SHAPE {eos_mask.shape}\n{eos_mask}")

    ret_lens = torch.sum(tmp_output != eos_mask, dim=1) + 1

    # Return sum of all values of output along batch axis
    return ret_lens.tolist()