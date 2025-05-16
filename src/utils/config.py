import os


HF_API_TOKEN_MODELS = [
    "mistralai/Mistral-7B-Instruct-v0.3",
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "mistralai/Ministral-8B-Instruct-2410",
    "mistralai/Mistral-Nemo-Instruct-2407",

    "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
    "meta-llama/Llama-3.3-70B-Instruct",
    
    "google/gemma-2-2b-it",
    "google/gemma-2-9b-it",
    "google/gemma-3-4b-it",
    "google/gemma-3-12b-it",
    "google/gemma-3-27b-it",
]

HF_SUPPORTED_MODELS = \
    HF_API_TOKEN_MODELS + \
        ["Qwen/Qwen2.5-7B-Instruct-1M",
         "Qwen/Qwen2.5-14B-Instruct",
         "Qwen/Qwen2.5-32B-Instruct",

         "microsoft/Phi-4-mini-instruct"
         ]
                   

DATASETS_SUPPORTED = [
    'DollyDataset', 
    'Alpaca',
    'ShareGPT', 
    'Mbpp',
    'Apps',
    'DS1000',
    'BigCodeBench',
    'PythonCode' 
]

MAX_INPUT_PROMPT_LEN = 2048

def check_model_name(model_name: str) -> bool:
    """
    Check if the model name is supported.
    
    Parameters:
    - model_name (str): Name of the model.
    
    Returns:
    - bool: True if the model is supported, False otherwise.
    """
    import logging
    logger = logging.getLogger(__name__)
    if model_name not in HF_SUPPORTED_MODELS:
        logger.error(f"Model {model_name} is not supported.")
        raise ValueError(f"Model {model_name} is not supported.")
    else:
        return True
    

def check_token_and_model(model_name: str, access_token: str):
    """
    Check if the model name is supported and the Hugging Face API token is set.
    
    Parameters:
    - model_name (str): Name of the model.
    """
    import logging
    logger = logging.getLogger(__name__)

    if access_token is None and model_name in HF_API_TOKEN_MODELS:
        logger.error("Huggingface API token not found as environment variable 'HF_API_TOKEN'.")
        raise ValueError("Huggingface API token not found. Please add it as an environment variable 'HF_API_TOKEN'")
    else:
        return True
    

def check_save_folder(save_folder: str) -> bool:
    """
    Check if the save folder exists.
    
    Parameters:
    - save_folder (str): Path to the save folder.
    
    Returns:
    - bool: True if the folder exists, False otherwise.
    """
    import logging
    logger = logging.getLogger(__name__)
    if not os.path.exists(save_folder):
        logger.error(f"Save folder does not exist: {save_folder}")
        raise FileNotFoundError(f"Save folder does not exist: {save_folder}")
    else:
        return True


def get_cache_dir(verbose: bool = True):
    """
    Get the cache directory for the transformers library.
    
    Returns:
    - str: Path to the cache directory.
    """
    import logging
    logger = logging.getLogger(__name__)
    cache_dir = os.getenv("HF_HOME", os.path.expanduser("~/.cache/huggingface/hub/"))
    if verbose:
        logger.info(f"HF_HOME huggingface cache directory: {cache_dir}")
    if not os.path.exists(cache_dir):
        logger.error(f"Cache directory does not exist: {cache_dir}")
        raise FileNotFoundError(f"Cache directory does not exist: {cache_dir}")
    return cache_dir


def get_project_root(target_name="llm-testbed"):
    """
    Get the project root directory.
    
    Parameters:
    - target_name (str): Name of the target project directory.
    
    Returns:
    - str: Path to the project root directory.
    """
    ctr = 10
    current_dir = os.path.dirname(os.path.abspath(__file__))
    while os.path.basename(current_dir) != target_name and ctr > 0:
        current_dir = os.path.dirname(current_dir)
        ctr -= 1
    if ctr <= 0:
        raise FileNotFoundError(f"Could not find project root directory with name: {target_name}")
    return current_dir


def get_raw_dataset_path(dataset_name):
    """
    Returns the full path to the raw dataset folder inside the project root.
    
    Args:
        dataset_name (str): Name of the dataset folder inside `data/raw/`.
    
    Returns:
        str: Full path to the dataset.
    """
    project_root = get_project_root()
    raw_data_path = os.path.join(project_root, "data", "raw_instruct", dataset_name)
    
    assert os.path.exists(raw_data_path), f"Path does not exist: {raw_data_path}"
    
    return raw_data_path


def get_model_string(model_name:str):
    """
    Get the model string for the specified model name.
    
    Parameters:
    - model_name (str): Name of the model.
    
    Returns:
    - str: Model string.
    """
    return model_name.replace("/", "--")