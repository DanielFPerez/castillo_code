# 🏰 CASTILLO: Characterizing Response Length Distributions in Large Language Models

The **CASTILLO** dataset is designed to support research on the variability of response lengths in large language models (LLMs). It provides statistical summaries of output lengths across 13 open-source LLMs evaluated on 7 instruction-following datasets. For each unique ⟨prompt, model⟩ pair, 10 independent responses were generated using fixed decoding parameters, and key statistics were recorded—such as mean, standard deviation, percentiles, as well as the longest and shortest completions. CASTILLO enables research in proactive resource allocation, inference optimization, and the analysis of generation behavior variability across models and prompts.

## Access Dataset

You can access the dataset in the huggingface hub, under https://huggingface.co/datasets/danfperam/castillo 
Instructions for how to use the dataset, as well as the schema is found in huggingface.

## Setup Environment

Follow these steps to create a python environment (currently works for linux-based systems):

### Python environment

1. Create a python environment
```bash
python -m venv venv
```

2. Install requirements: 
```bash
pip install -r requirements.txt
```

3. Activate environment.
```bash
source venv/bin/activate
```

### HF API Token

For generating more data (not for using the dataset), some huggingface models require an API token access. The script to generate data automatically reads this token from an environment variable. 
Make sure to define an environment variable for the API token, for example: 
```bash
export HF_API_TOKEN="hf_adfqFWERqERTWERtewRTWREgFWEgEwgWE"
``` 
And another for your Huggingface local cache:
```bash
export HF_HOME="/path/to/hf_hub"
```

## Generate more data

You can generate more data by running the module scripts.run_llm_response_generator for the repo home directory. 
For example, if you want to generate data using the Mixtral-8x7B model on three datasets (Dolly, ShareGPT, and Alpaca), run:  
```bash
python -m scripts.run_llm_response_generator --in_model="mistralai/Mixtral-8x7B-Instruct-v0.1" --in_device "auto" --datasets DollyDataset ShareGPT Alpaca --log_level INFO
```

And for more help and to see all parameters, run: 
```bash
python -m scripts.run_llm_response_generator --help
```

This will read the data from the respective "./data/raw_instruct" folder, and will save one ".json" file for each <model, dataset> pair within the folder "./data/preprocessed". 

Note that log files are stored under "./outputs/logs". 

## Merging the dataset 

To merge all datasets from the "./data/preprocessed" folder, you can run: 
```bash
python -m scripts.preprocessed_to_huggingface --raw_src_dir data/preprocessed --dst_dir <MY_DIR> --config_name <MY_CONFIG> --log_filename mylog.log
```

This will create a directory called "<MY_CONFIG>" under the folder "<MY_DIR>" with three .json files: a train, a test, and a validation file, which are the merge versions of the individual ones in "./data/preprocessed" 
The "mylog.logs" file will be saved under "./outputs/logs" directory.

## Sanitize Dataset

After merging the dataset, you can run the sanitation step to filter out and post-process the text degeneration instances. 
You can run the command: 
```bash
python -m scripts.sanitize_dataset --src_dir <MY_MERGED_DATA_DIR> --dst_dir <MY_DST_DIR> --log_filename "mylog.log"
```
This will create a "sanitized" and a "degenerate" subfolders within <MY_DST_DIR> with the respective dataset configurations.

The "mylog.logs" file will be saved under "./outputs/logs" directory.

## 📚 Citation 

```bibtex
@misc{perezramirez2025castillo,
      title={CASTILLO: Characterizing Response Length Distributions of Large Language Models}, 
      author={Daniel F. Perez-Ramirez and Dejan Kostic and Magnus Boman},
      year={2025},
      eprint={2505.16881},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2505.16881}, 
}
```
## 🪪 License

This code is distributed under the Apache 2.0 license.