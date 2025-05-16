import logging
import os
import torch
import time
import gc

from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, PreTrainedModel, PreTrainedTokenizer, DynamicCache, LlamaForCausalLM

import src.utils.config as utils_config 
import src.data.rawdata_processor as rawdataproc


from transformers import logging as transf_logging
transf_logging.disable_progress_bar()


MAX_GENERATION_LENGTH = 15000


class ModelHandler:
    def __init__(self, model_name, device_str, padding_side: str = 'left', verbose: bool = False):
        
        self.logger = logging.getLogger(__name__)
        self.verbose = verbose
        
        self.access_tk = os.getenv("HF_API_TOKEN")
        if model_name == "meta-llama/Llama-3.3-70B-Instruct":
            self.access_tk = os.getenv("HF_API_TOKEN_LLAMA70B")
        
        utils_config.check_model_name(model_name)
        utils_config.check_token_and_model(model_name, self.access_tk)

        self.model_name: str = model_name
        self.device_str: str = device_str
        self.padding_side: str = padding_side

        self.cache_dir = utils_config.get_cache_dir()
        self.logger.info(f"Using cache directory: {self.cache_dir}")
        if self.verbose: print(f"Using cache directory: {self.cache_dir}")
        
        self.inputs = torch.zeros(1, 1)

        self.max_new_tokens = 99999999
        self.max_new_tokens_cached = 1000
        self.dynam_cache = None

        
    def load_model_and_tokenizer(self):
        self.logger.info(f"Loading model and tokenizer for model: {self.model_name}")

        if "gemma" in self.model_name:
            tmp_type = torch.bfloat16
        else: 
            tmp_type = torch.float16

        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, token=self.access_tk, device_map=self.device_str, 
                                                          torch_dtype=tmp_type, cache_dir=self.cache_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, token=self.access_tk, device_map=self.device_str, 
                                                       torch_dtype=tmp_type, padding_side=self.padding_side, cache_dir=self.cache_dir)
        
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Check that model name exists. If not, add it
        if not self.model.config.name_or_path:
            self.model.config.name_or_path = self.model_name

        self.model.eval()

        self.logger.info(f"Loaded model and tokenizer for model: {self.model_name}")
        if self.verbose: print(f"Loaded model and tokenizer for model: {self.model_name}") 


    def load_generation_config(self, out_hidden_states: bool = False, out_logits: bool = False, out_attentions: bool = False, 
                               in_top_k: str = str(), in_top_p: str = str(), in_temp: str = str()):
        """
        Loads the generation config for the model.
        """
        # Load the model config
        self.gen_config = GenerationConfig.from_pretrained(self.model_name, token=self.access_tk, cache_dir=self.cache_dir)

        self.logger.info(f"Loading generation config for model: {self.model_name}")
        
        # Define LLM maximum response length 
        if "text_config" in self.model.config.__dict__.keys():
            self.max_new_tokens = min(self.model.config.text_config.max_position_embeddings - rawdataproc.MAX_PROMTP_LEN, MAX_GENERATION_LENGTH)
        else:
            self.max_new_tokens = min(self.model.config.max_position_embeddings - rawdataproc.MAX_PROMTP_LEN, MAX_GENERATION_LENGTH)

        self.logger.info(f"Setting max_new_tokens to: {self.max_new_tokens}")   

        # Setup the generation config
        self.gen_config.pad_token_id = self.tokenizer.eos_token_id
        
        self.gen_config.output_hidden_states = out_hidden_states
        self.gen_config.output_logits = out_logits
        self.gen_config.output_attentions = out_attentions
        
        self.gen_config.use_cache = True

        if out_hidden_states or out_logits or out_attentions:
            log_txt = f"Setting GenerationConfig to output_hidden_states: {out_hidden_states}, output_logits: {out_logits}, output_attentions: {out_attentions}."
            if self.verbose: print(log_txt)
            self.logger.info(log_txt)
        self.gen_config.return_dict_in_generate = True
        
        if in_top_k:
            value = int(in_top_k)
            assert value > 0 and value < 100, f"Top-k must be between 0 and 100. Got: {value}"
            self.gen_config.top_k = int(in_top_k)
            self.model.config.top_k = int(in_top_k)
            self.logger.info(f"Setting top_k to: {self.gen_config.top_k}")
            if self.verbose: print(f"Setting top_k to: {self.gen_config.top_k}")
        
        if in_top_p:
            value = float(in_top_p)
            assert value > 0 and value < 1.00000001, f"Top-p must be between 0 and 1. Got: {value}"
            self.gen_config.top_p = float(in_top_p)
            self.model.config.top_p = float(in_top_p)
            self.logger.info(f"Setting top_p to: {self.gen_config.top_p}")
            if self.verbose: print(f"Setting top_p to: {self.gen_config.top_p}")
        
        if in_temp:
            value = float(in_temp)
            assert  0 <= value <= 2.0000001, f"Temperature must be between 0 and 2 (inclussive). Got: {value}" # according to openAI API: https://platform.openai.com/docs/api-reference/responses/create#responses-create-temperature 
            self.gen_config.temperature = float(in_temp)
            self.model.config.temperature = float(in_temp)
            self.logger.info(f"Setting temperature to: {self.gen_config.temperature}")
            if self.verbose: print(f"Setting temperature to: {self.gen_config.temperature}")

        log_str = f"Generation Strategy: do_sample: {self.gen_config.do_sample}, num_beams: {self.gen_config.num_beams}, top-k: {self.gen_config.top_k}, top-p: {self.gen_config.top_p},  min-p: {self.gen_config.min_p}, temperature: {self.gen_config.temperature}"
        
        self.logger.info(log_str)
        if self.verbose: print(log_str)
        
        if not self.gen_config.do_sample:
            self.logger.info(f"Setting Generation Config do_sample to True.")
            self.gen_config.do_sample = True

    @torch.no_grad()
    def generate_with_retry(self, prompt_text: str, in_batch_size: int = 10, retries: int = 6):
        """
        Generates responses for a list of identical prompts with retries on OOM errors.
        """
        current_batch_size = in_batch_size
        current_max_new_tokens = self.max_new_tokens
        self.gen_config.max_new_tokens = self.max_new_tokens

        for attempt in range(retries):
            try:
                self.dynam_cache = None

                prompt_batch = rawdataproc.build_rawprompt_text_batch(prompt_text, current_batch_size)
                
                self.inputs = self.tokenizer.apply_chat_template(prompt_batch, padding=True, return_tensors='pt', add_generation_prompt=True, dtype=self.model.dtype)

                attention_mask = torch.ones_like(self.inputs, dtype=self.model.dtype)

                inputs_dev = self.inputs.to(self.model.device)
                attention_mask_dev = attention_mask.to(self.model.device)
                
                self.logger.debug(f"Starting inference on attempt {attempt+1}/{retries}...")
                t_start = time.time()
                self.output = self.model.generate(inputs_dev, attention_mask=attention_mask_dev, generation_config=self.gen_config,
                                        # past_key_values=dynam_cache # TODO: GEMMA does NOT support past_key_values
                                        )
                gen_time = round(time.time()-t_start, 5)
                
                self.output['sequences'] = self.output.sequences.detach().cpu()

                if self.gen_config.output_hidden_states:
                    # ## For the first token generated (output.hidden_states[0]), AND for all layers of one element in the batch (elem[0,:,:])
                    self.output['hidden_states'] = [elem[0, :, :].detach().cpu().numpy() for elem in self.output.hidden_states[0]]
                if self.gen_config.output_logits:
                    # ## For the first token generated (output.logits[0]), AND for all layers of one element in the batch (elem[0,:,:])
                    self.output['logits'] = [elem.detach().cpu().numpy() for elem in self.output.logits]
                
                break

            except RuntimeError as e:
                if "out of memory" in str(e):
                    self.logger.warning(f"OOM Error caught on attempt {attempt+1}/{retries}.")
                    if attempt == 0: 
                        self.logger.warning(f"Error: {e}")
                        lim = len(prompt_text) if len(prompt_text) < 60 else 60
                        self.logger.warning(f"FAILED PROMPT: {prompt_text[0:lim]}")
                    
                    if attempt < retries and current_max_new_tokens > self.max_new_tokens_cached:
                        current_max_new_tokens = max(self.max_new_tokens_cached, current_max_new_tokens//2)
                        self.gen_config.max_new_tokens = current_max_new_tokens
                        self.logger.info(f"Attempt {attempt}: reducing max new tokens to {current_max_new_tokens}")
                    else:
                        self.logger.error(f"Failed after {retries} attempts due to repeated OOM errors. SKIPPING SAMPLE.")
                        self.output, gen_time = None, None
                        break   
                else: 
                    raise e
                
        return self.inputs.shape[1], self.output, gen_time


    @property
    def inputs(self):
        return self._inputs
    @inputs.setter
    def inputs(self, value): 
        self._inputs = value
    @inputs.deleter
    def inputs(self):
        del self._inputs
    
    @property
    def output(self):
        return self._output
    @output.setter
    def output(self, value):
        self._output = value
    @output.deleter
    def output(self):
        del self._output

    @property
    def max_new_tokens(self):
        return self._max_new_tokens
    @max_new_tokens.setter
    def max_new_tokens(self, value):
        if not isinstance(value, int) or value < 50:
            raise ValueError("Max new tokens must be of type int and bigger than 20.")
        self._max_new_tokens = value
        self.logger.debug(f"Maximum response length changed to: {value}")

    @property
    def max_new_tokens_cached(self):
        return self._max_new_tokens_cached
    @max_new_tokens_cached.setter
    def max_new_tokens_cached(self, value):
        if not isinstance(value, int) or value < 50:
            raise ValueError("Max new tokens cached must be of type int and bigger than 20.")
        self._max_new_tokens_cached = value
        self.logger.info(f"NEW Maximum cached length (modelhandler.max_new_tokens_cached): {value}")






    # def reload_model(self):
    #     # Explicitly delete and dereference
    #     del self.model
    #     del self.tokenizer
    #     del self._inputs
    #     del self._output
    #     del self._gen_config

    #     self.model = None
    #     self.tokenizer = None
    #     self._inputs = torch.zeros(1, 1).to("cpu")
    #     self._output = dict()
    #     self._gen_config = None

    #     # Garbage collect and clear CUDA memory
    #     gc.collect()
    #     torch.cuda.empty_cache()
    #     torch.cuda.ipc_collect()

    #     self.model = AutoModelForCausalLM.from_pretrained(self.model_name, token=self.access_tk, device_map=self.device_str, torch_dtype=torch.float16, cache_dir=self.cache_dir)
    #     self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, token=self.access_tk, device_map=self.device_str, torch_dtype=torch.float16, padding_side=self.padding_side, cache_dir=self.cache_dir)
    #     self.tokenizer.pad_token = self.tokenizer.eos_token

    #     self.model.eval()
        
    #     self.load_generation_config()

    #     self.logger.info(f"Reloaded model: {self.model_name}")