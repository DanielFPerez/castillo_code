#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import logging
import os
import json
import sys
import numpy as np 

sys.path.append("../")
from src.utils.logger import setup_logger
import src.utils.config as utils_config 



def sanitize_dataset(dataset):
    # Initialize our two new datasets
    degeneration_dataset = []
    sanitized_dataset = []
    
    for sample in dataset:
        # Check if any response hit the max token limit (14999+)
        max_output_size = max(sample["output_sizes"])
        has_max_length_degeneration = max_output_size >= 14999
        
        # Check for high variance degeneration
        has_high_variance_degeneration = (
            sample["output_std"] > 2 * sample["output_mean"] and 
            max_output_size > 8499
        )
        
        if has_max_length_degeneration or has_high_variance_degeneration:
            # Add to the degeneration dataset
            degeneration_dataset.append(sample.copy())
            
            # For sanitized dataset, check if we can clean it
            if has_max_length_degeneration and all(size >= 14999 for size in sample["output_sizes"]):
                # If all responses hit the limit, we can't clean it
                # We don't add it to the sanitized dataset
                continue
            
            # Create a sanitized version
            sanitized_sample = sample.copy()
            
            # Clean the output sizes
            if has_high_variance_degeneration:
                # Remove elements larger than 8499
                sanitized_output_sizes = [size for size in sample["output_sizes"] if size <= 8499]
            else:  # has_max_length_degeneration
                # Remove elements larger than or equal to 14999
                sanitized_output_sizes = [size for size in sample["output_sizes"] if size < 14999]
            
            # If we have sanitized output sizes, recompute metrics
            if sanitized_output_sizes:
                sanitized_sample["output_sizes"] = sanitized_output_sizes
                sanitized_sample["output_mean"] = float(np.mean(sanitized_output_sizes))
                sanitized_sample["output_std"] = float(np.std(sanitized_output_sizes))
                
                # Recompute percentiles
                percentiles = [25, 50, 75, 99]
                percentile_values = np.percentile(sanitized_output_sizes, percentiles)
                sanitized_sample["output_percentiles"] = {
                    f"p{p}": float(val) for p, val in zip(percentiles, percentile_values)
                }
                
                # Set longest response to "TEXT DEGENERATION"
                sanitized_sample["longest_response"] = "TEXT DEGENERATION"
                
                # Add to the sanitized dataset
                sanitized_dataset.append(sanitized_sample)
        else:
            # No degeneration, add original sample to sanitized dataset
            sanitized_dataset.append(sample.copy())
    
    return degeneration_dataset, sanitized_dataset



def process_files(src_dir, dst_dir):
    """
    Process all JSON files in the source directory and save sanitized and degenerate versions.
    
    Args:
        src_dir (str): Path to source directory with JSON files
        dst_dir (str): Base path for output directories
    """
    # Create output directories
    sanitized_dir = os.path.join(dst_dir, "sanitized")
    degenerate_dir = os.path.join(dst_dir, "degenerate")
    
    os.makedirs(sanitized_dir, exist_ok=True)
    os.makedirs(degenerate_dir, exist_ok=True)
    
    logging.info(f"Created output directories: {sanitized_dir} and {degenerate_dir}")
    
    # Get all JSON files in source directory
    json_files = [f for f in os.listdir(src_dir) if f.endswith('.json')]
    logging.info(f"Found {len(json_files)} JSON files in {src_dir}")
    
    for filename in json_files:
        filepath = os.path.join(src_dir, filename)
        logging.info(f"Processing file: {filepath}")
        
        # Extract split name (train, test, validate) from filename
        split_name = filename.replace('original_', '').replace('.json', '')
        
        try:
            # Load the dataset
            with open(filepath, 'r', encoding='utf-8') as f:
                dataset = json.load(f)
            
            logging.info(f"Loaded {len(dataset)} samples from {filename}")
            
            # Apply sanitization
            degeneration_dataset, sanitized_dataset = sanitize_dataset(dataset)
            
            logging.info(f"Identified {len(degeneration_dataset)} samples with degeneration")
            logging.info(f"Created sanitized dataset with {len(sanitized_dataset)} samples")
            
            # Save sanitized dataset
            sanitized_filename = f"sanitized_{split_name}.json"
            sanitized_filepath = os.path.join(sanitized_dir, sanitized_filename)
            
            with open(sanitized_filepath, 'w', encoding='utf-8') as f:
                json.dump(sanitized_dataset, f)
            
            logging.info(f"Saved sanitized dataset to {sanitized_filepath}")
            
            # Save degeneration dataset
            degenerate_filename = f"degenerate_{split_name}.json"
            degenerate_filepath = os.path.join(degenerate_dir, degenerate_filename)
            
            with open(degenerate_filepath, 'w', encoding='utf-8') as f:
                json.dump(degeneration_dataset, f)
            
            logging.info(f"Saved degeneration dataset to {degenerate_filepath}")
            
        except Exception as e:
            logging.error(f"Error processing {filename}: {str(e)}", exc_info=True)



def get_parser():
    """Create and return the argument parser."""
    parser = argparse.ArgumentParser(description="Process and sanitize text generation datasets")
    
    parser.add_argument("--src_dir", type=str, default="castillo/original",
                        help="Source directory containing the default dataset splits")
    parser.add_argument("--dst_dir", type=str, default="castillo",
                        help="Destination directory for sanitized and degenerate datasets")
    parser.add_argument("--log_filename", type=str, default="sanitize_dataset.log",
                        help="Name of the log file")
    parser.add_argument("--log_level", type=str, default="INFO", 
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Logging level")
    parser.add_argument("--log_format", type=str, default="", help="Format for log messages.")
    
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    log_path = os.path.join(utils_config.get_project_root(), f"outputs/logs/{args.log_filename}")

    # log level from arguments
    log_level = getattr(logging, args.log_level, logging.INFO)

    # Setup logging with arguments
    setup_logger(log_file_path=log_path, log_format=args.log_format,
                 console_level=log_level, 
                 file_level=log_level)
    
    # Process the dataset files
    logging.info(f"Starting dataset sanitization process")
    process_files(args.src_dir, args.dst_dir)
    logging.info(f"Dataset sanitization completed successfully")