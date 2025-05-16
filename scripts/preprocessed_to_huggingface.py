import argparse
import logging
import os
import json
import sys
from pathlib import Path
from glob import glob

sys.path.append("../")
from src.utils.logger import setup_logger
import src.utils.config as utils_config 
import src.data.rawdata_processor as rawdataproc


def merge_json_files(src_dir, dst_file, filter_gemma=True):
    """Merges all JSON files in a directory into a single JSON list."""
    logger = logging.getLogger(__name__)
    all_data = []
    json_files = glob(os.path.join(src_dir, "*.json"))
    logger.info(f"Found {len(json_files)} files in {src_dir}")
    if not json_files:
        logger.warning(f"No JSON files found in {src_dir}.")
        return
    
    for file_path in json_files:
        if "gemma-3-27b" in file_path and filter_gemma:
            logger.info(f"Skipping file: {file_path}")
            continue
        try:
            logger.debug(f"Reading file: {file_path}")
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    all_data.extend(data)
                else:
                    logger.warning(f"File {file_path} did not contain a list. Skipping.")
        except Exception as e:
            logger.error(f"Error reading {file_path}: {e}")

    logger.info(f"Merged {len(all_data)} samples from {src_dir}")
    with open(dst_file, "w", encoding="utf-8") as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)

def main(raw_src_dir, dst_dir, config_name):
    """Main function to merge all splits into HuggingFace-style JSON."""
    logger = logging.getLogger(__name__)
    splits = ["train", "validate", "test"]
    output_base = Path(dst_dir) / config_name
    output_base.mkdir(parents=True, exist_ok=True)

    for split in splits:
        src_path = Path(raw_src_dir) / split
        if not src_path.exists():
            logger.error(f"Split directory not found: {src_path}")
            raise FileNotFoundError(f"Split directory not found: {src_path}")
        
        output_file = output_base / f"{config_name}_{split}.json"
        merge_json_files(src_path, output_file)
        logger.info(f"Saved merged file for split '{split}' to: {output_file}")


def get_parser():
    """Defines and returns the argument parser for the script."""
    parser = argparse.ArgumentParser(description="Merge preprocessed JSON samples into one json and saves it in the HuggingFace dataset repo.")

    parser.add_argument("--raw_src_dir", type=str, required=True,
                        help="Top-level directory containing 'train', 'validate', and 'test' subdirectories.")
    parser.add_argument("--dst_dir", type=str, required=True,
                        help="Destination directory to store merged JSON files.")
    parser.add_argument("--config_name", type=str, required=True,
                        help="Name of the config used as directory and file prefix.")
    
    parser.add_argument("--log_filename", type=str, required=True,
                        help="File path to save log outputs.")
    
    parser.add_argument("--log_format", type=str, default="", help="Format for log messages.")
    parser.add_argument("--log_level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Logging level for the logger.")

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

    logging.info("Starting preprocessing to HuggingFace format.")
    main(args.raw_src_dir, args.dst_dir, args.config_name)
    logging.info("Processing complete.")
