"""Extracts and saves the pretrained model vector model on the problem text"""
import logging
from pathlib import Path

import torch

from math_qa.math_qa import load_all_dataset
from models.pretrained_lm import get_pretrained_albert

import config


def get_vector_dir(part: str) -> Path:
    return config.PRETRAINED_PROBLEM_VECTOR_DIR / part


def get_vector_path(part: str, index: int) -> Path:
    return get_vector_dir(part) / f"{index}.pt"


def extract_vectors():
    transformer, tokenizer = get_pretrained_albert()
    transformer.eval()
    with torch.no_grad():
        all_data = load_all_dataset()

        for part, entries in all_data.items():
            vector_dir = get_vector_dir(part)
            vector_dir.mkdir(exist_ok=True, parents=True)

            for i, entry in enumerate(entries):
                logging.info(f"extracting vector part={part} i={i} out of {len(entries)}")
                tokens = tokenizer(entry.problem,
                                   return_tensors='pt',
                                   padding=True,
                                   truncation=True,
                                   max_length=transformer.config.max_position_embeddings)
                output = transformer(**tokens)
                vector_path = get_vector_path(part, i)
                torch.save(output.pooler_output, vector_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    extract_vectors()
