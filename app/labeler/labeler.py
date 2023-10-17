import json
import os
from typing import List

from jinja2 import Environment, FileSystemLoader

from app.llm.llm import chat_completion
from app.settings import settings
import numpy as np

import tiktoken

tokenizer = tiktoken.encoding_for_model(settings.CHAT_MODEL)


def extract_chunks(resource: str):
    tokenized = tokenizer.encode(resource)
    total_chunks = settings.CHUNKS_PER_DOC
    chunk_offset = 0

    # Gather chunks
    if len(tokenized) > total_chunks * settings.TOKENS_PER_CHUNK:
        chunk_offset = (len(tokenized) - (total_chunks * settings.TOKENS_PER_CHUNK)) // (total_chunks + 1)

    chunks = []
    for i in range(chunk_offset, len(tokenized), chunk_offset + settings.TOKENS_PER_CHUNK):
        # find a smarter split boundary
        start = i
        end = i + settings.TOKENS_PER_CHUNK
        max_dist = settings.MAX_SEARCH_DISTANCE
        start_chunk = ""
        end_chunk = ""

        # Split chunk along newline boundaries if possible
        while start < (i + max_dist) and "\n" not in start_chunk:
            start += 1
            start_chunk = tokenizer.decode(tokenized[start:(start + 2)])

        while end < (i + settings.TOKENS_PER_CHUNK + max_dist) and "\n" not in end_chunk:
            end += 1
            end_chunk = tokenizer.decode(tokenized[(end - 2):end])

        chunk = tokenizer.decode(tokenized[start:end])
        if "\n" in start_chunk:
            chunk = chunk.split("\n", 1)[1]
        if "\n" in end_chunk:
            chunk = chunk.rsplit("\n", 1)[0]

        chunks.append(chunk)
    return chunks


def render_template(template, template_dir, **keys):
    with open(f"{template_dir}/{template}.jinja") as f:
        template_str = f.read()

    template = Environment(
        loader=FileSystemLoader(template_dir)
    ).from_string(template_str)
    instruction = template.render(**keys)
    return instruction


def load_functions(template_dir):
    with open(f"{template_dir}/functions.json") as f:
        functions = json.load(f)
    return functions


def get_final_score(scores):
    final_score = 0
    if all([s >= 3 for s in scores]) and scores[-1] >= 3.5:
        final_score = 1
    return final_score


def rate_data(resource: str, lens_type: str, version: int = 1):
    lens_dir = os.path.join(settings.LENS_DIR, lens_type)
    system_prompt = render_template("system", lens_dir)
    functions = load_functions(lens_dir)
    labels = functions["parameters"]["required"]
    chunks = extract_chunks(resource)
    resource_scores = []
    for chunk in chunks:
        user_prompt = render_template("prompt", lens_dir, resource=chunk)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        chat_response = chat_completion(lens_type, messages, functions, version=version)
        try:
            scores = json.loads(chat_response["function_call"]["arguments"])
            flat_scores = [scores[k] for k in labels]
        except (json.JSONDecodeError, KeyError, TypeError):
            print(f"Unable to extract scores from response: {chat_response}")
            continue

        resource_scores.append(flat_scores)

    if len(resource_scores) == 0:
        return

    # Take the mean of each column value
    resource_scores = np.array(resource_scores)
    resource_scores = list(np.mean(resource_scores, axis=0))
    labeled_scores = {k: v for k, v in zip(labels, resource_scores)}
    labeled_scores["final"] = get_final_score(resource_scores)
    return labeled_scores