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
    chunks = [c.strip() for c in chunks if len(c.strip()) > settings.TOKENS_PER_CHUNK // 4]
    return chunks


def render_template(template, template_dir, **keys):
    with open(f"{template_dir}/{template}.jinja") as f:
        template_str = f.read()

    template = Environment(
        loader=FileSystemLoader(template_dir)
    ).from_string(template_str)
    instruction = template.render(**keys)
    return instruction


def load_function(template_dir):
    with open(f"{template_dir}/function.json") as f:
        functions = json.load(f)
    return functions


def get_final_score(scores):
    final_score = 0
    if all([s >= 2 for s in scores]) and scores[-1] >= 2.5:
        final_score = 1
    return final_score


def rate_data(resource: str, lens_type: str, version: int = 1):
    lens_dir = os.path.join(settings.LENS_DIR, lens_type)
    system_prompt = render_template("system", lens_dir)
    function = load_function(lens_dir)
    labels = function["parameters"]["required"]
    # these are the numeric scores
    score_labels  = [l for l in labels if function["parameters"]["properties"][l]["type"] in ["integer", "float", "number"]]
    # these are the rationale and other CoT data
    rationale_labels = [l for l in labels if function["parameters"]["properties"][l]["type"] == "string"]
    chunks = extract_chunks(resource)
    resource_scores = []
    resource_data = []
    for chunk in chunks:
        user_prompt = render_template("prompt", lens_dir, resource=chunk)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        chat_response = chat_completion(lens_type, messages, [function], version=version)
        try:
            scores = json.loads(chat_response["function_call"]["arguments"])
            flat_scores = [scores[k] for k in score_labels]
            rationales = [scores[k] for k in rationale_labels]
            message = chat_response["content"]
        except (json.JSONDecodeError, KeyError, TypeError):
            print(f"Unable to extract scores from response: {chat_response}")
            continue

        resource_scores.append(flat_scores)
        resource_data.append({
            "scores": flat_scores,
            "rationales": rationales,
            "message": message,
            "chunk": chunk
        })

    if len(resource_scores) == 0:
        return

    # Take the mean of each column value
    resource_scores = np.array(resource_scores)
    resource_scores = list(np.mean(resource_scores, axis=0))
    labeled_scores = {k: v for k, v in zip(score_labels, resource_scores)}
    labeled_scores["final"] = get_final_score(resource_scores)
    return {"summary": labeled_scores, "data": resource_data}
