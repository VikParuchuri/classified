import json
from typing import List

from app.labeler.lens import Lens
from app.labeler.raters.common import get_final_score, extract_chunks
from app.llm.llm import chat_completion
import numpy as np


def rate_data(resource: List[str], lens_type: str, version: int = 1):
    lens = Lens(lens_type)
    chunks = extract_chunks(resource[0])
    resource_scores = []
    resource_data = []
    for chunk in chunks:
        user_prompt = lens.prompt_template(chunk)
        messages = [
            {"role": "system", "content": lens.system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        chat_response = chat_completion(lens_type, messages, [lens.function], version=version)
        try:
            scores = json.loads(chat_response["function_call"]["arguments"])
            flat_scores = [scores[k] for k in lens.score_labels()]
            rationales = [scores[k] for k in lens.rationale_labels()]
            message = chat_response["content"]
        except (json.JSONDecodeError, KeyError, TypeError):
            print(f"Unable to extract scores from response: {chat_response}")
            continue

        resource_scores.append(flat_scores)
        resource_data.append({
            "scores": flat_scores,
            "rationales": rationales,
            "message": message,
            lens.config["input_fields"][0]: chunk
        })

    if len(resource_scores) == 0:
        return

    # Take the mean of each column value
    resource_scores = np.array(resource_scores)
    resource_scores = list(np.mean(resource_scores, axis=0))
    labeled_scores = {k: v for k, v in zip(lens.score_labels(), resource_scores)}
    labeled_scores["final"] = get_final_score(resource_scores)
    return {"summary": labeled_scores, "data": resource_data}
