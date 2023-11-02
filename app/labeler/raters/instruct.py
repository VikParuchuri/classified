import json
from typing import List

from app.labeler.lens import Lens
from app.labeler.raters.common import get_final_score
from app.llm.llm import chat_completion


def rate_data(resource: List[str], lens_type: str, version: int = 1):
    lens = Lens(lens_type)
    instruction, output = resource
    user_prompt = lens.prompt_template(instruction, output)
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
        return

    resource_data = [{
        "scores": flat_scores,
        "rationales": rationales,
        "message": message,
        lens.config["input_fields"][0]: instruction,
        lens.config["input_fields"][1]: output
    }]
    labeled_scores = {k: v for k, v in zip(lens.score_labels(), flat_scores)}
    labeled_scores["final"] = get_final_score(flat_scores)
    return {"summary": labeled_scores, "data": resource_data}