from tenacity import retry, wait_random_exponential, stop_after_attempt
import openai

from app.llm.exceptions import RateLimitError
from app.llm.models import query_cached_response, save_cached_response
from app.settings import settings
from typing import List, Dict
import time

openai.api_key = settings.OPENAI_KEY


@retry(wait=wait_random_exponential(multiplier=1, min=30), stop=stop_after_attempt(3))
def instruct_completion(prompt: str, model=settings.INSTRUCT_MODEL, max_tokens=settings.MAX_GENERATION_TOKENS, temperature=.2) -> str | None:
    try:
        response = openai.Completion.create(
            model=model,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        response_message = response["choices"][0]["text"]
        return response_message
    except openai.error.RateLimitError as e:
        print(f"Rate limit error: {e}")
        time.sleep(60)
        # Raise error to retry
        raise RateLimitError
    except Exception as e:
        print(f"Unable to generate Completion response: {e}")


@retry(wait=wait_random_exponential(multiplier=1, min=30), stop=stop_after_attempt(3))
def _chat_completion(messages, functions, model, max_tokens, temperature) -> str | None:
    function_call = "auto"
    if len(functions) == 1:
        # Force it to call the function
        function_call = {"name": functions[0]["name"]}

    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            functions=functions,
            function_call=function_call,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        response_message = response["choices"][0]["message"]
        return response_message
    except openai.error.RateLimitError as e:
        print(f"Rate limit error: {e}")
        time.sleep(60)
        # Raise error to retry
        raise RateLimitError
    except Exception as e:
        print(f"Unable to generate ChatCompletion response: {e}")


def chat_completion(lens_type, messages, functions: None | List[Dict] = None, model=settings.CHAT_MODEL, max_tokens=settings.MAX_GENERATION_TOKENS, temperature=.2, version=1, cache=True):
    if cache:
        response = query_cached_response(lens_type, messages, functions, version)
        if response:
            return response.response

    response = _chat_completion(messages, functions, model, max_tokens, temperature)

    if cache and response:
        save_cached_response(lens_type, messages, functions, response, version)

    return response