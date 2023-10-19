# Classified

This repository will classify LLM training data by quality, similar to the phi paper.  It can be used to summarize dataset quality, filter a dataset, or train custom classifiers.

You can classify according to multiple *lenses* (see the `app/lenses` folder for examples).  Each lens rates the data in a different way, according to your needs. Currently, it supports the `learning_value` and `code_quality` objectives.  You can easily add your own lenses to classified.

## Install

You'll need Python 3.9+ (ideally 3.11).

- `git clone https://github.com/VikParuchuri/classified.git`
- `cd classified`
- `poetry install`
- `alembic upgrade head` - this will make a sqlite database

## Configure

You can override any settings in `app/settings.py` by setting an environment variable with the same name, or creating a `local.env` file with the key.

- You will need to set your `OPENAI_KEY`.
- You can also configure `CHAT_MODEL` - this will change the openai model used for rating.  It is `gpt-4` by default.  You can change it to `gpt-3.5-turbo` also.  GPT-4 seems to give the best results.

# Usage

## Summarize data quality

This will give you an overview of how good a dataset is on multiple axes.  You can rate any dataset on the huggingface hub, or a local dataset file.  It will print some stats to the console, and also dump them to a csv file.  The full scoring, including rationale given by the llm, will be available in jsonl files.

Usage example:

`python summary.py vikp/textbook_quality_programming markdown textbook_quality --max 25 --workers 1 --stream`

    Evaluation results for vikp/textbook_quality_programming with gpt-3.5-turbo
    Raw Scores
    
    Lens                overall
    ----------------  ---------
    textbook_quality          3
    
    Summary
    Lens              Poor    Low    Medium    Good      Total
    ----------------  ------  -----  --------  ------  -------
    textbook_quality  0.00%   0.00%  4.00%     96.00%       25

- You can use any dataset on huggingface hub, or a local dataset.  Here we use `vikp/textbook_quality_programming`.
- You will need to specify which column you want to rate (`markdown` in the example).
- You will need to specify which rating lenses you want to run (`textbook_quality` in the example). You can comma separate multiple lenses to run them at once.
- `--max` specifies the maximum number of examples you want to rate.  Data will be shuffled first.
- `--workers` specifies the number of parallel workers to use.  This will be limited by your OpenAI rate limit.  With the default GPT-4 rate limit, you will want to set this to `1`.
- `--stream` will stream the dataset from the huggingface hub instead of downloading it all.

Check the help for additional cli options.

# Adding new lenses

You can add new lenses in the `app/lenses` folder.  You will need to create `system.jinja` for the system prompt, `prompt.jinja` for the main prompt, and `functions.json` to store the function call.  Look to the existing tasks for examples.

You can then use the lenses in any of the scripts above.