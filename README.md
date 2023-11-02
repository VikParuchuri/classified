# Classified

Classified helps you classify LLM pretraining or instruction data by quality.  It can be used to summarize dataset quality, filter a dataset, or train custom classifiers (coming soon). It works with datasets on disk or streamed from the Huggingface Hub.

You can classify according to multiple *lenses* (see the `app/lenses` folder for examples).  Each lens rates the data in a different way, according to your needs. Currently, it supports these lenses:

- `instruction_following` - how well outputs follow instructions
- `code_quality` - quality of code pretraining data
- `textbook_quality` - quality of textbook pretraining data
- `learning_value` - learning value of pretraining data (similar to phi)
- `textbook_depth` - depth/specificity of textbook pretraining data

By default, GPT-4 is used to do the rating.  You can also use GPT-3.5-turbo, which is faster but less accurate.  Soon, you'll also be able to train custom models to do the rating.

## Install

You'll need Python 3.9+ (ideally 3.11).

- `git clone https://github.com/VikParuchuri/classified.git`
- `cd classified`
- `poetry install`
- `alembic upgrade head` - this will make a sqlite database

## Configure

You can override any setting in `app/settings.py` by setting an environment variable with the same name, or creating a `local.env` file with the key.

- You will need to set your `OPENAI_KEY`.
- You can also configure `CHAT_MODEL` - this will change the openai model used for rating.  It is `gpt-4` by default.  You can change it to `gpt-3.5-turbo` also.  GPT-4 gives the best results.

# Summarize

## Pretraining

This will give you an overview of how good a pretraining dataset is.  You can rate any dataset on the huggingface hub, or a local dataset file.  It will print some stats to the console, dump them to a csv file, and show the full scoring and rationale in jsonl files inside `app/data`.

Usage example:

`python summary.py vikp/textbooks_grounded markdown textbook_quality --max 25 --workers 2 --stream --version 1`
    
    Raw Scores
    
    Lens                overall
    ----------------  ---------
    textbook_quality        2.8
    
    Summary
    Lens              Poor    Low    Medium    Good      Total
    ----------------  ------  -----  --------  ------  -------
    textbook_quality  0.00%   0.00%  13.04%    86.96%       23


- You can use any dataset on huggingface hub, or a local dataset.  Here we use `vikp/textbook_quality_programming`.
- Specify which column you want to rate (`markdown` in the example).
- Specify which rating lenses you want to run (`textbook_quality` in the example). Comma separate multiple lenses to run them at once.
- `--max` specifies the maximum number of examples you want to rate.  Data will be shuffled first.
- `--workers` specifies the number of parallel workers to use.  This will be limited by your OpenAI rate limit.  With the default GPT-4 rate limit, you will want to set this to `2`.
- `--stream` will stream the dataset from the huggingface hub instead of downloading it all.

Check the help for additional cli options.

## Instruct Data

You can also do this for instruct data.  Usage example:

`python summary.py vikp/code_instructions_filtered instruction,output instruction_following --max 25 --workers 2 --stream`

The main difference from pretraining data is that two fields are specified (`instruction,output`).  This is because the `instruction_following` lens requires two inputs, the instruction and the associated output.

# Caching

All data is cached by default.  The cache is specific to the lens type, input data, and model. Use the `--version` flag if you want to bypass the cache and re-rate, for example if you want to see if the ratings change.

# Adding new lenses

You can add new lenses in the `app/lenses` folder.  You will need to create `system.jinja` for the system prompt, `prompt.jinja` for the main prompt, `config.json` for the lens config, and `functions.json` to store the function call.  Look to the existing tasks for examples.

You can then use the lenses in any of the scripts above.

# Train custom models

The functionality to train custom models and use them to filter larger datasets is coming soon.