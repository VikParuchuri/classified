# Classified

This repository will classify LLM training data by quality, similar to the phi paper.  It can be used to identify dataset quality, or filter a dataset.  By default, this will use gpt-3.5, but you can use gpt-4, or train your own custom classifier to score much larger datasets.

## Install

You will need Python 3.9+ (ideally 3.11).

- `git clone https://github.com/VikParuchuri/classified.git`
- `cd classified`
- `poetry install`
- `alembic revision --autogenerate`
- `alembic upgrade head`