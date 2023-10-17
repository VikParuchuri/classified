from itertools import chain
from typing import List
from tqdm.contrib.concurrent import process_map

from app.labeler.labeler import rate_data


def run_rating_tasks(task_types: List[str], resources: List[str], version: int = 1, workers: int = 5):
    final_scores = {}

    for task in task_types:
        scores = process_map(rate_data, resources, [task] * len(resources), [version] * len(resources), desc=f"Rating {task}", max_workers=workers, chunksize=1)
        final_scores[task] = scores
    return final_scores