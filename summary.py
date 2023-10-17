import argparse
import os
from collections import defaultdict
from tqdm.contrib.concurrent import process_map

from app.labeler.tasks import run_rating_tasks
import datasets

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Summarize the quality of a dataset.")
    parser.add_argument("dataset", help="Input resources to score, as a huggingface dataset.  Either a path to a local dataset, or a huggingface dataset name.", type=str)
    parser.add_argument("data_column", help="The column name to look for the data under.")
    parser.add_argument("tasks", help="The tasks to score the data on, as a comma-separated list.", type=str)
    parser.add_argument("--max", type=int, default=None, help="Maximum number of resources to score.  This will sample randomly.")
    parser.add_argument("--workers", type=int, default=5, help="Number of workers to use")
    parser.add_argument("--split", type=str, default="train", help="Which dataset split to use")
    parser.add_argument("--version", type=int, default=1, help="Which version of the task to generate.  Enables skipping cache if incremented.")
    args = parser.parse_args()

    if not os.path.exists(args.dataset):
        dataset = datasets.load_dataset(args.dataset, split=args.split)
    else:
        dataset = datasets.load_from_disk(args.dataset)

    if args.max is not None:
        dataset = dataset.shuffle(seed=42)
        dataset = dataset[:args.max]

    resources = list(dataset[args.data_column])

    task_names = args.tasks.split(",")
    scores = run_rating_tasks(task_names, resources, version=1, workers=args.workers)
    for task in scores:
        averages = defaultdict(list)
        good_count = 0
        not_null = 0
        for item in scores[task]:
            if item is None:
                continue
            for k, v in item.items():
                averages[k].append(v)

            not_null += 1
            good_count += item["final"]

        averages = {k: sum(v) / len(v) for k, v in averages.items()}
        print(f"Task: {task}")
        print(f"Percentage of high quality data: {good_count / not_null}")
        print(f"Raw scores: {averages}")







