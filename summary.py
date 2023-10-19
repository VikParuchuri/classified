import argparse
import os
from collections import defaultdict

from datasets import IterableDataset
from app.labeler.tasks import run_rating_lenses
from app.settings import settings
import json

import datasets

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Summarize the quality of a dataset.")
    parser.add_argument("dataset", help="Input resources to score, as a huggingface dataset.  Either a path to a local dataset, or a huggingface dataset name.", type=str)
    parser.add_argument("data_column", help="The column name to look for the data under.")
    parser.add_argument("lenses", help="The lenses to score the data on, as a comma-separated list.", type=str)
    parser.add_argument("--max", type=int, default=None, help="Maximum number of resources to score.  This will sample randomly.")
    parser.add_argument("--workers", type=int, default=5, help="Number of workers to use")
    parser.add_argument("--split", type=str, default="train", help="Which dataset split to use")
    parser.add_argument("--version", type=int, default=1, help="Which version of the lenses to generate.  Enables skipping cache if incremented.")
    parser.add_argument("--stream", action="store_true", help="Stream the dataset instead of loading it all into memory.", default=False)
    args = parser.parse_args()

    # Stream dataset if we set a max number of rows
    if not os.path.exists(args.dataset):
        dataset = datasets.load_dataset(args.dataset, split=args.split, streaming=args.stream)
    else:
        dataset = datasets.load_from_disk(args.dataset)

    if args.max is not None:
        dataset = dataset.shuffle(seed=42)

    if isinstance(dataset, IterableDataset):
        dataset = dataset.take(args.max)
        resources = list([item[args.data_column] for item in dataset])
    else:
        if args.max is not None:
            dataset = dataset[:args.max]
        resources = list(dataset[args.data_column])

    lens_names = args.lenses.split(",")
    scores = run_rating_lenses(lens_names, resources, version=1, workers=args.workers)
    for lens in scores:
        averages = defaultdict(list)
        good_count = 0
        not_null = 0
        data_fname = os.path.join(settings.DATA_DIR, f"{lens}.jsonl")
        with open(data_fname, "w+") as f:
            for item in scores[lens]:
                if item is None:
                    continue
                summary = item["summary"]
                data = item["data"]
                for k, v in summary.items():
                    averages[k].append(v)

                not_null += 1
                good_count += summary["final"]

                f.write(json.dumps(data) + "\n")

        averages = {k: round((sum(v) / len(v)) * 10) / 10 for k, v in averages.items()}
        del averages["final"]

        print(f"Lens: {lens}")
        print(f"Percentage of high quality data: {good_count / not_null:.2%}")
        print(f"Raw scores: {averages}")







