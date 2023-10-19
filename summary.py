import argparse
import os
from collections import defaultdict

from datasets import IterableDataset
from app.labeler.tasks import run_rating_lenses
from app.settings import settings
import json
from tabulate import tabulate

import datasets


def rate_single_dataset(dataset_name, args):
    # Stream dataset if we set a max number of rows
    stream = args.stream
    if not os.path.exists(dataset_name):
        dataset = datasets.load_dataset(dataset_name, split=args.split, streaming=stream)
    else:
        dataset = datasets.load_from_disk(dataset_name)

    if args.max is not None:
        dataset = dataset.shuffle(seed=42)
        if stream:
            dataset = dataset.take(args.max)
        else:
            dataset = dataset[:args.max]

    if stream:
        resources = list([item[args.data_column] for item in dataset])
    else:
        resources = list(dataset[args.data_column])

    lens_names = args.lenses.split(",")
    scores = run_rating_lenses(lens_names, resources, version=args.version, workers=args.workers)
    table_data = []
    print(f"{dataset_name} results")
    print("Raw Scores")
    for lens in scores:
        table_row = [lens]
        averages = defaultdict(list)
        good_count = defaultdict(int)
        not_null = 0

        base_folder = os.path.join(settings.DATA_DIR, dataset_name.replace("/", "_"))
        data_fname = os.path.join(base_folder, f"{lens}.jsonl")
        os.makedirs(base_folder, exist_ok=True)
        with open(data_fname, "w+") as f:
            for item in scores[lens]:
                if item is None:
                    continue
                summary = item["summary"]
                data = item["data"]
                for k, v in summary.items():
                    averages[k].append(v)

                not_null += 1
                good_count[summary["final"]] += 1

                for chunk in data:
                    f.write(json.dumps(chunk) + "\n")
        table_row += [f"{good_count[i] / not_null:.2%}" for i in range(4)]
        table_row.append(not_null)
        averages = {k: round((sum(v) / len(v)) * 10) / 10 for k, v in averages.items()}
        del averages["final"]
        table_data.append(table_row)
        print()
        avg_headers, avg_data = zip(*averages.items())
        avg_data = [lens] + list(avg_data)
        avg_headers = ["Lens"] + list(avg_headers)
        print(tabulate([avg_data], headers=avg_headers))

    print()
    print("Summary")
    table_headers = ["Lens", "Poor", "Low", "Medium", "Good", "Total"]
    print(tabulate(table_data, headers=table_headers))

    return table_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Summarize the quality of a dataset.")
    parser.add_argument("dataset", help="Input resources to score, as a huggingface dataset.  Either a path to a local dataset, or a huggingface dataset name. Comma separate to run multiple.", type=str)
    parser.add_argument("data_column", help="The column name to look for the data under.", default="markdown")
    parser.add_argument("lenses", help="The lenses to score the data on, as a comma-separated list.", type=str, default="textbook_quality")
    parser.add_argument("--max", type=int, default=None, help="Maximum number of resources to score.  This will sample randomly.")
    parser.add_argument("--workers", type=int, default=1, help="Number of workers to use")
    parser.add_argument("--split", type=str, default="train", help="Which dataset split to use")
    parser.add_argument("--version", type=int, default=1, help="Which version of the lenses to generate.  Enables skipping cache if incremented.")
    parser.add_argument("--stream", action="store_true", help="Stream the dataset instead of loading it all into memory.", default=False)
    args = parser.parse_args()

    dataset_names = args.dataset.split(",")
    csv_rows = {}
    for dataset_name in dataset_names:
        dataset_rows = rate_single_dataset(dataset_name, args)
        csv_rows[dataset_name] = dataset_rows

    summary_path = os.path.join(settings.DATA_DIR, "summary.csv")
    csv_data = ["Dataset,Lens,Poor,Low,Medium,Good,Total"]
    for dataset_name, dataset_rows in csv_rows.items():
        for row in dataset_rows:
            csv_data.append(",".join([dataset_name] + [str(r) for r in row]))

    with open(summary_path, "w+") as f:
        f.write("\n".join(csv_data))

    print(f"Summary available at {summary_path}, and additional information, including the rationales available in dataset-specific folders at {settings.DATA_DIR}")






