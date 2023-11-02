from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
from typing import List

from tqdm import tqdm

from app.labeler.lens import Lens
from app.labeler.raters import pretrain, instruct

import faulthandler
import signal
import traceback
faulthandler.register(signal.SIGUSR1.value)


def run_rating_lenses(lens_types: List[str], resources: List[List[str]], version: int = 1, workers: int = 5):
    final_scores = {}

    for lens_type in lens_types:
        lens = Lens(lens_type)

        match lens.rater_type():
            case "pretrain":
                func = pretrain.rate_data
            case "instruct":
                func = instruct.rate_data
            case _:
                raise ValueError(f"Unknown rater type {lens.rater_type()} for lens {lens_type}")

        # Check that the number of input fields matches the number of inputs
        assert len(resources[0]) == len(lens.input_fields), f"Number of needed input fields {len(lens.input_fields)} does not match number of inputs {len(resources[0])}"

        scores = []
        progress_bar = tqdm(total=len(resources), position=0, leave=True, desc=f"Running {lens_type}", dynamic_ncols=True)
        with ProcessPoolExecutor(max_workers=workers) as executor:
            # A list to hold the future objects.
            futures = []
            # Submitting tasks to the executor.
            for i in range(len(resources)):
                future = executor.submit(func, resources[i], lens_type, version)
                futures.append(future)

            # Iterate over the futures and wait for each to complete with a timeout.
            for future in as_completed(futures):
                try:
                    result = future.result(timeout=120)
                    scores.append(result)
                except TimeoutError:
                    print("Rating a resource took longer than 120s.")
                except Exception as e:
                    print(f"Unhandled exception when rating: {e}")
                    traceback.print_exc()

                with progress_bar.get_lock():
                    progress_bar.update(1)

        progress_bar.close()
        final_scores[lens_type] = scores
    return final_scores