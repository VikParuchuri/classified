from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

from tqdm import tqdm

from app.labeler.labeler import rate_data

import faulthandler
import signal
faulthandler.register(signal.SIGUSR1.value)


def run_rating_lenses(lens_types: List[str], resources: List[str], version: int = 1, workers: int = 5):
    final_scores = {}

    for lens in lens_types:
        scores = []
        progress_bar = tqdm(total=len(resources), position=0, leave=True, desc=f"Running {lens}", dynamic_ncols=True)
        with ThreadPoolExecutor(max_workers=5) as executor:
            # A list to hold the future objects.
            futures = []
            # Submitting tasks to the executor.
            for i in range(len(resources)):
                future = executor.submit(rate_data, resources[i], lens, version)
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

                with progress_bar.get_lock():
                    progress_bar.update(1)

        progress_bar.close()
        final_scores[lens] = scores
    return final_scores