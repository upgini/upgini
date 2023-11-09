import itertools
from typing import List, Tuple


def combine_search_keys(search_keys: List[str]) -> List[Tuple[str]]:
    combined_search_keys = []
    for L in range(1, len(search_keys) + 1):
        for subset in itertools.combinations(search_keys, L):
            combined_search_keys.append(subset)
    return combined_search_keys
