import numpy as np


def kfold_split(
        N: int, k: int, shuffle: bool=True, rng: np.random.Generator | None=None
        ) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Returns: list of length k with tuples (train_idx, val_idx)
        Iff shuffle, shuffle the indices [0...N-1] before splitting
    Recall that first, we split the N indices into k groups. Then, for each iteration,
        use one of the groups for testing (validation), the rest for training
    By convention, if N % k != 0, put extras in the first of the k groups
    """
    if rng is None:
        rng = np.random.default_rng()
    indices = np.arange(N)
    if shuffle:
        indices = rng.permutation(indices)
    k_fold_split: list[np.ndarray] = []
    testing_group_idxs: list[np.ndarray] = []
    base_group_size = N // k
    cur_idx = 0
    while cur_idx < N:
        idxs_left = N - cur_idx
        cur_group_size = base_group_size
        if idxs_left % base_group_size != 0:
            cur_group_size += 1
        testing_group_idxs.append(np.arange(cur_idx, cur_idx+cur_group_size))
        cur_idx += cur_group_size
    return [
        (np.delete(indices, group), indices[group])
        for group in testing_group_idxs
    ]
