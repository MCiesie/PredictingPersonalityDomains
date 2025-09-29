import random
from collections import defaultdict

# shuffle data according to PID
def shuffle_data(metadata):
    patient_to_indices = defaultdict(list)
    for idx, meta in enumerate(metadata):
        pid = meta["PID"]
        patient_to_indices[pid].append(idx)

    patient_ids = list(patient_to_indices.keys())
    random.seed(42)
    random.shuffle(patient_ids)

    # data split akin to paper
    num_patients = len(patient_ids)
    train_end = int(0.57 * num_patients)
    val_end = int(0.76 * num_patients)

    train_pids = set(patient_ids[:train_end])
    val_pids = set(patient_ids[train_end:val_end])
    test_pids = set(patient_ids[val_end:])

    train_idx, val_idx, test_idx = [], [], []
    for pid, idxs in patient_to_indices.items():
        if pid in train_pids:
            train_idx.extend(idxs)
        elif pid in val_pids:
            val_idx.extend(idxs)
        else:
            test_idx.extend(idxs)

    return train_idx, val_idx, test_idx


# Subset tensors by index
def subset_data(data_dict, idx_list):
    return {k: v[idx_list] for k, v in data_dict.items()}
