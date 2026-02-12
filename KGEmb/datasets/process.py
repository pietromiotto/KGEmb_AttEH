"""Knowledge Graph dataset pre-processing functions."""
import argparse
import collections
import os
import pickle

import numpy as np


def get_idx(path):
    """Map entities and relations to unique ids."""
    entities, relations = set(), set()
    for split in ["train", "valid", "test"]:
        file_path = os.path.join(path, split)
        if not os.path.exists(file_path):
            continue
        with open(file_path, "r") as lines:
            for line in lines:
                lhs, rel, rhs = line.strip().split("\t")
                entities.add(lhs)
                entities.add(rhs)
                relations.add(rel)
    ent2idx = {x: i for (i, x) in enumerate(sorted(entities))}
    rel2idx = {x: i for (i, x) in enumerate(sorted(relations))}
    return ent2idx, rel2idx


def to_np_array(dataset_file, ent2idx, rel2idx):
    """Map raw dataset file to numpy array with unique ids."""
    examples = []
    with open(dataset_file, "r") as lines:
        for line in lines:
            lhs, rel, rhs = line.strip().split("\t")
            try:
                examples.append([ent2idx[lhs], rel2idx[rel], ent2idx[rhs]])
            except KeyError:
                continue
    return np.array(examples, dtype=np.int64)


def get_filters(examples, n_relations):
    """Create filtering lists for evaluation."""
    lhs_filters = collections.defaultdict(set)
    rhs_filters = collections.defaultdict(set)
    for lhs, rel, rhs in examples:
        rhs_filters[(lhs, rel)].add(rhs)
        lhs_filters[(rhs, rel + n_relations)].add(lhs)

    lhs_final = {k: sorted(v) for k, v in lhs_filters.items()}
    rhs_final = {k: sorted(v) for k, v in rhs_filters.items()}
    return lhs_final, rhs_final


def process_dataset(path, splits=None):
    """Map entities and relations to ids and save split pickles."""
    if splits is None:
        splits = ["train", "valid", "test"]

    ent2idx, rel2idx = get_idx(path)
    with open(os.path.join(path, "ent2idx.pickle"), "wb") as f:
        pickle.dump(ent2idx, f)
    with open(os.path.join(path, "rel2idx.pickle"), "wb") as f:
        pickle.dump(rel2idx, f)

    examples = {}
    for split in splits:
        file_path = os.path.join(path, split)
        examples[split] = to_np_array(file_path, ent2idx, rel2idx)

    all_examples = np.concatenate([examples[s] for s in splits], axis=0)
    lhs_skip, rhs_skip = get_filters(all_examples, len(rel2idx))
    filters = {"lhs": lhs_skip, "rhs": rhs_skip}
    return examples, filters


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pre-process Knowledge Graph datasets into indexed pickles."
    )
    parser.add_argument(
        "--no_valid", action="store_true",
        help="If set, skip looking for a 'valid' split (process only train & test). This will affect all datasets."
    )
    args = parser.parse_args()

    data_path = os.environ.get("DATA_PATH")
    print(data_path)
    if not data_path:
        raise RuntimeError("Please set the DATA_PATH environment variable to your datasets root.")

    for dataset_name in os.listdir(data_path):
        dataset_path = os.path.join(data_path, dataset_name)

        base_splits = ["train", "valid", "test"]
        if args.no_valid:
            splits = ["train", "test"]
        else:
            splits = [s for s in base_splits if os.path.exists(os.path.join(dataset_path, s))]
            if "valid" not in splits:
                print(f"No '{dataset_path}/valid' file found; processing splits: {splits}")

        examples, filters = process_dataset(dataset_path, splits=splits)
        for split in splits:
            out_file = os.path.join(dataset_path, f"{split}.pickle")
            with open(out_file, "wb") as f:
                pickle.dump(examples[split], f)

        with open(os.path.join(dataset_path, "to_skip.pickle"), "wb") as f:
            pickle.dump(filters, f)

