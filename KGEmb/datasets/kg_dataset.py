import os
import pickle as pkl

import numpy as np
import torch


class KGDataset(object):
    """Knowledge Graph dataset class, with auto-detection of 'valid' split and override flag."""

    def __init__(self, data_path, debug, no_valid=False):
        """
        Args:
            data_path: Path to directory containing train/valid/test pickle files produced by process.py
            debug: boolean indicating whether to use debug mode (if True, only first 1000 examples)
            no_valid: boolean indicating whether to skip loading the 'valid' split even if present
        """
        self.data_path = data_path
        self.debug = debug
        self.no_valid = no_valid
        self.data = {}

        
        base_splits = ["train", "valid", "test"]
        if self.no_valid:
            splits = ["train", "test"]
        else:
            splits = [s for s in base_splits
                      if os.path.exists(os.path.join(self.data_path, f"{s}.pickle"))]

        
        if "valid" not in splits and not self.no_valid:
            print(f"⚠️  no '{self.data_path}/valid.pickle' found; loading splits={splits}")

       
        if "train" not in splits or "test" not in splits:
            raise ValueError("Both 'train.pickle' and 'test.pickle' must be present in data_path.")

        
        for split in splits:
            file_path = os.path.join(self.data_path, f"{split}.pickle")
            with open(file_path, "rb") as f:
                self.data[split] = pkl.load(f)

       
        with open(os.path.join(self.data_path, "to_skip.pickle"), "rb") as f:
            self.to_skip = pkl.load(f)

       
        max_axis = np.max(self.data["train"], axis=0)
        self.n_entities = int(max(max_axis[0], max_axis[2]) + 1)
        self.n_predicates = int(max_axis[1] + 1) * 2

    def get_examples(self, split, rel_idx=-1):
        """Get examples in a split.

        Args:
            split: String indicating the split to use (train/valid/test)
            rel_idx: integer for relation index to keep (-1 to keep all relations)

        Returns:
            examples: torch.LongTensor containing KG triples in the split
        """
        if split not in self.data:
            raise KeyError(f"Split '{split}' is not loaded in this dataset.")

        examples = self.data[split]
        if split == "train":
            # this adds inverse relations for training.
            #N.W. inverse edges are created here, so AFTER the data was split transductively.
            #Therefore no data leakege from inverse edges is possible
            copy = np.copy(examples)
            copy[:, [0, 2]] = copy[:, [2, 0]]
            copy[:, 1] += self.n_predicates // 2
            examples = np.vstack((examples, copy))

        if rel_idx >= 0:
            examples = examples[examples[:, 1] == rel_idx]

        if self.debug:
            examples = examples[:1000]

        return torch.from_numpy(examples.astype("int64"))

    def get_filters(self):
        """Return filter dict to compute ranking metrics in the filtered setting."""
        return self.to_skip

    def get_shape(self):
        """Returns KG dataset dimensions: (n_entities, n_predicates, n_entities)"""
        return self.n_entities, self.n_predicates, self.n_entities

