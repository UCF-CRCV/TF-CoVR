import ast
import random
from pathlib import Path
from typing import Dict, List, Literal, Union


import pandas as pd
import torch
from lightning import LightningDataModule
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence


from src.data.transforms import transform_test, transform_train
from src.data.utils import FrameLoader, id2int, pre_caption, id2seq
from src.tools.files import write_txt
from src.tools.utils import print_dist
from src.data.finegd_test import collate_fn, FineGDDataset
import os
from collections import defaultdict
import json
import torch.nn.functional as F


from torch.utils.data.sampler import Sampler
import numpy as np
import re
import pickle

Image.MAX_IMAGE_PIXELS = None  # Disable DecompressionBombWarning


class FineGDCoVRDataModule(LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        num_workers: int = 4,
        model: str = "clip",
        pkl_path: str = "None",
        pin_memory: bool = False,
        annotation: dict = {"train": "", "val": ""},
        vid_dirs: dict = {"train": "", "val": ""},
        emb_dirs: dict = {"train": "", "val": ""},
        image_size: int = 384,
        emb_pool: str = "query",
        iterate: str = "pth2",
        vid_query_method: str = "middle",
        vid_frames: int = 1,
        n_embs: int = 15,
        si_tc_weight=0,
        **kwargs,  # type: ignore
    ) -> None:
        super().__init__()
        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.emb_pool = emb_pool
        self.iterate = iterate
        self.vid_query_method = vid_query_method
        self.vid_frames = vid_frames
        self.model = model
        self.pkl_path = pkl_path

        self.transform_train = transform_train(image_size)
        self.transform_test = transform_test(image_size)

        self.data_train = FineGDCoVRDataset(
            transform=self.transform_train,
            annotation=annotation["train"],
            vid_dir=vid_dirs["train"],
            emb_dir=emb_dirs["train"],
            split="train",
            pkl_path=self.pkl_path,
            emb_pool=self.emb_pool,
            iterate=self.iterate,
            vid_query_method=self.vid_query_method,
            vid_frames=self.vid_frames,
            n_embs=n_embs,
            si_tc_weight=si_tc_weight,
            model=model,
        )

        self.data_val = FineGDDataset(
            transform=self.transform_test,
            annotation=annotation["val"],
            vid_dir=vid_dirs["val"],
            pkl_path=self.pkl_path,
            data_path=None,
            emb_dir=emb_dirs["val"],
            vid_query_method=self.vid_query_method,
            vid_frames=self.vid_frames,
            split="val",  # "val",
            model=model,
        )

    def prepare_data(self):
        # things to do on 1 GPU/TPU (not on every GPU/TPU in DDP)
        # download data, pre-process, split, save to disk, etc...
        pass

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            # pin_memory=self.pin_memory,
            pin_memory=False,
            shuffle=True,
            drop_last=True,
            collate_fn=collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            # pin_memory=self.pin_memory,
            pin_memory=False,
            shuffle=False,
            drop_last=False,
            collate_fn=collate_fn,
        )


class FineGDCoVRTestDataModule(LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        annotation: str,
        vid_dirs: str,
        emb_dirs: str,
        num_workers: int = 4,
        # pin_memory: bool = True,
        pin_memory: bool = False,
        image_size: int = 384,
        emb_pool: str = "query",
        n_embs: int = 15,
        iterate: str = "pth2",
        vid_query_method: str = "middle",
        vid_frames: int = 1,
        **kwargs,  # type: ignore
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.emb_pool = emb_pool
        self.iterate = iterate
        self.vid_query_method = vid_query_method
        self.vid_frames = vid_frames

        self.transform_test = transform_test(image_size)

        self.data_test = CIRCODataset(
            transform=self.transform_test,
            vid_dir=vid_dirs,
            data_path=annotation,
            emb_dir=emb_dirs,
            split="test",
            vid_query_method=vid_query_method,
            vid_frames=vid_frames,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            # pin_memory=self.pin_memory,
            pin_memory=False,
            shuffle=False,
            drop_last=False,
        )


def collate_fn(batch):
    """
    Custom collate function for Fabric Lightning that ensures all tensors in a batch
    have the same shape by padding them to the maximum length in the batch.
    """
    # import pdb; pdb.set_trace()
    # Initialize a dictionary to hold batched data
    batch_dict = {}

    # Iterate over keys in the first item of the batch
    for key in batch[0]:
        # Extract all items corresponding to the current key
        items = [item[key] for item in batch]

        if isinstance(items[0], torch.Tensor):
            # If items are tensors, check for consistent shapes
            if all(item.shape == items[0].shape for item in items):
                # Stack tensors directly if shapes are the same
                batch_dict[key] = torch.stack(items)
            else:
                # Pad sequences to the maximum length if shapes differ
                if items[0].dim() == 1:  # Handling 1D tensors (e.g., sequences)
                    batch_dict[key] = pad_sequence(items, batch_first=True)
                else:
                    # For higher-dimensional tensors, pad each dimension accordingly
                    max_shape = torch.tensor([list(item.shape) for item in items]).max(
                        dim=0
                    )[0]
                    padded_items = []
                    for item in items:
                        padding = [
                            (0, max_dim - item_dim)
                            for item_dim, max_dim in zip(
                                item.shape[::-1], max_shape[::-1]
                            )
                        ]
                        padding = [
                            p for sublist in padding for p in sublist
                        ]  # Flatten the list of tuples
                        padded_items.append(torch.nn.functional.pad(item, padding))
                    batch_dict[key] = torch.stack(padded_items)
        else:
            # For non-tensor items (e.g., strings), return a list of items

            batch_dict[key] = items

    return batch_dict


def load_json(file_path, description):
    """Loads a JSON file and handles exceptions."""
    try:
        with open(file_path, "r") as file:
            return json.load(file)
    except Exception as e:
        print(f"Error loading {description} from {file_path}: {e}")
        return {}


# Define your mapping dict
num2text = {
    0.5: "half",
    1.0: "one",
    1.5: "one and a half",
    2.0: "two",
    2.5: "two and a half",
    3.0: "three",
    3.5: "three and a half",
    4.0: "four",
    4.5: "four and a half",
    5.0: "five",
    5.5: "five and a half",
    6.0: "six",
    6.5: "six and a half",
    7.0: "seven",
    # add more mappings as needed…
}

# Pre‐compile a regex that matches integers or decimals
number_pattern = re.compile(r"\b\d+(\.\d+)?\b")


def replace_numbers_with_text(text: str) -> str:
    """
    Find all numbers in the text and replace them according to num2text dict.
    Unmapped numbers are left as-is.
    """

    def repl(match: re.Match) -> str:
        num_str = match.group(0)
        # Cast to float so "1" and "1.0" both map correctly
        val = float(num_str)
        return num2text.get(val, num_str)

    return number_pattern.sub(repl, text)


def load_models_from_pkl(pkl_path: str) -> dict:
    """
    Loads a pickle file that contains one or more pickled dicts of models.
    Returns a single dict mapping model names to loaded objects.
    """
    models = {}
    with open(pkl_path, "rb") as f:
        try:
            # Keep loading sub-dicts until EOF
            while True:
                subdict = pickle.load(f)
                if isinstance(subdict, dict):
                    models.update(subdict)
                else:
                    raise ValueError("Expected a dict in the pickle stream")
        except EOFError:
            pass
    return models


class FineGDCoVRDataset(Dataset):
    def __init__(
        self,
        transform,
        annotation: str,
        vid_dir: str,
        emb_dir: str,
        split: str,
        pkl_path: str,
        max_words: int = 30,
        emb_pool: str = "query",
        n_embs: int = 15,
        iterate: str = "pth2",
        vid_query_method: str = "middle",
        vid_frames: int = 1,
        si_tc_weight=0,
        model: str = "clip",
    ) -> None:
        super().__init__()

        self.transform = transform

        self.pkl_path = pkl_path
        self.annotation_pth = Path(annotation)
        assert (
            self.annotation_pth.exists()
        ), f"Annotation file {annotation} does not exist"

        self.df = pd.read_csv(annotation)
        self.df["target_labels"] = self.df["target_labels"].apply(eval)
        self.df["target_labels_modification"] = self.df[
            "target_labels_modification"
        ].apply(eval)

        self.vid_dir = Path(vid_dir)
        self.emb_dir = Path(emb_dir)
        self.model = model

        print(f"Loading data for the {self.model}")
        assert self.vid_dir.exists(), f"Image directory {self.vid_dir} does not exist"

        assert split in [
            "train",
            "val",
            "test",
        ], f"Invalid split: {split}, must be one of train, val, or test"
        self.split = split

        vid_pths = self.vid_dir.glob("*.mp4")

        self.query_indices = list(range(len(self.df["query_video"])))

        self.max_words = max_words

        self.emb_pool = emb_pool
        self.n_embs = n_embs

        self.frame_loader = FrameLoader(
            transform=self.transform, method=vid_query_method, frames_video=vid_frames
        )

        if self.model == "aim":
            PKL_PATH = self.pkl_path
            self.models_dict = load_models_from_pkl(PKL_PATH)
            print(f"Loaded {len(self.models_dict)} models:")
        elif self.model == "clip-mlp-15":
            print("loading images for clip with 15 frame")
            PKL_PATH = self.pkl_path  # path to your pickle file
            print(f"loading embedding from {PKL_PATH}")
            self.models_dict = load_models_from_pkl(PKL_PATH)
            print(f"Loaded {len(self.models_dict)} models:")
        elif self.model == "clip-mlp":
            print("loading images for clip with sinlge frame")
            PKL_PATH = self.pkl_path  # path to your pickle file
            print(f"loading embedding from {PKL_PATH}")
            self.models_dict = load_models_from_pkl(PKL_PATH)
            print(f"Loaded {len(self.models_dict)} models:")
        elif self.model == "blip2-mlp":
            print("loading images for blip with sinlge frame")
            PKL_PATH = self.pkl_path  # path to your pickle file
            print(f"loading embedding from {PKL_PATH}")
            self.models_dict = load_models_from_pkl(PKL_PATH)
            print(f"Loaded {len(self.models_dict)} models:")
        elif self.model == "blip2-text":
            print("loading images for blip2 for text")
            PKL_PATH = self.pkl_path  # path to your pickle file
            print(f"loading embedding from {PKL_PATH}")
            self.models_dict = load_models_from_pkl(PKL_PATH)
            print(f"Loaded {len(self.models_dict)} models:")

        else:
            PKL_PATH = self.pkl_path  # path to your pickle file
            print(f"loading embedding from {PKL_PATH}")
            self.models_dict = load_models_from_pkl(PKL_PATH)
            print(f"Loaded {len(self.models_dict)} models:")

    def __len__(self) -> int:
        # return len(self.target_txts)
        return len(self.query_indices)

    def __getitem__(self, index):

        query_idx = self.query_indices[index]
        ann = self.df.loc[query_idx]
        if ann.ndim > 1:
            ann = ann.sample()
            ann = ann.iloc[0]

        if self.model == "clip":
            reference_pth = os.path.join(self.emb_dir, str(ann["query_video"] + ".pth"))
        elif self.model == "aim":
            reference_pth = os.path.join(self.emb_dir, str(ann["query_video"] + ".pth"))
        elif self.model == "blip2-text":
            reference_pth = os.path.join(self.emb_dir, str(ann["query_video"] + ".pth"))
        elif self.model == "blip2-mlp":
            reference_pth = os.path.join(self.emb_dir, str(ann["query_video"] + ".pth"))
        elif self.model == "clip-mlp":
            reference_pth = os.path.join(self.emb_dir, str(ann["query_video"] + ".pth"))
        elif self.model == "clip-mlp-15":
            reference_pth = os.path.join(self.emb_dir, str(ann["query_video"] + ".pth"))
        else:
            reference_pth = os.path.join(self.vid_dir, str(ann["query_video"] + ".mp4"))
        target_pth = os.path.join(self.emb_dir, str(ann["target_video"] + ".pth"))
        source_label = ann["source_label"]
        target_label = ann["target_label"]

        if self.model == "clip":
            reference_vid = torch.load(reference_pth, map_location="cpu").to(
                torch.float32
            )
            reference_vid = reference_vid.mean(0)
        elif self.model == "aim":
            reference_vid = self.models_dict[ann["query_video"]].to(torch.float32)
        elif self.model == "blip":
            reference_vid = self.frame_loader(reference_pth)
        elif self.model == "blip2-mlp":
            reference_vid = self.models_dict[ann["query_video"]].to(torch.float32)
        elif self.model == "blip2-text":
            reference_vid = self.models_dict[ann["query_video"]].to(torch.float32)
        elif self.model == "clip-mlp":
            reference_vid = self.models_dict[ann["query_video"]].to(torch.float32)
        elif self.model == "clip-mlp-15":
            reference_vid = self.models_dict[ann["query_video"]].to(torch.float32)
        else:
            reference_vid = self.frame_loader(reference_pth)

        modification = ann["target_labels_modification"][str(target_label)][
            "modification"
        ]
        modification = replace_numbers_with_text(modification)
        caption = pre_caption(modification, self.max_words)

        return_dict = {
            "ref_img": reference_vid,
            "edit": caption,
            "pair_id": index,
            "source_label": source_label,
            "target_label": target_label,
            "ref_filename": ann["query_video"],
        }

        if self.model == "clip":
            target_emb = (
                torch.load(target_pth, map_location="cpu").cpu().to(torch.float32)
            )
            target_emb = target_emb.mean(0)
        elif self.model == "aim":
            target_emb = self.models_dict[ann["target_video"]].to(torch.float32)
        elif self.model == "blip2-mlp":
            target_emb = self.models_dict[ann["target_video"]].to(torch.float32)
        elif self.model == "blip2-text":
            target_emb = self.models_dict[ann["target_video"]].to(torch.float32)
        elif self.model == "clip-mlp":
            target_emb = self.models_dict[ann["target_video"]].to(torch.float32)
        elif self.model == "clip-mlp-15":
            target_emb = self.models_dict[ann["target_video"]].to(torch.float32)
        else:
            target_emb = self.models_dict[ann["target_video"]].to(torch.float32)

        return_dict["tar_img_feat"] = target_emb
        return return_dict
