import json
from pathlib import Path
from typing import Dict, List, Literal, Union

import torch
from lightning import LightningDataModule
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.data.transforms import transform_test
from src.data.utils import pre_caption, FrameLoader
import pandas as pd
import os
from torch.nn.utils.rnn import pad_sequence
import re
import pickle


Image.MAX_IMAGE_PIXELS = None  # Disable DecompressionBombWarning


class FineGDTestDataModule(LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        split: Literal["val", "test"],
        vid_dirs: str,
        annotation: str,
        data_path: str,
        emb_dir: str,
        pkl_path: str = "None",
        num_workers: int = 4,
        pin_memory: bool = True,
        image_size: int = 384,
        vid_query_method: str = "middle",
        vid_frames: int = 1,
        model: str = "clip",
        **kwargs,  # type: ignore
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.model = model
        self.pkl_path = pkl_path

        self.transform_test = transform_test(image_size)

        self.data_test = FineGDDataset(
            transform=self.transform_test,
            annotation=annotation,
            vid_dir=vid_dirs,
            data_path=data_path,
            pkl_path=self.pkl_path,
            emb_dir=emb_dir,
            split=split,
            vid_query_method=vid_query_method,
            vid_frames=vid_frames,
            model=self.model,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
            drop_last=False,
            collate_fn=collate_fn,
        )


def collate_fn(batch):
    """
    Custom collate function for Fabric Lightning that ensures all tensors in a batch
    have the same shape by padding them to the maximum length in the batch.
    """
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


class FineGDDataset(Dataset):
    """
    FineGD dataset, code adapted from miccunifi/CIRCO
    """

    def __init__(
        self,
        transform,
        vid_dir: str,
        data_path: Union[str, Path],
        annotation: str,
        emb_dir: str,
        pkl_path: str,
        split: Literal["train_val", "val", "test"],
        vid_query_method: Literal["middle", "sample"] = "middle",
        vid_frames: int = 1,
        max_words: int = 30,
        model: str = "clip",
    ) -> None:
        """
        Args:
            transform (callable): function which preprocesses the image
            data_path (Union[str, Path]): path to CIRCO dataset
            split (str): dataset split, should be in ['test', 'val']
        """

        self.transform = transform

        self.annotation_pth = Path(annotation)
        assert (
            self.annotation_pth.exists()
        ), f"Annotation file {annotation} does not exist"

        self.annotations = pd.read_csv(annotation)

        self.split = split
        self.pkl_path = pkl_path
        self.max_words = max_words
        self.img_dir = Path(vid_dir)
        self.emb_dir = Path(emb_dir)
        self.model = model
        assert split in [
            "train_val",
            "val",
            "test",
        ], f"Invalid split: {split}, must be one of val or test"
        assert self.img_dir.exists(), f"Image directory {img_dir} does not exist"
        assert self.emb_dir.exists(), f"Embedding directory {emb_dir} does not exist"

        self.img_paths = list(self.img_dir.glob("*.mp4"))
        self.img_ids = [i for i, img in enumerate(self.img_paths)]

        self.img_ids_indexes_map = {
            str(img_id).split("/")[-1].split(".")[0]: i
            for i, img_id in enumerate(self.img_paths)
        }

        self.img_ids_indexes_filenames = {
            v: k for k, v in self.img_ids_indexes_map.items()
        }

        self.tar_ids = []
        for img_info in self.annotations["target_video"]:
            if img_info not in self.img_ids_indexes_map:
                print(img_info)
                continue
            else:
                idx = self.img_ids_indexes_map[img_info]
            self.tar_ids.append(idx)

        self.annotations["target_videos_ids"] = self.annotations[
            "target_videos_ids"
        ].apply(eval)

        # Get maximum number of ground truth images (for padding when loading the images)
        self.max_num_gts = 10

        # Get the embeddings
        emb_pth = Path(f"../{split}_all_embs_{vid_frames}-v2.pt")
        print(f"Loading {split} embeddings from {emb_dir}")

        emb_pths = list(self.emb_dir.glob("*.pth"))

        img_id2emb_pth = {p.stem: p for p in emb_pths}

        embs_list = []
        for vid in tqdm(self.annotations["target_video"]):
            emb = (
                torch.load(img_id2emb_pth[vid], weights_only=True)
                .cpu()
                .to(torch.float32)
            )
            # if emb.ndim != 2:
            if self.model == "clip":
                emb = emb.mean(0)
            elif self.model == "aim":
                emb = emb
            elif self.model == "clip-mlp":
                emb = emb
            elif self.model == "clip-mlp-15":
                emb = emb.mean(0)
            else:
                emb = emb.mean(1).mean(0)  # now [T, D]
                emb = emb.unsqueeze(0)  # ensure 2D: [1, D]
            embs_list.append(emb)

        # now stack safely
        self.embs = torch.stack(embs_list)

        # save
        embs_dict = {
            "ids": self.img_ids,
            "embs": self.embs,
        }
        torch.save(embs_dict, emb_pth)

        self.frame_loader = FrameLoader(
            transform=self.transform, method=vid_query_method, frames_video=vid_frames
        )

        if self.model == "aim":
            PKL_PATH = self.pkl_path
            self.models_dict = load_models_from_pkl(PKL_PATH)
            print(f"Loaded {len(self.models_dict)} emeddings")
        elif self.model == "clip-mlp":
            print("loading images for clip with sinlge frame")
            PKL_PATH = self.pkl_path  # path to your pickle file
            print(f"loading embedding from {PKL_PATH}")
            self.models_dict = load_models_from_pkl(PKL_PATH)
            print(f"Loaded {len(self.models_dict)} models:")
        elif self.model == "clip-mlp-15":
            print("loading images for clip with 15 frame")
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
            PKL_PATH = self.pkl_path
            print(f"loading embedding from {PKL_PATH}")
            self.models_dict = load_models_from_pkl(PKL_PATH)
            print(f"Loaded {len(self.models_dict)} models:")
        else:
            PKL_PATH = self.pkl_path  # path to your pickle file
            print(f"loading embedding from {PKL_PATH}")
            self.models_dict = load_models_from_pkl(PKL_PATH)
            print(f"Loaded {len(self.models_dict)} models:")

    def get_target_img_ids(self, index) -> Dict[str, int]:
        """
        Returns the id of the target image and ground truth images for a given query

        Args:
            index (int): id of the query

        Returns:
             Dict[str, int]: dictionary containing target image id and a list of ground truth image ids
        """

        return {
            "target_img_id": self.img_ids_indexes_map[
                self.annotations.loc[index]["target_video"]
            ],
            "gt_img_ids": [
                str(self.img_ids_indexes_map[x])
                for x in self.annotations.loc[index]["target_videos_ids"]
            ],
        }

    def get_filename_from_img_id(self, img_id: int) -> str:
        # Map from img_id to filename
        return self.img_ids_indexes_filenames[img_id]

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index) -> dict:
        """
        Returns a specific item from the dataset based on the index.

        In 'relative' mode, the dataset yields dictionaries with the following keys:
            - [reference_img, reference_img_id, target_img, target_img_id, relative_caption, shared_concept, gt_img_ids,
            query_id] if split == val
            - [reference_img, reference_img_id, relative_caption, shared_concept, query_id]  if split == test
        """
        # Get the query id
        query_id = str(self.annotations.loc[index]["query_id"])

        # Get relative caption and shared concept
        relative_caption = self.annotations.loc[index]["modification"]
        relative_caption = replace_numbers_with_text(relative_caption)
        relative_caption = pre_caption(relative_caption, self.max_words)

        # Get the reference image

        reference_img_path = str(self.annotations.loc[index]["query_video"])
        reference_img_id = str(self.img_ids_indexes_map[reference_img_path])

        if self.model == "clip":
            reference_img_path = os.path.join(
                self.emb_dir, str(reference_img_path) + ".pth"
            )
        elif self.model == "aim":
            reference_img_path = os.path.join(
                self.emb_dir, str(reference_img_path) + ".pth"
            )
        elif self.model == "clip-mlp":
            reference_img_path = os.path.join(
                self.emb_dir, str(reference_img_path) + ".pth"
            )
        elif self.model == "clip-mlp-15":
            reference_img_path = os.path.join(
                self.emb_dir, str(reference_img_path) + ".pth"
            )
        elif self.model == "blip2-mlp":
            reference_img_path = os.path.join(
                self.emb_dir, str(reference_img_path) + ".pth"
            )
        elif self.model == "blip2-text":
            reference_img_path = os.path.join(
                self.emb_dir, str(reference_img_path) + ".pth"
            )
        else:
            reference_img_path = os.path.join(
                self.img_dir, str(reference_img_path) + ".mp4"
            )

        if self.model == "clip":
            reference_img = (
                torch.load(reference_img_path, weights_only=True)
                .cpu()
                .to(torch.float32)
            )
            reference_img = reference_img.mean(0)
        elif self.model == "aim":
            reference_img = self.models_dict[
                str(self.annotations.loc[index]["query_video"])
            ].to(torch.float32)
        elif self.model == "clip-mlp":
            reference_img = self.models_dict[
                str(self.annotations.loc[index]["query_video"])
            ].to(torch.float32)
        elif self.model == "clip-mlp-15":
            reference_img = self.models_dict[
                str(self.annotations.loc[index]["query_video"])
            ].to(torch.float32)
        elif self.model == "blip2-mlp":
            reference_img = self.models_dict[
                str(self.annotations.loc[index]["query_video"])
            ].to(torch.float32)
        elif self.model == "blip2-text":
            reference_img = self.models_dict[
                str(self.annotations.loc[index]["query_video"])
            ].to(torch.float32)
        elif self.model == "blip":
            reference_img = self.frame_loader(reference_img_path)
        else:
            reference_img = self.frame_loader(reference_img_path)

        if self.split == "test":
            return {
                "reference_img": reference_img,
                "reference_img_id": reference_img_id,
                "relative_caption": relative_caption,
                "query_id": query_id,
                "ref_filename": str(self.annotations.loc[index]["query_video"]),
            }

        # Get the target image and ground truth images
        target_img_path = str(self.annotations.loc[index]["target_video"])
        target_img_id = str(self.img_ids_indexes_map[target_img_path])
        target_img_path = os.path.join(self.emb_dir, str(target_img_path) + ".pth")
        gt_img_ids = [
            str(self.img_ids_indexes_map[x])
            for x in self.annotations.loc[index]["target_videos_ids"]
        ]

        if self.model == "clip":
            target_img = (
                torch.load(target_img_path, weights_only=True).cpu().to(torch.float32)
            )
            target_img = target_img.mean(0)
        elif self.model == "aim":
            target_img = self.models_dict[
                str(self.annotations.loc[index]["target_video"])
            ].to(torch.float32)
        elif self.model == "clip-mlp":
            target_img = self.models_dict[
                str(self.annotations.loc[index]["target_video"])
            ].to(torch.float32)
        elif self.model == "clip-mlp-15":
            target_img = self.models_dict[
                str(self.annotations.loc[index]["target_video"])
            ].to(torch.float32)
        elif self.model == "blip2-mlp":
            target_img = self.models_dict[
                str(self.annotations.loc[index]["target_video"])
            ].to(torch.float32)
        elif self.model == "blip2-text":
            target_img = self.models_dict[
                str(self.annotations.loc[index]["target_video"])
            ].to(torch.float32)
        else:
            target_img = self.models_dict[
                str(self.annotations.loc[index]["target_video"])
            ].to(torch.float32)

        # Pad ground truth image IDs with zeros for collate_fn
        gt_img_ids += [""] * (self.max_num_gts - len(gt_img_ids))

        return {
            "reference_img": reference_img,
            "reference_img_id": reference_img_id,
            "target_img": target_img,
            "target_img_id": target_img_id,
            "relative_caption": relative_caption,
            "gt_img_ids": gt_img_ids,
            "query_id": query_id,
            "ref_filename": str(self.annotations.loc[index]["query_video"]),
        }
