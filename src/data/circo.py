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


class CIRCOTestDataModule(LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        split: Literal["val", "test"],
        vid_dirs: str,
        data_path: str,
        emb_dir: str,
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

        self.transform_test = transform_test(image_size)

        self.data_test = CIRCODataset(
            transform=self.transform_test,
            vid_dir=vid_dirs,
            data_path=data_path,
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


class CIRCODataset(Dataset):
    """
    CIRCO dataset, code adapted from miccunifi/CIRCO
    """

    def __init__(
        self,
        transform,
        vid_dir: str,
        data_path: Union[str, Path],
        emb_dir: str,
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
        data_path = "/home/animesh/FineGym-CoVR-Github/annotations/"
        data_path = Path(data_path)
        assert data_path.exists(), f"Annotation file {data_path} does not exist"
        self.split = split
        self.max_words = max_words
        img_dir = f"/home/animesh/FineGD_data_vid_all"
        self.img_dir = Path(img_dir)
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
        # for f in self.img_dir.glob("*.mp4"):
        #     print(f)
        # import pdb; pdb.set_trace()

        # Load COCO images information
        # with open(
        #     "/home/agupta2/FineGym-CoVR-Github/annotations/image_info.json",
        #     "r",
        # ) as f:
        #     imgs_info = json.load(f)
        # imgs_info = pd.read_csv(f)

        # self.img_paths = [
        #     img_dir / Path(img_info["file_name"]) for img_info in imgs_info["images"]
        # ]
        # self.img_ids = [img_info["id"] for img_info in imgs_info["images"]]
        self.img_ids = [i for i, img in enumerate(self.img_paths)]

        self.img_ids_indexes_map = {
            str(img_id).split("/")[-1].split(".")[0]: i
            for i, img_id in enumerate(self.img_paths)
        }

        self.img_ids_indexes_filenames = {
            v: k for k, v in self.img_ids_indexes_map.items()
        }

        # import pdb; pdb.set_trace()

        # self.img_path_ids_map = {
        #     img_info["file_name"]: img_info["id"] for img_info in imgs_info["images"]
        # }

        # import pdb; pdb.set_trace()
        # get CIRCO annotations
        print(f"Loading {split} annotations from {data_path}")
        with open(data_path / f"{split}_video_pairs_v2.csv", "r") as f:
            # with open("/home/animesh/FineGym-CoVR-Github/annotations/Final_gym_captioned_combined_videos_test.csv", "r") as f:
            # self.annotations: List[dict] = json.load(f)
            self.annotations = pd.read_csv(f)

        # import pdb; pdb.set_trace()
        self.tar_ids = []
        for img_info in self.annotations["target_video"]:
            if img_info not in self.img_ids_indexes_map:
                print(img_info)
                continue
            else:
                idx = self.img_ids_indexes_map[img_info]
            self.tar_ids.append(idx)

        # self.tar_ids = [self.img_ids_indexes_map[str(img_info)] for img_info in self.annotations["target_video"]]

        self.annotations["target_videos_ids"] = self.annotations[
            "target_videos_ids"
        ].apply(eval)

        # import pdb; pdb.set_trace()
        # for row in self.annotations["target_videos_ids"]:
        #     for vid in row:
        #         if vid not in self.img_ids_indexes_map:
        #             print(vid)
        #             continue
        #         else:
        #             idx = self.img_ids_indexes_map[vid]
        #         self.tar_ids.append(idx)
        #         self.target_files_gt.append(vid)

        # self.tar_ids = list(set(self.tar_ids))
        # self.target_files_gt = list(set(self.target_files_gt))

        # Get maximum number of ground truth images (for padding when loading the images)
        # self.max_num_gts = 23  # Maximum number of ground truth images

        self.max_num_gts = 10

        # Get the embeddings
        # import pdb; pdb.set_trace()
        emb_pth = Path(f"../{split}_all_embs_{vid_frames}-v2.pt")
        print(f"Loading {split} embeddings from {emb_dir}")
        # if emb_pth.exists():
        #     embs_dict = torch.load(emb_pth, weights_only=True)
        #     self.embs = embs_dict["embs"]
        #     assert self.img_ids == embs_dict["ids"], "Image IDs do not match"
        # else:

        emb_pths = list(self.emb_dir.glob("*.pth"))
        # assert (
        #     len(emb_pths) == len(self.img_ids)
        # ), f"Number of embeddings {len(emb_pths)} does not match number of images {len(self.img_ids)}"

        img_id2emb_pth = {p.stem: p for p in emb_pths}

        # Use a list to store embeddings temporarily
        # embs_list = []

        # # import pdb; pdb.set_trace()
        # for data in tqdm(self.annotations["target_video"]):
        #     # img_id = data.get("target_img_id", None)

        # # for img_id in tqdm(self.img_ids):
        #     emb = torch.load(img_id2emb_pth[data], weights_only=True)
        #     emb = emb.cpu()

        #     if len(emb.shape) != 2:
        #         emb = emb.mean(1).mean(0)
        #         # continue
        #     # else:
        #     embs_list.append(emb)  # Move to CPU to free up GPU memory

        #     # Save in chunks to avoid excessive memory usage
        #     if len(embs_list) >= 500:  # Adjust chunk size based on your RAM
        #         torch.save(
        #             {"ids": self.img_ids[: len(embs_list)], "embs": embs_list},
        #             emb_pth.with_suffix(".tmp"),
        #         )
        #         embs_list.clear()  # Free up memory

        #     # # Find the maximum sequence length
        #     # max_len = max(emb.shape[0] for emb in embs_list)

        #     # # Pad all tensors to the max_len
        #     # padded_embs = [torch.nn.functional.pad(emb, (0, 0, 0, 0, 0, max_len - emb.shape[0])) for emb in embs_list]

        #     # Final save operation
        #     # self.embs = torch.stack(padded_embs)  # Only stack at the end
        # self.embs = torch.stack(embs_list)  # Only stack at the end
        # embs_dict = {
        #     "ids": self.img_ids,
        #     "embs": self.embs,
        # }
        # torch.save(embs_dict, emb_pth)

        # import pdb; pdb.set_trace()
        embs_list = []
        # import pdb; pdb.set_trace()
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

        # find the max time length
        # max_len = max(e.shape[0] for e in embs_list)

        # # pad each to [max_len, D]
        # padded_embs = [
        #     torch.nn.functional.pad(e, (          # pads: (feat_left, feat_right, time_top, time_bottom)
        #         0, 0,           # no pad on feature dim
        #         0, max_len - e.shape[0]  # pad time dim at end
        #     ))
        #     for e in embs_list
        # ]

        # now stack safely
        # self.embs = torch.stack(padded_embs)  # [B, max_len, D]
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

        # import pdb; pdb.set_trace()
        if self.model == "aim":
            # aim_diving48_test_32f.pkl
            # aim_k400_test_32f
            # aim_k400_test_16f.pkl
            # aim_k400_test_8f.pkl
            # PKL_PATH = "/home/animesh/FineGym-CoVR-Github/aim_embeddings_12f_all_files.pkl"  # path to your pickle file
            # PKL_PATH = "/home/animesh/FineGym-CoVR-Github/aim_k400_test_8f.pkl"  # path to your pickle file
            PKL_PATH = (
                "/home/animesh/FineGym-CoVR-Github/aim_k400_train_val_test_8f.pkl"
            )
            # aim_k400_train_val_test_8f.pkl
            self.models_dict = load_models_from_pkl(PKL_PATH)
            print(f"Loaded {len(self.models_dict)} emeddings")
        elif self.model == "clip-mlp":
            print("loading images for clip with sinlge frame")
            PKL_PATH = "/home/animesh/FineGym-CoVR-Github/clip-1_train_val_test_v2.pkl"  # path to your pickle file
            print(f"loading embedding from {PKL_PATH}")
            self.models_dict = load_models_from_pkl(PKL_PATH)
            print(f"Loaded {len(self.models_dict)} models:")
        elif self.model == "clip-mlp-15":
            print("loading images for clip with 15 frame")
            PKL_PATH = "/home/animesh/FineGym-CoVR-Github/clip-15_train_val_test_v2.pkl"  # path to your pickle file
            print(f"loading embedding from {PKL_PATH}")
            self.models_dict = load_models_from_pkl(PKL_PATH)
            print(f"Loaded {len(self.models_dict)} models:")
        elif self.model == "blip2-mlp":
            print("loading images for blip with sinlge frame")
            PKL_PATH = "/home/animesh/FineGym-CoVR-Github/blip2-1_train_val_test_v2.pkl"  # path to your pickle file
            print(f"loading embedding from {PKL_PATH}")
            self.models_dict = load_models_from_pkl(PKL_PATH)
            print(f"Loaded {len(self.models_dict)} models:")
        elif self.model == "blip2-text":
            print("loading images for blip2 for text")
            # PKL_PATH = "/home/animesh/FineGym-CoVR-Github/blip2_tar_train_val_v2.pkl"  # path to your pickle file
            PKL_PATH = "/home/animesh/FineGym-CoVR-Github/blip2-1_train_val_test_v2.pkl"
            print(f"loading embedding from {PKL_PATH}")
            self.models_dict = load_models_from_pkl(PKL_PATH)
            print(f"Loaded {len(self.models_dict)} models:")
        else:
            PKL_PATH = "/home/animesh/FineGym-CoVR-Github/blip2_tar_train_val_v2.pkl"  # path to your pickle file
            # PKL_PATH = "/home/animesh/FineGym-CoVR-Github/blip2-1_train_val_test_v2.pkl"
            print(f"loading embedding from {PKL_PATH}")
            self.models_dict = load_models_from_pkl(PKL_PATH)
            print(f"Loaded {len(self.models_dict)} models:")

        # assert (
        #     len(self.embs) == len(self.img_ids)
        # ), f"Number of embeddings {len(self.embs)} does not match number of images {len(self.img_ids)}"

    def get_target_img_ids(self, index) -> Dict[str, int]:
        """
        Returns the id of the target image and ground truth images for a given query

        Args:
            index (int): id of the query

        Returns:
             Dict[str, int]: dictionary containing target image id and a list of ground truth image ids
        """

        # import pdb; pdb.set_trace()
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
        # import pdb; pdb.set_trace()
        # Get the query id
        query_id = str(self.annotations.loc[index]["query_id"])

        # Get relative caption and shared concept
        # relative_caption = self.annotations[index]["relative_caption"]
        relative_caption = self.annotations.loc[index]["modification"]
        # relative_caption = self.annotations.loc[index]["description"]
        relative_caption = replace_numbers_with_text(relative_caption)
        relative_caption = pre_caption(relative_caption, self.max_words)
        # shared_concept = self.annotations[index]["shared_concept"]
        # shared_concept = pre_caption(shared_concept, self.max_words)

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
        # reference_img_path = self.img_paths[self.img_ids_indexes_map[reference_img_id]]
        # reference_img = Image.open(reference_img_path).convert("RGB")

        if self.model == "clip":
            reference_img = (
                torch.load(reference_img_path, weights_only=True)
                .cpu()
                .to(torch.float32)
            )
            reference_img = reference_img.mean(0)
        elif self.model == "aim":
            # reference_img = torch.load(reference_img_path, weights_only=True).cpu().to(torch.float32)
            # reference_img = reference_img.mean(0)
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
            # reference_img = self.models_dict[str(self.annotations.loc[index]["query_video"])].to(torch.float32)
            # reference_img = self.transform(reference_img)

        if self.split == "test":
            return {
                "reference_img": reference_img,
                "reference_img_id": reference_img_id,
                "relative_caption": relative_caption,
                # "shared_concept": shared_concept,
                "query_id": query_id,
                "ref_filename": str(self.annotations.loc[index]["query_video"]),
            }

        # import pdb; pdb.set_trace()
        # Get the target image and ground truth images
        target_img_path = str(self.annotations.loc[index]["target_video"])
        target_img_id = str(self.img_ids_indexes_map[target_img_path])
        target_img_path = os.path.join(self.emb_dir, str(target_img_path) + ".pth")
        gt_img_ids = [
            str(self.img_ids_indexes_map[x])
            for x in self.annotations.loc[index]["target_videos_ids"]
        ]
        # target_img_path = self.img_paths[self.img_ids_indexes_map[target_img_id]]
        # target_img = Image.open(target_img_path).convert("RGB")
        # target_img = self.frame_loader(target_img_path)
        # target_img = self.transform(target_img)

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
            # target_img = torch.load(target_img_path, weights_only=True).cpu().to(torch.float32)
            target_img = self.models_dict[
                str(self.annotations.loc[index]["target_video"])
            ].to(torch.float32)
            # target_img = target_img.mean(1)
            # target_img = target_img.mean(0)

        # Pad ground truth image IDs with zeros for collate_fn
        gt_img_ids += [""] * (self.max_num_gts - len(gt_img_ids))

        return {
            "reference_img": reference_img,
            "reference_img_id": reference_img_id,
            "target_img": target_img,
            "target_img_id": target_img_id,
            "relative_caption": relative_caption,
            # "shared_concept": shared_concept,
            "gt_img_ids": gt_img_ids,
            "query_id": query_id,
            "ref_filename": str(self.annotations.loc[index]["query_video"]),
        }
