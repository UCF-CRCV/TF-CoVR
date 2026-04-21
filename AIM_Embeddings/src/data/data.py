import ast
import random
from pathlib import Path

import pandas as pd
import torch
from lightning import LightningDataModule
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
from collections import Counter
import pickle
import re


from src.data.transforms import transform_test, transform_train
from src.data.utils import FrameLoader

Image.MAX_IMAGE_PIXELS = None  # Disable DecompressionBombWarning

class TFCoVRDataModule(LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        val_batch_size: int,
        num_workers: int = 4,
        pin_memory: bool = True,
        annotation: dict = {"train": "", "val": ""},
        vid_dirs: dict = {"train": "", "val": ""},
        emb_dirs: dict = {"train": "", "val": ""},
        image_size: int = 384,
        emb_pool: str = "query",
        iterate: str = "target_video",
        vid_query_method: str = "middle",
        vid_frames: int = 1,
        max_frames: int = 32,
        skiprate: int=3,
        image_resolution: int = 364,
        n_embs: int = 15,
        si_tc_weight=0,
        **kwargs,  # type: ignore
    ) -> None:
        super().__init__()
        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.emb_pool = emb_pool
        self.iterate = iterate
        self.vid_query_method = vid_query_method
        self.vid_frames = vid_frames
        self.max_frames = max_frames
        self.skiprate = skiprate
        self.image_resolution = image_resolution

        self.transform_train = transform_train(224)
        self.transform_test = transform_test(224)

        self.data_train = FineGymCoVRDataset(
            transform=self.transform_train,
            annotation=annotation["train"],
            vid_dir=vid_dirs["train"],
            emb_dir=emb_dirs["train"],
            split="train",
            emb_pool=self.emb_pool,
            iterate=self.iterate,
            vid_query_method=self.vid_query_method,
            vid_frames=self.vid_frames,
            max_frames=self.max_frames,
            image_resolution=self.image_resolution,
            skiprate=self.skiprate,
            n_embs=n_embs,
            si_tc_weight=si_tc_weight,
        )
        self.data_val = FineGymCoVRDataset(
            transform=self.transform_test,
            # transform=None,
            annotation=annotation["val"],
            vid_dir=vid_dirs["val"],
            emb_dir=emb_dirs["val"],
            split="val",
            emb_pool=self.emb_pool,
            iterate=self.iterate,
            vid_query_method=self.vid_query_method,
            vid_frames=self.vid_frames,
            max_frames=self.max_frames,
            image_resolution=self.image_resolution,
            skiprate=self.skiprate,
            n_embs=n_embs,
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
            pin_memory=self.pin_memory,
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
            drop_last=False,
        )


class TFCoVRTestDataModule(LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        val_batch_size: int,
        annotation: str,
        vid_dirs: str,
        emb_dirs: str,
        num_workers: int = 4,
        pin_memory: bool = True,
        image_size: int = 384,
        emb_pool: str = "query",
        n_embs: int = 15,
        iterate: str = "target_video",
        vid_query_method: str = "middle",
        vid_frames: int = 1,
        max_frames: int = 32,
        skiprate: int=3,
        image_resolution: int = 364,
        **kwargs,  # type: ignore
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.emb_pool = emb_pool
        self.iterate = iterate
        self.vid_query_method = vid_query_method
        self.vid_frames = vid_frames
        self.max_frames = max_frames
        self.skiprate = skiprate
        self.image_resolution = image_resolution

        self.transform_test = transform_test(image_size)

        self.data_test = FineGymCoVRDataset(
            transform=self.transform_test,
            annotation=annotation,
            vid_dir=vid_dirs,
            emb_dir=emb_dirs,
            split="test",
            emb_pool=self.emb_pool,
            n_embs=n_embs,
            iterate=self.iterate,
            vid_query_method=self.vid_query_method,
            vid_frames=self.vid_frames,
            max_frames=self.max_frames,
            image_resolution=self.image_resolution,
            skiprate=self.skiprate,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
            drop_last=False,
        )

label2index = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, '10': 10, '11': 11, '12': 12, '13': 13, '14': 14, '15': 15, '16': 16, '17': 17, '18': 18, '19': 19, '20': 20, '21': 21, '22': 22, '23': 23, '25': 24, '27': 25, '28': 26, '29': 27, '30': 28, '31': 29, '32': 30, '33': 31, '34': 32, '35': 33, '36': 34, '37': 35, '38': 36, '40': 37, '41': 38, '42':39, '43': 40, '44': 41, '45': 42, '46': 43, '47': 44, '48': 45, '49': 46, '51': 47, '52': 48, '53': 49, '54': 50, '55': 51, '56': 52, '57': 53, '58': 54, '59': 55, '60': 56, '61': 57, '62': 58, '63': 59, '65': 60, '66': 61, '67': 62, '68': 63, '69': 64, '70': 65, '71': 66, '73': 67, '76': 68, '77': 69, '78': 70, '81': 71, '82': 72, '83': 73, '84': 74, '85': 75, '86': 76, '88': 77, '89':78, '90': 79, '91': 80, '92': 81, '93': 82, '94': 83, '95': 84, '97': 85, '98': 86, '99': 87, '100': 88, '101': 89, '102': 90, '103': 91, '104': 92, '105': 93, '106': 94, '107': 95, '108': 96, '109': 97, '110': 98, '111': 99, '112': 100, '113': 101, '114': 102, '115': 103, '116': 104, '117': 105, '118': 106, '119': 107, '120': 108, '121': 109, '122': 110, '123': 111, '124': 112, '125':113, '126': 114, '127': 115, '128': 116, '129': 117, '133': 118, '134': 119, '135': 120, '136': 121, '137': 122, '138': 123, '139': 124, '140': 125, '141': 126, '142': 127, '143': 128, '144': 129, '145': 130, '146': 131, '147': 132, '148': 133, '150': 134, '151': 135, '152': 136, '153': 137, '154': 138, '155': 139, '156': 140, '158': 141, '159': 142, '160': 143, '161': 144, '162': 145,'163': 146, '165': 147, '166': 148, '167': 149, '168': 150, '169': 151, '170': 152, '171': 153, '172': 154, '173': 155, '174': 156, '175': 157, '176': 158, '177': 159, '178': 160, '179': 161, '180': 162, '181': 163, '182': 164, '183': 165, '184': 166, '185': 167, '186': 168, '187': 169, '188': 170, '189': 171, '191': 172, '192': 173, '193': 174, '194': 175, '195': 176, '196': 177, '197':178, '198': 179, '207': 180, '208': 181, '209': 182, '210': 183, '211': 184, '212': 185, '213': 186, '214': 187, '215': 188, '216': 189, '217': 190, '218': 191, '219': 192, '220': 193, '221': 194, '222': 195, '223': 196, '224': 197, '225': 198, '226': 199, '227': 200, '228': 201, '229': 202, '230': 203, '231': 204, '232': 205, '233': 206, '234': 207, '235': 208, '236': 209, '237': 210,'238': 211, '239': 212, '240': 213, '241': 214, '242': 215, '243': 216, '244': 217, '245': 218, '246': 219, '247': 220, '248': 221, '249': 222, '250': 223, '251': 224, '252': 225, '253': 226, '254': 227, '255': 228, '256': 229, '257': 230, '258': 231, '259': 232, '260': 233, '262': 234, '263': 235, '264': 236, '266': 237, '267': 238, '268': 239, '269': 240, '270': 241, '271': 242, '272':243, '273': 244, '274': 245, '275': 246, '276': 247, '277': 248, '278': 249, '279': 250, '280': 251, '281': 252, '282': 253, '283': 254, '284': 255, '285': 256, '286': 257, '287': 258, '407c': 259, '5253b': 260, '107b': 261, '6245d': 262, '207c': 263, '5152b': 264, '5255b': 265, '6243d': 266, '109c': 267, '626c': 268, '307c': 269, '207b': 270, '5156b': 271, '407b': 272, '409c': 273,'6142d': 274, '305c': 275, '405b': 276, '205b': 277, '5235d': 278, '612b': 279, '103b': 280, '403b': 281, '101b': 282, '5331d': 283, '5132d': 284, '614b': 285, '5231d': 286, '5154b': 287, '5251b': 288, '107c': 289, '105b': 290, '6241b': 291, '5237d': 292, '5353b': 293, '5337d': 294, '5355b': 295, '405c': 296, '5335d': 297, '5172b': 298, '636c': 299, '205c': 300, '626b': 301, '401b': 302, '109b':303, '303c': 304, '5233d': 305}
    

class TFCoVRCoVRDataset(Dataset):
    def __init__(
        self,
        transform,
        annotation: str,
        vid_dir: str,
        emb_dir: str,
        split: str,
        max_words: int = 30,
        max_frames: int = 16,
        # frame_resolution: int = 224,
        skiprate: int=3,
        frame_order: int = 0,
        feature_framerate: int = 1,
        image_resolution: int = 364,
        emb_pool: str = "query",
        n_embs: int = 15,
        iterate: str = "target_video",
        vid_query_method: str = "middle",
        vid_frames: int = 1,
        si_tc_weight=0,
    ) -> None:
        super().__init__()

        self.transform = transform
        self.max_frames = max_frames
        self.skiprate = skiprate
        print(f"Running with {self.max_frames} frames and skiprate {self.skiprate}")
        self.FrameLoader = FrameLoader(transform=self.transform, frames_video=self.max_frames, method="sample")
        self.frame_order = frame_order
        self.split = split

        self.anns_file = annotation
        self.df = pd.read_csv(self.anns_file)

        # Original data
        self.train_files = self.df["video_file"].tolist()
        self.train_all_labels = self.df["label"].tolist()
        self.train_labels = self.df["label"].unique().tolist()
        self.label2index = label2index
        print(len({lbl:idx for idx, lbl in enumerate(self.train_labels)}))
        print(len(self.label2index))
        
        # Print labels not present in label2index
        labels_not_in_index = set(self.train_all_labels) - set(self.label2index.keys())
        if labels_not_in_index:
            print(f"Labels not found in label2index: {sorted(labels_not_in_index)}")
        
        # Filter out examples whose label is not in label2index
        filtered = [
            (vid, lbl) for vid, lbl in zip(self.train_files, self.train_all_labels)
            if lbl in self.label2index
        ]
        
        # Unzip filtered list back into separate lists
        self.train_files, self.train_all_labels = zip(*filtered) if filtered else ([], [])
        
        # Convert to lists
        self.train_files = list(self.train_files)
        self.train_all_labels = list(self.train_all_labels)
        
        # Sanity check
        assert len(self.train_files) == len(self.train_all_labels), \
            "len of train video ids and labels should be equal"

        
        # Update video2label mapping
        self.video2label = {vid: lbl for vid, lbl in zip(self.train_files, self.train_all_labels)}

        
        # Set video path
        self.video_path = vid_dir

        # Print dataset information
        print(f"Total {'Training' if split == 'train' else 'Testing'} Sample: {len(self.train_files)}")

    def __getitem__(self, item):

        if self.split == "train":
            video_id = self.train_files[item]
            video_path = os.path.join(self.video_path, video_id + ".mp4")
            label = self.video2label[video_id]
            label_idx = torch.tensor(self.label2index[label])

            video_raw = self.FrameLoader(video_path)

            sample = {
                "video": video_raw,
                "label": label_idx,
            }
            return sample

        else:

            video_id = self.train_files[item]
            video_path = os.path.join(self.video_path, video_id + ".mp4")
            label = self.video2label[video_id]
            label_idx = torch.tensor(self.label2index[label])

            video_raw = self.FrameLoader(video_path)

            sample = {
                "video": video_raw,
                "label": label_idx,
            }
            return sample

    def __len__(self):
        if self.split == "train":
            return len(self.train_files)
        elif self.split == "val":
            return len(self.train_files)


def read_pkl(data_url):
    file = open(data_url,'rb')
    content = pickle.load(file)
    file.close()
    return content

        