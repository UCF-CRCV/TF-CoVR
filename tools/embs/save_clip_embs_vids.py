import os
import sys
from pathlib import Path

import torch
from torchvision import transforms
from tqdm.auto import tqdm
from PIL import Image

script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
sys.path.append(project_root)

import clip
from src.data.embs import VideoDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def main(args):
    args.save_dir = args.save_dir.expanduser()
    args.video_dir = args.video_dir.expanduser()

    save_dir = args.save_dir / f"clip-rn50-embs-all-{args.frames_video}"
    save_dir.mkdir(parents=True, exist_ok=True)

    print("Loading CLIP model...")
    model, preprocess = clip.load("RN50", device=device)
    model.eval()

    # Overriding preprocess to work with variable image size
    preprocess = transforms.Compose(
        [
            transforms.Resize(
                (args.image_size, args.image_size),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.CenterCrop(args.image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711],
            ),
        ]
    )

    dataset = VideoDataset(
        video_dir=args.video_dir,
        todo_ids=args.todo_ids,
        num_shards=args.num_shards,
        shard_id=args.shard_id,
        frames_video=args.frames_video,
        save_dir=save_dir,
        image_size=args.image_size,
    )
    dataset.transform = preprocess
    dataset.pixel_size = args.image_size

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=args.num_workers,
    )

    for video_ids, f_idxs, frames in tqdm(loader):
        frames = frames.to(device)  # [B, T, C, H, W]
        bs, nf, c, h, w = frames.shape
        frames = frames.view(bs * nf, c, h, w)

        image_features = model.encode_image(frames)  # [bs * nf, 640]
        image_features = image_features.view(bs, nf, 1024).cpu()  # [bs, nf, 640]

        for video_id, f_idx, frm_feat in zip(video_ids, f_idxs, image_features):
            frm_feat = frm_feat[f_idx > -1]
            f_idx = f_idx[f_idx > -1]
            if len(f_idx) == 0:
                continue
            save_pth = save_dir / f"{video_id}.pth"
            if save_pth.exists():
                continue
            save_pth.parent.mkdir(exist_ok=True)
            torch.save(frm_feat, save_pth)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video_dir", type=Path, required=True, help="Path to video directory"
    )
    parser.add_argument(
        "--save_dir",
        type=Path,
        default="PATH_TO_SAVE_DIR",
        help="Path to save directory",
    )
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--frames_video", type=int, default=15)
    parser.add_argument("--todo_ids", type=str, default=None)
    parser.add_argument(
        "--image_size", type=int, default=224, choices=[224, 364]
    )  # CLIP default is 224
    args = parser.parse_args()

    main(args)
