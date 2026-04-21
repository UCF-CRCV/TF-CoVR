import os
from glob import glob
from pathlib import Path

import hydra
import lightning as L
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from tqdm import tqdm

from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

from decord import VideoReader
from PIL import Image
import numpy as np

normalize = transforms.Normalize(
    (0.48145466, 0.4578275, 0.40821073),
    (0.26862954, 0.26130258, 0.27577711),
)


class transform_test:
    def __init__(self, image_size=384):
        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    (image_size, image_size),
                    interpolation=InterpolationMode.BICUBIC,
                ),
                transforms.ToTensor(),
                normalize,
            ]
        )

    def __call__(self, img):
        return self.transform(img)


# -------------------------
# Frame sampling + loader
# -------------------------
def sample_frames(num_frames, vlen):
    acc_samples = min(num_frames, vlen)
    intervals = np.linspace(0, vlen, num=acc_samples + 1).astype(int)

    frame_idxs = [(intervals[i] + intervals[i + 1] - 1) // 2 for i in range(acc_samples)]
    return frame_idxs


class FrameLoader:
    def __init__(self, transform, frames_video=8):
        self.transform = transform
        self.frames_video = frames_video

    def __call__(self, video_pth: str):
        vr = VideoReader(video_pth, num_threads=1)

        total_frames = len(vr)
        frame_idxs = sample_frames(self.frames_video, total_frames)

        # Handle short videos
        if len(frame_idxs) < self.frames_video:
            frame_idxs = (frame_idxs * self.frames_video)[: self.frames_video]
            print(f"[WARN] {video_pth} has fewer frames than expected")

        # frames = vr.get_batch(frame_idxs).asnumpy()
        frames = vr.get_batch(frame_idxs)

        # handle both cases (decord ndarray OR torch tensor)
        if hasattr(frames, "asnumpy"):
            frames = frames.asnumpy()
        elif isinstance(frames, torch.Tensor):
            frames = frames.cpu().numpy()
        else:
            frames = np.array(frames)

        frames = [Image.fromarray(f) for f in frames]
        frames = [self.transform(f) for f in frames]

        return torch.stack(frames)  # [T, C, H, W]


# -------------------------
# Hook
# -------------------------
def get_avg_pool_output_hook(module, input, output):
    module._output = output.detach()


# -------------------------
# Main
# -------------------------
@hydra.main(version_base=None, config_path="configs", config_name="train")
def main(cfg: DictConfig):

    video_glob = "path-to-all-mp4-videos"
    save_path = "path-to-save-videos"
    os.makedirs(save_path, exist_ok=True)

    ckpt_path = getattr(cfg, "extract_ckpt", None)

    L.seed_everything(cfg.seed, workers=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # -------------------------
    # Model
    # -------------------------
    model = instantiate(cfg.model)

    if not hasattr(model.model, "avg_pool"):
        raise AttributeError("model.model.avg_pool not found")

    hook = model.model.avg_pool.register_forward_hook(get_avg_pool_output_hook)

    # -------------------------
    # Load checkpoint
    # -------------------------
    if ckpt_path:
        print(f"Loading checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu")

        if "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        elif "model" in ckpt:
            state_dict = ckpt["model"]
        else:
            state_dict = ckpt

        model.load_state_dict(state_dict, strict=False)

    model = model.to(device)
    model.eval()

    # -------------------------
    # Data
    # -------------------------
    all_files = sorted(glob(video_glob))
    print(f"Found {len(all_files)} videos")

    transform_fn = transform_test(224)
    frameloader = FrameLoader(transform_fn, frames_video=12)

    # -------------------------
    # Extraction
    # -------------------------
    with torch.no_grad():
        for file_path in tqdm(all_files):

            try:
                vid = frameloader(file_path)       # [T, C, H, W]
                vid = vid.unsqueeze(0).to(device)  # [1, T, C, H, W]

                _ = model.model(vid)

                feat = model.model.avg_pool._output.squeeze().cpu()

                filename = Path(file_path).stem
                save_file = os.path.join(save_path, filename + ".pth")

                torch.save(feat, save_file)

            except Exception as e:
                print(f"[ERROR] {file_path}: {e}")

    hook.remove()
    print("Done!")


if __name__ == "__main__":
    main()