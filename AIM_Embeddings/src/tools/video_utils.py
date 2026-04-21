import random
import numpy as np
import cv2
from decord import VideoReader, cpu
import torch

from torchvision import transforms
from torchvision.transforms._transforms_video import NormalizeVideo


def sample_frames(num_frames, vlen, sample="rand", fix_start=None):
    acc_samples = min(num_frames, vlen)
    intervals = np.linspace(start=0, stop=vlen, num=acc_samples + 1).astype(int)
    ranges = []
    for idx, interv in enumerate(intervals[:-1]):
        ranges.append((interv, intervals[idx + 1] - 1))
    if sample == "rand":
        frame_idxs = [random.choice(range(x[0], x[1])) for x in ranges]
    elif fix_start is not None:
        frame_idxs = [x[0] + fix_start for x in ranges]
    elif sample == "uniform":
        frame_idxs = [(x[0] + x[1]) // 2 for x in ranges]
    elif sample == "middle_repeat":
        frame_idxs = [vlen // 2] * num_frames
    else:
        raise NotImplementedError

    return frame_idxs


def uniform_sample(lst, n):
    assert n <= len(lst)
    m = len(lst)
    step = m // n  # Calculate the step size
    return [lst[i * step] for i in range(n)]

def read_frames_cv2(video_path, max_frames=16, video_framerate=1, num_video_frames=16, frame_resolution=224, fix_start=None, end_time=None):

    # Load video with decord
    vr = VideoReader(video_path, num_threads=1, width=frame_resolution, height=frame_resolution)

    fps = vr.get_avg_fps()
    total_frames = len(vr)

    # Compute frame indices based on start & end time
    if end_time is not None:
        start_frame = int(fps * start_time)
        end_frame = min(int(fps * end_time), total_frames)  # Clip to avoid overflow
    else:
        start_frame = 0
        end_frame = total_frames

    vlen = end_frame - start_frame  # Number of frames in range

    # Sample frame indices
    frame_idxs = sample_frames(self.frames_video, vlen)
    frame_idxs = [idx + start_frame for idx in frame_idxs]

    # Handle case where video has fewer frames than expected
    if len(frame_idxs) < self.frames_video:
        frame_idxs = (frame_idxs * self.frames_video)[: self.frames_video]
        print(f"Video {video_pth} has less than {self.frames_video} frames")

    # Read frames efficiently using decord
    frames = vr.get_batch(frame_idxs).numpy()  # Shape: (num_frames, H, W, C)

    # Convert to PIL images and apply transformation
    frames_pil = [Image.fromarray(frame) for frame in frames]
    video_data = [self.transform(frame) for frame in frames_pil]

    if len(video_data) > 0:
        return video_data
    else:
        raise ValueError(f"video path: {video_pth} error.")

    return video_frames, None


class FrameLoader:
    def __init__(self, max_frames=16, video_framerate=1, num_video_frames=16, frame_resolution=224, fix_start=None, end_time=None):
        self.max_frames = max_frames
        # self.method = method
        self.fix_start = fix_start
        self.video_framerate = video_framerate
        self.num_video_frames = num_video_frames
        self.frame_resolution = frame_resolution
        # self.sample = sample
        self.end_time = end_time

        normalize = NormalizeVideo(
            mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
        )
        self.transforms = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(256),
                transforms.Resize(224),
                normalize,
            ]
        )

    def __call__(self, video_path):
        frames, success_idxs = read_frames_cv2(
            video_path, max_frames=self.max_frames, video_framerate=self.video_framerate, num_video_frames=self.num_video_frames, frame_resolution=self.frame_resolution, fix_start=None, end_time=None
        )

        if self.max_frames > 1:
            frames = frames.transpose(0, 1)  # [T, C, H, W] ---> [C, T, H, W]
            frames = self.transforms(frames)
            # frames = frames.transpose(0, 1)  # recover
        else:
            frames = self.transforms(frames)

        return frames
