# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
import subprocess

import mmengine

data_root = 'data'
video_root = 'data/finegym'
anno_root = f'{data_root}/annotations'
anno_file = 'finegym_annotation_info_v1.1.json'

event_anno_file = f'{anno_root}/event_annotation.json'
event_root = f'{data_root}/events'

videos = os.listdir(video_root)
videos = set(videos)
annotation = mmengine.load(anno_file)
event_annotation = {}

mmengine.mkdir_or_exist(event_root)

def find_video_file(video_folder, video_id):
    """Find the video file by its ID and automatically detect its extension."""
    for ext in ['.mp4', '.avi', '.mkv', '.mov', '.webm']:  # Add more extensions if needed
        video_path = os.path.join(video_folder, f"{video_id}{ext}")
        if os.path.exists(video_path):
            return video_path
    return None

for k, v in annotation.items():

    video_path = find_video_file(video_root, k)

    if video_path is None:
        print(f'video {k} has not been downloaded')
        continue
        
    if video_path.split("/")[-1] not in videos:
        print(f'video {k} has not been downloaded')
        continue

    for event_id, event_anno in v.items():
        timestamps = event_anno['timestamps'][0]
        start_time, end_time = timestamps
        event_name = k + '_' + event_id

        output_filename = event_name + '.mp4'

        command = [
            'ffmpeg', '-i',
            '"%s"' % video_path, '-ss',
            str(start_time), '-t',
            str(end_time - start_time), '-c:a', 'copy',
            '-threads', '8', '-loglevel', 'panic',
            '"%s"' % osp.join(event_root, output_filename)
        ]

        command = ' '.join(command)
        try:
            subprocess.check_output(
                command, shell=True, stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError:
            print(
                f'Trimming of the Event {event_name} of Video {k} Failed',
                flush=True)

        segments = event_anno['segments']
        if segments is not None:
            event_annotation[event_name] = segments

mmengine.dump(event_annotation, event_anno_file)
