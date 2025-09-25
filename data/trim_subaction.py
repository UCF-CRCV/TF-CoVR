import os
import os.path as osp
from multiprocessing import Pool, cpu_count
from moviepy.video.io.VideoFileClip import VideoFileClip
import mmengine

# Paths and data setup
data_root = 'data'
anno_root = f'{data_root}/annotations'

event_anno_file = f'{anno_root}/event_annotation.json'
event_root = f'{data_root}/events'
subaction_root = f'{data_root}/subactions'

events = os.listdir(event_root)
events = set(events)
annotation = mmengine.load(event_anno_file)

mmengine.mkdir_or_exist(subaction_root)


def process_subaction(args):
    """
    Function to process a single subaction using MoviePy.
    Args:
        args (tuple): Contains video_path, start_time, end_time, output_file, and subaction_name.
    """
    video_path, start_time, end_time, output_filename, subaction_name, k = args

    if osp.exists(output_filename):
        print(f"Subaction {subaction_name} already exists, skipping...")
        return

    try:
        # Load video clip and trim
        with VideoFileClip(video_path) as clip:
            trimmed_clip = clip.subclipped(start_time, end_time)
            trimmed_clip.write_videofile(output_filename, audio=False, preset='ultrafast', threads=8, logger="bar")
        
        print(f"Successfully processed Subaction {subaction_name} of Event {k}")
    except Exception as e:
        print(f"Trimming of the Subaction {subaction_name} of Event {k} Failed: {e}", flush=True)


def main():
    # Create a list of arguments to process
    tasks = []

    for k, v in annotation.items():
        if k + '.mp4' not in events:
            print(f'Video {k[:11]} has not been downloaded '
                  f'or the event clip {k} not generated')
            continue

        video_path = osp.join(event_root, k + '.mp4')

        for subaction_id, subaction_anno in v.items():
            timestamps = subaction_anno['timestamps']
            start_time, end_time = timestamps[0][0], timestamps[-1][1]
            subaction_name = k + '_' + subaction_id

            output_filename = osp.join(subaction_root, subaction_name + '.mp4')

            tasks.append((video_path, start_time, end_time, output_filename, subaction_name, k))

    # Use multiprocessing to parallelize the processing
    num_processes = min(cpu_count(), 8)  # Use at most 8 processes
    print(f"Starting processing with {num_processes} processes...")

    with Pool(processes=num_processes) as pool:
        pool.map(process_subaction, tasks)

    print("All subactions have been processed.")


if __name__ == "__main__":
    main()
