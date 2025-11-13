import cv2
import os
from natsort import natsorted

def create_video_from_images(image_folder, files, fps=30):
    # Get all image files from the folder
    images = natsorted(files)  # Sort images naturally (handles numbers correctly)

    dest_folder = "dest_folder_path"
    os.makedirs(dest_folder, exist_ok=True)

    if not images:
        print("No images found in the folder.")
        return

    # Read the first image to get dimensions
    first_image_path = os.path.join(image_folder, images[0])
    frame = cv2.imread(first_image_path)
    height, width, _ = frame.shape

    filename = "_".join(image_folder.split("/")[-2:]) + ".mp4"
    output_video = os.path.join(dest_folder, filename)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4 format
    video_writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    for image in images:
        image_path = os.path.join(image_folder, image)
        frame = cv2.imread(image_path)

        if frame is None:
            print(f"Skipping {image_path}, unable to read image.")
            continue

        video_writer.write(frame)

    video_writer.release()
    print(f"Video saved as {output_video}")

# Example usage
for folder, subfolders, files in os.walk("FINADiving_MTL_256s_folder_path"):
    if files:
        create_video_from_images(folder, files)
