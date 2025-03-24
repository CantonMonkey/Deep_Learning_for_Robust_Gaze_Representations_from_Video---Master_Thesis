import os
import cv2

# List of exact video filenames to process
target_videos = [
    "webcam_c_eyes.mp4",
    "webcam_c_face.mp4",
    "webcam_l_eyes.mp4",
    "webcam_l_face.mp4",
    "webcam_r_eyes.mp4",
    "webcam_r_face.mp4"
]

def extract_and_split_frames(video_path, output_folder, cam_position, process_face=False):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}.")
        return

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Processing {os.path.basename(video_path)}: FPS={video_fps}, Total Frames={total_frames}")

    cam_folder = os.path.join(output_folder, cam_position)
    os.makedirs(cam_folder, exist_ok=True)

    if process_face:
        face_folder = os.path.join(cam_folder, "face")
        os.makedirs(face_folder, exist_ok=True)
    else:
        left_eye_folder = os.path.join(cam_folder, "left_eye")
        right_eye_folder = os.path.join(cam_folder, "right_eye")
        os.makedirs(left_eye_folder, exist_ok=True)
        os.makedirs(right_eye_folder, exist_ok=True)

    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape

        if process_face:
            if w != 256 or h != 256:
                print(f"Skipping frame {frame_idx}: Unexpected face dimensions {w}x{h}")
                continue
            face_filename = os.path.join(face_folder, f"face_{frame_idx:04d}.jpg")
            cv2.imwrite(face_filename, frame)
        else:
            if w != 256 or h != 128:
                print(f"Skipping frame {frame_idx}: Unexpected eye dimensions {w}x{h}")
                continue
            left_eye = frame[:, :128]
            right_eye = frame[:, 128:]
            cv2.imwrite(os.path.join(left_eye_folder, f"left_eye_{frame_idx:04d}.jpg"), left_eye)
            cv2.imwrite(os.path.join(right_eye_folder, f"right_eye_{frame_idx:04d}.jpg"), right_eye)

        frame_idx += 1

    cap.release()
    print(f"Extracted {frame_idx} frames from {os.path.basename(video_path)}")

def process_all_subfolders(base_dir, output_base):
    for subfolder in os.listdir(base_dir):
        subfolder_path = os.path.join(base_dir, subfolder)
        if not os.path.isdir(subfolder_path):
            continue

        print(f"\n==> Entering folder: {subfolder}")

        for video_name in target_videos:
            video_path = os.path.join(subfolder_path, video_name)
            if os.path.isfile(video_path):
                cam_position = video_name.split('_')[1]  # 'r', 'l', or 'c'
                is_face = "face" in video_name
                output_path = os.path.join(output_base, subfolder)
                extract_and_split_frames(video_path, output_path, cam_position, is_face)





# Base directory where the folders are located
base_directory = r"C:\Users\rohan\Desktop\Master\Master Thesis\Master-Thesis\EVE\eve_mini\train01"
# Base output directory for processed frames
output_directory = r"C:\Users\rohan\Desktop\Master\Master Thesis\Master-Thesis\OP"

process_all_subfolders(base_directory, output_directory)
