import os
import cv2
import ssl
import shutil
import tempfile
from irods.session import iRODSSession


env_file = os.path.expanduser("~/.irods/irods_environment.json")
ssl_context = ssl.create_default_context()
session = iRODSSession(irods_env_file=env_file, ssl_context=ssl_context)


target_videos = [
    "webcam_c_eyes.mp4",
    "webcam_c_face.mp4",
    "webcam_l_eyes.mp4",
    "webcam_l_face.mp4",
    "webcam_r_eyes.mp4",
    "webcam_r_face.mp4"
]




def download_video_to_temp(remote_path):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    with session.data_objects.get(remote_path).open('r') as f:
        shutil.copyfileobj(f, tmp)
    tmp.close()
    return tmp.name

def extract_and_split_frames(video_path, output_folder, cam_position, process_face=False):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}.")
        return

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
   

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
    


def process_all_from_mango(mango_base_path, output_base):
    coll = session.collections.get(mango_base_path)
    for subcoll in coll.subcollections:
        print(f"\n==> Entering folder: {subcoll.name}")

        for video_name in target_videos:
            remote_video_path = f"{subcoll.path}/{video_name}"
            try:
                local_temp_video = download_video_to_temp(remote_video_path)
            except Exception as e:
                print(f"⚠ Could not load {remote_video_path}: {e}")
                continue

            cam_position = video_name.split('_')[1]
            is_face = "face" in video_name
            output_path = os.path.join(output_base, subcoll.name)
            extract_and_split_frames(local_temp_video, output_path, cam_position, is_face)
            os.remove(local_temp_video)

        for cam_position in ['r', 'l', 'c']:
            h5_filename = f"webcam_{cam_position}.h5"
            remote_h5_path = f"{subcoll.path}/{h5_filename}"
            h5_output_folder = os.path.join(output_base, subcoll.name, cam_position)
            os.makedirs(h5_output_folder, exist_ok=True)
            local_h5_target = os.path.join(h5_output_folder, h5_filename)

            try:
                with session.data_objects.get(remote_h5_path).open('r') as remote_h5:
                    with open(local_h5_target, 'wb') as out_file:
                        shutil.copyfileobj(remote_h5, out_file)
                print(f"Copied {h5_filename} → {h5_output_folder}")
            except Exception as e:
                print(f"{h5_filename} not found in {subcoll.name} — skipping.")

mango_base_collection = "/set/home/ciis-lab/eve_dataset/train01"

base_directory = "/data/leuven/374/vsc37437/mango_to_vsc_test/OP"

process_all_from_mango(mango_base_collection, base_directory)
session.cleanup()
