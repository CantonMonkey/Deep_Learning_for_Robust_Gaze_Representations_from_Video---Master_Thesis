import os
import h5py
import torch
import numpy as np

def spherical_to_cartesian(theta_phi):
    theta = theta_phi[:, 0]
    phi = theta_phi[:, 1]
    if torch.isnan(theta_phi).any():
        print("NaN detected in spherical_to_cartesian")
    x = torch.sin(theta) * torch.cos(phi)
    y = torch.sin(theta) * torch.sin(phi)
    z = torch.cos(theta)
    return torch.stack([x, y, z], dim=1)

# 模拟你的 class 初始化的一些目录结构变量
root_dir = "D:/thesis_code/OP"  # 顶层数据路径
step_folders = os.listdir(root_dir)  # 每一轮 step，例如 step1, step2 ...
camera_dirs = ['l', 'r', 'c']  # 相机目录

all_labels = []

for step_folder in step_folders:
    for camera_dir in camera_dirs:
        camera_path = os.path.join(root_dir, step_folder, camera_dir)
        
        if os.path.exists(camera_path) and all(
            os.path.exists(os.path.join(camera_path, folder)) 
            for folder in ['left_eye', 'right_eye', 'face']
        ):
            # 只读取label（.h5）
            label_files = [f for f in os.listdir(camera_path) if f.endswith('.h5')]
            if label_files:
                label_path_h5 = os.path.join(camera_path, label_files[0])  # 只取第一个
                print(f"Reading label from: {label_path_h5}")
                
                with h5py.File(label_path_h5, 'r') as f:
                    if 'face_g_tobii/data' in f:
                        h5_labels = f['face_g_tobii/data'][:]
                        all_labels.append(h5_labels)
                    else:
                        print(f"Label key 'face_g_tobii/data' not found in {label_path_h5}")

if all_labels:
    labels_array = np.vstack(all_labels)
    labels_tensor = torch.tensor(labels_array, dtype=torch.float32)

    if labels_tensor.max() > 3.2:  # 度转弧度
        labels_tensor = labels_tensor * (np.pi / 180.0)

    cartesian_labels = spherical_to_cartesian(labels_tensor)
    print("Cartesian shape:", cartesian_labels.shape)
    print("First few vectors:\n", cartesian_labels[:5])
else:
    print("No labels found.")
