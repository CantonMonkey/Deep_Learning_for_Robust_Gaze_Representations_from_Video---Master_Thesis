import torch
import torchaudio
from Integrated_model import WholeModel
# import torch.utils.data as data
import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn



# def dataLoader():
#     # load data from dataset remotely
#     vsc_dir = ""
#     ## load from remote VSC URL
#     ###########

#     # pytorch dataloader for data loading
#     loaded = data.DataLoader(
#         dataset,  # the loaded dataset
#         batch_size=1, 
#         shuffle=False, 
#         sampler=None,
#         batch_sampler=None, 
#         num_workers=0, 
#         collate_fn=None,
#         pin_memory=False, 
#         drop_last=False, 
#         timeout=0,
#         worker_init_fn=None, 
#         *, # why?
#         prefetch_factor=2,
#         persistent_workers=False)
    
#     return loaded

class GazeDatasetFromPaths(Dataset):
    def __init__(self, folder_path, label_path):
        self.folder_path = folder_path
        self.labels = pd.read_csv(label_path, header=None).values.astype('float32') # converted to numpy array
        self.left_eye_files = sorted(os.listdir(os.path.join(folder_path, "left_eye")))
        self.right_eye_files = sorted(os.listdir(os.path.join(folder_path, "right_eye")))
        self.face_files = sorted(os.listdir(os.path.join(folder_path, "face")))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        left_path = os.path.join(self.folder_path, "left_eye", self.left_eye_files[idx])
        right_path = os.path.join(self.folder_path, "right_eye", self.right_eye_files[idx])
        face_path = os.path.join(self.folder_path, "face", self.face_files[idx])
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        print("--------------------------qwe-----------")
        print(label.shape)
        print("--------------------------qwe-----------")
        return left_path, right_path, face_path, label

def dot_product_loss(pred, target):

    pred = nn.functional.normalize(pred, p=2, dim=1) # Resulting vector will have the correct direction but unit vector
    target = nn.functional.normalize(target, p=2, dim=1)
    print(pred)
    print(target)
    return torch.sum(pred * target, dim=1).mean()

def angular_error(pred, target):
    pred = nn.functional.normalize(pred, p=2, dim=1)
    target = nn.functional.normalize(target, p=2, dim=1)
    cos_sim = torch.sum(pred * target, dim=1)
    return torch.acos(cos_sim) * (180.0 / torch.pi)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = WholeModel().to(device)

dataset_path = "C:/Users/rohan/Desktop/Master/Master Thesis/Master-Thesis/Dataset-Test/Output Folder/webcam_r"
label_excel = "C:/Users/rohan/Desktop/Master/Master Thesis/Master-Thesis/Dataset-Test/Output Folder/data.csv"

dataset = GazeDatasetFromPaths(dataset_path, label_excel)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True) # shuffle true? yes, cause label is included so no issue

def train():
    model = WholeModel()

    bs = 16 # batch size
    # hidden_size =
    # sequence_length =
    # num_classes =

    Leye = torch.rand(bs,3, 16, 16)
    Reye = torch.rand(bs,3, 16, 16)
    FaceData = torch.rand(bs,3, 16, 16)  # currently matched with eyes......

    out = model(Leye, Reye, FaceData)
    print(out.shape)





if __name__ == "__main__":
    train()



