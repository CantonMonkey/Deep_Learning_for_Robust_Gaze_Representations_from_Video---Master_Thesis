import torch
import torchaudio
from Integrated_model import WholeModel
# import torch.utils.data as data
import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import wandb # https://docs.wandb.ai/tutorials/experiments/
import torchvision.transforms as transforms
import random
import math
from PIL import Image







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



#substitute of MNIST dataset
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

        # https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
        '''transform should be applied here, not in the model'''
        # read imgs
        left_img = Image.open(left_path).convert("RGB")
        right_img = Image.open(right_path).convert("RGB")
        face_img = Image.open(face_path).convert("RGB")

        # transform the images to tensors
        if self.transform:
            left_img = self.transform(left_img)
            right_img = self.transform(right_img)
            face_img = self.transform(face_img)

        # return tensors of the images and the label, instead of the path
        return left_img, right_img, face_img, label
    pass
    

def get_dataloader(folder_path, label_path, batch_size, shuffle=True):
    # dataset = GazeDatasetFromPaths(dataset_path, label_excel)
    # dataloader = DataLoader(dataset, batch_size=1, shuffle=True) # shuffle true? yes, cause label is included so no issue
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = GazeDatasetFromPaths(folder_path, label_path, transform=transform)
    
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True,
        num_workers=2,
    )

    return loader


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
model = WholeModel().to(device)  # Load the model

# /data/leuven/374/vsc37415/OP/
dataset_path = "C:/Users/rohan/Desktop/Master/Master Thesis/Master-Thesis/Dataset-Test/Output Folder/webcam_r"
label_excel = "C:/Users/rohan/Desktop/Master/Master Thesis/Master-Thesis/Dataset-Test/Output Folder/data.csv"

dataset = GazeDatasetFromPaths(dataset_path, label_excel)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True) # shuffle true? yes, cause label is included so no issue


def validate_model(model, valid_dl, loss_func, log_images=False, batch_idx=0):
    "Compute performance of the model on the validation dataset and log a wandb.Table"
    model.eval()
    val_loss = 0.0
    with torch.inference_mode():
        correct = 0
        for i, (images, labels) in enumerate(valid_dl):
            images, labels = images.to(device), labels.to(device)

            # Forward pass ➡
            outputs = model(images)
            val_loss += loss_func(outputs, labels) * labels.size(0)

            # Compute accuracy and accumulate
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()

            # Log one batch of images to the dashboard, always same batch_idx.
            if i == batch_idx and log_images:
                log_image_table(images, predicted, labels, outputs.softmax(dim=1))
    return val_loss / len(valid_dl.dataset), correct / len(valid_dl.dataset)



# Create a teble to compare the predicted values versus the true value
def log_image_table(images, predicted, labels, probs):
    "Log a wandb.Table with (img, pred, target, scores)"
    # Create a wandb Table to log images, labels and predictions to
    table = wandb.Table(
        columns=["image", "pred", "target"] + [f"score_{i}" for i in range(10)]
    )
    for img, pred, targ, prob in zip(
        images.to("cpu"), predicted.to("cpu"), labels.to("cpu"), probs.to("cpu")
    ):
        table.add_data(wandb.Image(img[0].numpy() * 255), pred, targ, *prob.numpy())
    wandb.log({"predictions_table": table}, commit=False)


def train():
    # Train your model and upload checkpoints
    # Launch 3 experiments, trying different dropout rates
    for _ in range(3):
        # initialise a wandb run
        wandb.init(
            project="pytorch-intro",
            config={
                "epochs": 5,
                "batch_size": 128,
                "lr": 1e-3,
                "dropout": random.uniform(0.01, 0.80),
            },
        )

        # Copy your config
        config = wandb.config

        # Get the data
        train_dl = get_dataloader(is_train=True, batch_size=config.batch_size)
        valid_dl = get_dataloader(is_train=False, batch_size=2 * config.batch_size)
        n_steps_per_epoch = math.ceil(len(train_dl.dataset) / config.batch_size)

        # A simple MLP model
        # model = get_model(config.dropout)

        # Make the loss and optimizer
        loss_func = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

        # Training
        example_ct = 0
        step_ct = 0
        for epoch in range(config.epochs):
            model.train()
            for step, (images, labels) in enumerate(train_dl):
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                train_loss = loss_func(outputs, labels)
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()

                example_ct += len(images)
                metrics = {
                    "train/train_loss": train_loss,
                    "train/epoch": (step + 1 + (n_steps_per_epoch * epoch))
                    / n_steps_per_epoch,
                    "train/example_ct": example_ct,
                }

                if step + 1 < n_steps_per_epoch:
                    # Log train metrics to wandb
                    wandb.log(metrics)

                step_ct += 1

            val_loss, accuracy = validate_model(
                model, valid_dl, loss_func, log_images=(epoch == (config.epochs - 1))
            )

            # Log train and validation metrics to wandb
            val_metrics = {"val/val_loss": val_loss, "val/val_accuracy": accuracy}
            wandb.log({**metrics, **val_metrics})

            # Save the model checkpoint to wandb
            torch.save(model, "my_model.pt")
            wandb.log_model(
                "./my_model.pt",
                "my_mnist_model",
                aliases=[f"epoch-{epoch+1}_dropout-{round(wandb.config.dropout, 4)}"],
            )

            print(
                f"Epoch: {epoch+1}, Train Loss: {train_loss:.3f}, Valid Loss: {val_loss:3f}, Accuracy: {accuracy:.2f}"
            )

        # If you had a test set, this is how you could log it as a Summary metric
        wandb.summary["test_accuracy"] = 0.8

        # Close your wandb run
        wandb.finish()



# def train():
#     model = WholeModel()

#     bs = 16 # batch size
#     # hidden_size =
#     # sequence_length =
#     # num_classes =

#     Leye = torch.rand(bs,3, 16, 16)
#     Reye = torch.rand(bs,3, 16, 16)
#     FaceData = torch.rand(bs,3, 16, 16)  # currently matched with eyes......

#     out = model(Leye, Reye, FaceData)
#     print(out.shape)





if __name__ == "__main__":
    train()



