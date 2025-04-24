import torch
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
import h5py
from EarlyStopping import EarlyStopping
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# https://pytorch.org/tutorials/recipes/recipes/custom_dataset_transforms_loader.html

class GazeDatasetFromPaths(Dataset):
    '''for this class, the processing of label_path might need to be changed,
    because the label_path is a csv file, not a folder'''
    def __init__(self, root_dir, label_path, transform=None, camera_dirs=None):
        self.root_dir = root_dir
        self.label_path = label_path
        self.transform = transform

        if camera_dirs is None:
            self.camera_dirs = ['l', 'r', 'c']
        else:
            self.camera_dirs = camera_dirs

        # all sub folders starting with 'step' in the base path
        self.step_folders = sorted([
            d for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d)) and d.startswith('step')
        ])

        self.Leye_imgs_path = []
        self.Reye_imgs_path = []
        self.face_imgs_path = []
        self.labels = []

        for step_folder in self.step_folders:
            for camera_dir in self.camera_dirs:
                camera_path = os.path.join(root_dir, step_folder, camera_dir) # create a cam path, l or r or c
                # print("camera_path", camera_path)
                label_path = os.path.join(root_dir, step_folder, camera_dir)
                '''check if the camera path exists and contains the required folders'''
                if os.path.exists(camera_path):
                    if all(os.path.exists(os.path.join(camera_path, folder))
                           for folder in ['left_eye', 'right_eye', 'face']): # every l r c folder contains 3 sub folders
                        '''check if folders contain images'''
                        ''' is this still needed?'''
                        left_images_path = os.listdir(os.path.join(camera_path, 'left_eye'))
                        right_images_path = os.listdir(os.path.join(camera_path, 'right_eye'))
                        face_images_path = os.listdir(os.path.join(camera_path, 'face'))

                        if left_images_path and right_images_path and face_images_path:
                            for i in range(len(left_images_path)):
                                self.Leye_imgs_path.append(os.path.join(camera_path, 'left_eye', left_images_path[i]))
                                self.Reye_imgs_path.append(os.path.join(camera_path, 'right_eye', right_images_path[i]))
                                self.face_imgs_path.append(os.path.join(camera_path, 'face', face_images_path[i]))
                ''' how to match the label in .h5 file with the image sets?'''
                if os.path.exists(self.label_path):
                    # print("label_path exists", self.label_path)
                    # self.labels = pd.read_csv(label_path, header=None).values.astype('float32') ## change to read .h5 file
                    label_file = []
                    for l in os.listdir(label_path):
                        if l.endswith('.h5'):
                            label_file.append(l) # contai latest h5 file
                            #print(len(label_file))

                    if label_file:
                        label_path_h5 = os.path.join(camera_path, label_file[0])
                        # print("asefdvafcesrdfacdef")
                        # print(label_path_h5)
                        # print("asefdvafcesrdfacdef")
                        with h5py.File(label_path_h5, 'r') as f:
                            h5_labels = f['face_g_tobii/data'][:]
                            self.labels.extend(h5_labels)

        print(f"Number of path: {len(self.Leye_imgs_path)}")
        print(f"Number of labels: {len(self.labels)}")



    def __len__(self):
        '''this value is the number of images in the dataset, not the number of folders, here it's calculated by the number of image sets (from the folders)'''
        return len(self.Leye_imgs_path)

    def __getitem__(self, idx):
        '''remember, step_folder specifies the l r c cams, and camera_dir specifies the left_eye, right_eye, face folders'''
        # step_folder, camera_dir = self.image_sets[idx]
        # folder_path = os.path.join(self.root_dir, step_folder, camera_dir)
        ''' HERE!!!!!'''
        left_path = self.Leye_imgs_path[idx]
        right_path = self.Reye_imgs_path[idx]
        face_path = self.face_imgs_path[idx]
        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        left_img = Image.open(left_path).convert("RGB")
        right_img = Image.open(right_path).convert("RGB")
        face_img = Image.open(face_path).convert("RGB")

        if self.transform:
            left_img = self.transform(left_img)
            right_img = self.transform(right_img)
            face_img = self.transform(face_img)

        return left_img, right_img, face_img, label


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
        num_workers=1,
        drop_last=True # added this because there is always some error at the end of epoch

    )

    return loader


# def dot_product_loss(pred, target):
#
#     pred = nn.functional.normalize(pred, p=2, dim=1) # Resulting vector will have the correct direction but unit vector
#     target = nn.functional.normalize(target, p=2, dim=1)
#     print(pred)
#     print(target)
#     return torch.sum(pred * target, dim=1).mean()
def spherical_to_cartesian(theta_phi):
    theta = theta_phi[:, 0]
    phi = theta_phi[:, 1]
    if(torch.isnan(theta_phi).any()):
        print(f"theta range: min={theta.min().item()}, max={theta.max().item()}, NaN={torch.isnan(theta).any()}")
        print(f"phi range: min={phi.min().item()}, max={phi.max().item()}, NaN={torch.isnan(phi).any()}")

    x = torch.sin(theta) * torch.cos(phi)
    y = torch.sin(theta) * torch.sin(phi)
    z = torch.cos(theta)

    
    result = torch.cat([x.unsqueeze(1), y.unsqueeze(1), z.unsqueeze(1)], dim=1)
    # result = torch.stack([x, y, z], dim=1)

    if (torch.isnan(result).any()):
        print(f"cartesian result NaN check: {torch.isnan(result).any()}")

    return result


def angular_error(pred_theta_phi, target_theta_phi):
    if (torch.isnan(pred_theta_phi).any() or torch.isnan(pred_theta_phi).any()):
        print(f"pred_theta_phi NaN check: {torch.isnan(pred_theta_phi).any()}")
        print(f"target_theta_phi NaN check: {torch.isnan(target_theta_phi).any()}")

    pred_vec = spherical_to_cartesian(pred_theta_phi)
    target_vec = spherical_to_cartesian(target_theta_phi)

    pred_vec = nn.functional.normalize(pred_vec, p=2, dim=1)
    if (torch.isnan(pred_vec).any()):
        print(f"normalized pred_vec NaN check: {torch.isnan(pred_vec).any()}")

    target_vec = nn.functional.normalize(target_vec, p=2, dim=1)
    if (torch.isnan(target_vec).any()):
        print(f"normalized target_vec NaN check: {torch.isnan(target_vec).any()}")

    cos_sim = torch.sum(pred_vec * target_vec, dim=1)
    if (torch.isnan(cos_sim).any()):
        print(f"cos_sim range: min={cos_sim.min().item()}, max={cos_sim.max().item()}, NaN={torch.isnan(cos_sim).any()}")

    cos_sim = torch.clamp(cos_sim, -1.0, 1.0)

    result = torch.acos(cos_sim) * (180.0 / torch.pi)
    if (torch.isnan(result).any()):
        print(f"angular error NaN check: {torch.isnan(result).any()}")

    return result


# hook_usage_counter = {}
#
#
# def create_nan_guard_hook(name):
#     def hook(grad):
#         if torch.isnan(grad).any():
#             print(f"[HOOK] Replaced NaNs in gradient of: {name}")
#             hook_usage_counter[name] = hook_usage_counter.get(name, 0) + 1
#             return torch.where(torch.isnan(grad), torch.zeros_like(grad), grad)
#         return grad
#
#     return hook




def validate_model(model, valid_dl, loss_func, log_images=False, batch_idx=0):
    "Compute performance of the model on the validation dataset and log a wandb.Table"
    model.eval()
    # val_loss = 0.0 ## we have 2
    val_loss = 0.0  # total_loss
    total_ang_error = 0.0

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with torch.inference_mode():
        # correct = 0 # originally template is for MNIST, so this is for classification.
        for i, (Leyes, Reyes, faces, labels) in enumerate(valid_dl):
            Leyes, Reyes, faces, labels = Leyes.to(device), Reyes.to(device), faces.to(device), labels.to(device)

            # Forward pass ➡
            outputs = model(Leyes, Reyes, faces)
            val_loss += loss_func(outputs, labels) * labels.size(0)

            # # Compute accuracy and accumulate 
            # _, predicted = torch.max(outputs.data, 1)
            # correct += (predicted == labels).sum().item()
            # we are going to calculate angular error, so no need to calculate accuracy (that's for classification)     

            batch_error = angular_error(outputs, labels)  # angular error in degrees, base on batches 
            total_ang_error += batch_error.sum().item()

            # Log one batch of images to the dashboard, always same batch_idx.
            if i == batch_idx and log_images:
                log_image_table(Leyes, Reyes, faces, outputs, labels, outputs.softmax(dim=1))

    return val_loss / len(valid_dl.dataset), total_ang_error / len(valid_dl.dataset)



# Create a teble to compare the predicted values versus the true value
def log_image_table(Leyes, Reyes, faces, predicted, labels, errors):
    "Log a wandb.Table with (img, pred, target, scores)"
    # Create a wandb Table to log images, labels and predictions to
    # table = wandb.Table(
    #     columns=["Leyes", "Reyes", "faces", "pred", "target"] + [f"score_{i}" for i in range(10)]  # not this, this is for classification
    # )
    predicted = spherical_to_cartesian(predicted)
    labels = spherical_to_cartesian(labels)

    table = wandb.Table(
        columns=["left_eye", "right_eye", "face", "pred_x", "pred_y", "pred_z",
                 "target_x", "target_y", "target_z", "angular_error"]
    )

    errors = angular_error(predicted, labels)  # cal errors for each sample

    for left, right, face, pred, targ, err in zip(
            Leyes.to("cpu"), Reyes.to("cpu"), faces.to("cpu"), predicted.to("cpu"), labels.to("cpu"), errors.to("cpu")
    ):
        # img visualization in Wandblog, normalize them to 0-255
        '''normally pytorch tensors are in the shape of (c, h, w), but wandb expects them to be in the shape of (h, w, c)'''
        left_img = wandb.Image(
            left.permute(1, 2, 0).numpy() * 255)  # rearrange channels to (h, w, c) and rescale to [0, 255]
        right_img = wandb.Image(right.permute(1, 2, 0).numpy() * 255)
        face_img = wandb.Image(face.permute(1, 2, 0).numpy() * 255)

        # table.add_data(wandb.Image(img[0].numpy() * 255), pred, targ, *prob.numpy()) ### change???

        # add one row of data to the table
        table.add_data(
            left_img, right_img, face_img,
            float(pred[0]), float(pred[1]), float(pred[2]),  # predictied x,y,z
            float(targ[0]), float(targ[1]), float(targ[2]),  # actual x,y,z
            float(err.item())  # angular error, error is a tensor, so use item() to get the scalar value
            # Use err.item() to get the scalar value in the tensor instead of directly using float(err)
        )

    wandb.log({"gaze_predictions": table}, commit=False)
    # wandb.log({"predictions_table": table}, commit=False)



def train():

    
    model = WholeModel().to(device)  # Load the model
    grad_nan_log = []
    counter = [0]
    early_stopper = EarlyStopping(patience=5, min_delta=1e-3)

    # def make_ordered_hook(name):
    #     def hook_fn(grad):
    #         counter[0] += 1
    #         nan_count = torch.isnan(grad).sum().item()
    #         if nan_count > 0:
    #             grad_nan_log.append((counter[0], name, nan_count))
    #         return grad
    #
    #     return hook_fn

    # # Attach to all parameters
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         param.register_hook(make_ordered_hook(name))

    # /data/leuven/374/vsc37415/OP/
    # D:\thesis_code\OP
    # dataset_path = "/data/leuven/374/vsc37415/OP/"
    dataset_path = "/data/leuven/374/vsc37437/mango_to_vsc_test/OP"
    # dataset_validation_path = "/data/leuven/374/vsc37437/mango_to_vsc_test/OP-Val"
    # dataset_path = r"C:\Users\rohan\Desktop\Master\Master Thesis\Datasets\OP"
    # Leye_path = "/data/leuven/374/vsc37415/OP/"
    # Reye_path = ""
    # faces_path = ""
    # label_excel = "/data/leuven/374/vsc37415/data.csv"
    # label_excel = "D:/thesis_code/data.csv"
    label_excel = "/data/leuven/374/vsc37437/mango_to_vsc_test/OP"
    # label_excel = r"C:\Users\rohan\Desktop\Master\Master Thesis\Datasets\OP"

    # dataset = GazeDatasetFromPaths(dataset_path, label_excel)
    # dataloader = DataLoader(dataset, batch_size=1, shuffle=True) # shuffle true? yes, cause label is included so no issue

    # Train your model and upload checkpoints
    # Launch 3 experiments, trying different dropout rates
    for _ in range(1):
        # initialise a wandb run
        wandb.init(
            project="pytorch-intro",
            config={
                "epochs": 20,
                "batch_size": 4,
                "lr": 1e-3,
                "dropout": random.uniform(0.2, 0.3) #trying different dropout rates
            },
        )

        # Copy your config
        config = wandb.config

        # Get the data
        # combine different images from different folders
        train_dl = get_dataloader(dataset_path, label_excel, batch_size=config.batch_size, shuffle=True)
        '''!!!!!!!!!!!!'''
        valid_dl = get_dataloader(dataset_path, label_excel, batch_size=config.batch_size, shuffle=False)   # no shuffle for validation, also 2 times batch size for faster validation?
        '''!!!!!!!!!!!!'''
        n_steps_per_epoch = math.ceil(len(train_dl.dataset) / config.batch_size) 

        #########################################
        ''' what's the difference in valid_dl'''
        # # Get the data
        # train_dl = get_dataloader(is_train=True, batch_size=config.batch_size)
        # valid_dl = get_dataloader(is_train=False, batch_size=2 * config.batch_size)
        # n_steps_per_epoch = math.ceil(len(train_dl.dataset) / config.batch_size)
        ########################################


        # Make the loss and optimizer
        # loss_func = nn.CrossEntropyLoss()
        loss_func = angular_error
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

        # Training
        example_ct = 0  # track the number of examples seen so far
        step_ct = 0 # track the number of steps taken so far
        ##### same as     total_loss = 0.0 total_ang_error = 0.0 ?????????
        n = 0

        ## Register hooks for every model parameter
        # for name, param in model.named_parameters():
        #     if param.requires_grad:
        #         param.register_hook(create_nan_guard_hook(name))


        for epoch in range(config.epochs):
            model.train()
            for step, (Leyes, Reyes, faces, labels) in enumerate(train_dl):
                Leyes, Reyes, faces, labels = Leyes.to(device), Reyes.to(device), faces.to(device), labels.to(device)

                outputs = model(Leyes, Reyes, faces)
                if (torch.isnan(outputs).any()):
                    print(f"Model outputs NaN check: {torch.isnan(outputs).any()}")

                train_loss = loss_func(outputs, labels)
                if torch.isnan(train_loss).any():
                    print(f"Loss NaN check: {torch.isnan(train_loss).any()}")
                batch_error = angular_error(outputs, labels) # angular error in degrees, base on batches
                if torch.isnan(batch_error).any():
                    print(f" NaN detected in train_loss at epoch {epoch}, step {step}")


                optimizer.zero_grad()

                train_loss.mean().backward()
                # if grad_nan_log:
                #     print("\n=== NaN Gradient Order ===")
                #     for order, name, nan_count in sorted(grad_nan_log):
                #         print(f"[{order:03}] 🚨 {name} → {nan_count} NaNs")

                for name, param in model.named_parameters():
                    if(torch.isnan(param.grad).any()):
                        param.grad = torch.where(torch.isnan(param.grad), torch.zeros_like(param.grad), param.grad) # replaces all nans in the grad to zero
                        n = n + 1
                        print(n)
                        print(name)
                        print("replacing nan with 0!!..................................")


                optimizer.step()


                # example_ct += len(images) # use imgs of labels for tracking?
                example_ct += len(labels)

                metrics = {
                    "train/train_loss": train_loss.mean().item(),
                    "train/angular_error": batch_error.mean().item(),
                    "train/epoch": (step + 1 + (n_steps_per_epoch * epoch))
                    / n_steps_per_epoch,
                    "train/example_ct": example_ct,
                }

                if step + 1 < n_steps_per_epoch:
                    wandb.log(metrics)

                step_ct += 1

            # acc?? no we dont use accuracy here, we use angular error
            val_loss, val_error = validate_model(
                model, valid_dl, loss_func, log_images=(epoch == (config.epochs - 1))
            )

            # Log train and validation metrics to wandb
            val_metrics = {"val/val_loss": val_loss, 
                           "val/val_error": val_error}
            wandb.log({**metrics, **val_metrics})
            if(early_stopper(val_error)):
                print("Early stopping..................................")
                break

            # Save the model checkpoint to wandb
            torch.save(model, "my_model.pt")
            wandb.log_model(
                "./my_model.pt",
                "my_mnist_model",  # change?
                aliases=[f"epoch-{epoch+1}_dropout-{round(wandb.config.dropout, 4)}"],
            )

            print(
                f"Epoch: {epoch+1}"
            )

        # If you had a test set, this is how you could log it as a Summary metric
        wandb.summary["test_Val_error"] = val_error

        # Close your wandb run
        wandb.finish()






if __name__ == "__main__":
    train()



