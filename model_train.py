import torch
from Integrated_model import WholeModel
# import torch.utils.data as data
import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader, Subset
import torch.nn as nn
import wandb  # https://docs.wandb.ai/tutorials/experiments/
import torchvision.transforms as transforms
import random
import math
from PIL import Image
import h5py
from EarlyStopping import EarlyStopping
import logging
import sys
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from losses.angular import AngularLoss
# from src.core.gaze import pitchyaw_to_vector
from models.common import pitchyaw_to_vector
from core.gaze import angular_error as np_angular_error
import torch.nn.functional as F



# logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training_debug.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Enable anomaly detection
# torch.autograd.set_detect_anomaly(True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# logger.info(f"Using device: {device}")


# https://pytorch.org/tutorials/recipes/recipes/custom_dataset_transforms_loader.html

class GazeDatasetFromPaths(Dataset):

    def __init__(self, root_dir, label_path, transform=None, camera_dirs=None, is_validation=False):
        self.root_dir = root_dir
        self.label_path = label_path
        self.transform = transform
        self.is_validation = is_validation

        if camera_dirs is None:
            self.camera_dirs = ['l', 'r', 'c']
        else:
            self.camera_dirs = camera_dirs
        
        # train, test, val folders
        self.data_folders = sorted([
            d for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d)) and (d.startswith('train') or d.startswith('test') or d.startswith('val'))
        ])


        self.Leye_imgs_path = []
        self.Reye_imgs_path = []
        self.face_imgs_path = []
        self.labels = []

        for data_folder in self.data_folders:
            data_folder_path = os.path.join(root_dir, data_folder)
             # all sub folders starting with 'step' in the base path
            step_folders = sorted([
                d for d in os.listdir(data_folder_path)
                if os.path.isdir(os.path.join(data_folder_path, d)) and d.startswith('step')
            ])
        
            for step_folder in step_folders:
                step_path = os.path.join(data_folder_path, step_folder)
                for camera_dir in self.camera_dirs:
                    camera_path = os.path.join(step_path, camera_dir)  # create a cam path, l or r or c
                    # print("camera_path", camera_path)
                    label_path = os.path.join(step_path, camera_dir)
                    '''check if the camera path exists and contains the required folders'''
                    if os.path.exists(camera_path):
                        if all(os.path.exists(os.path.join(camera_path, folder))
                            for folder in
                            ['left_eye', 'right_eye', 'face']):  # every l r c folder contains 3 sub folders
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
                                label_file.append(l)  # contai latest h5 file
                                # print(len(label_file))

                        if label_file:
                            label_path_h5 = os.path.join(camera_path, label_file[0])
                            # print("asefdvafcesrdfacdef")
                            # print(label_path_h5)
                            # print("asefdvafcesrdfacdef")
                            with h5py.File(label_path_h5, 'r') as f:
                                h5_labels = f['face_g_tobii/data'][:]
                                self.labels.extend(h5_labels)

        logger.info(f"Number of path: {len(self.Leye_imgs_path)}")
        logger.info(f"Number of labels: {len(self.labels)}")
        
        # save the original dataset for validation (via a reference)
        self.original_full_dataset = self

    def __len__(self):
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


def get_dataloader(folder_path, label_path, batch_size, shuffle=True, is_validation=False):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = GazeDatasetFromPaths(folder_path, label_path, transform=transform, is_validation=is_validation)
    
    ''' create a subset of the dataset for validation '''
    # when calling get_dataloader, if "is_validation=True", create a subset of the dataset
    if is_validation:
        # a reference to the original dataset is saved in the dataset object itself
        dataset.original_full_dataset = dataset
        
        num_subset = 128  
        if len(dataset) > num_subset:
            subset_indices = sorted(np.random.permutation(len(dataset))[:num_subset])
            subset = Subset(dataset, subset_indices)
            # reference to the original dataset for the subset
            subset.original_full_dataset = dataset
            dataset = subset
            logger.info(f"Created validation subset with {len(subset)} samples from {len(subset.original_full_dataset)} total samples")
    
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True,
        num_workers=8,
        drop_last=True if shuffle else False  # added this because there is always some error at the end of epoch # but for validation, we need all samples
    )
    
    return loader

def do_final_full_test(model, valid_dl, loss_func):
    logger.info("# Now beginning full test on the complete validation dataset.")
    logger.info("# Hold on tight, this might take a while.")
    
    # get the original full dataset from the validation dataloader
    if isinstance(valid_dl.dataset, Subset) and hasattr(valid_dl.dataset, 'original_full_dataset'):
        # if the dataset is a subset, get the original full dataset
        full_dataset = valid_dl.dataset.original_full_dataset
    else:
        # if the dataset is not a subset, use the dataset directly
        full_dataset = valid_dl.dataset
    
    # create a new dataloader for the full val set
    full_loader = DataLoader(
        full_dataset,
        batch_size=valid_dl.batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )
    
    logger.info(f"Created full validation dataloader with {len(full_dataset)} samples")
    
    model.eval()
    val_loss = 0.0
    total_ang_error = 0.0
    samples_processed = 0
    
    with torch.inference_mode():
        for i, (Leyes, Reyes, faces, labels) in enumerate(full_loader):
            batch_size = labels.size(0)
            
            Leyes, Reyes, faces, labels = Leyes.to(device), Reyes.to(device), faces.to(device), labels.to(device)

            # forward pass
            outputs = model(Leyes, Reyes, faces)
            
            val_loss += loss_func.calculate_mean_loss(outputs, labels).item() * batch_size
            batch_error = loss_func.calculate_loss(outputs, labels)
            total_ang_error += batch_error.sum().item()
            
            samples_processed += batch_size
            
            # record the loss and error for each batch
            if i == 0:
                log_image_table(Leyes, Reyes, faces, outputs, labels, outputs.softmax(dim=1))
    
    final_loss = val_loss / samples_processed
    final_error = total_ang_error / samples_processed
    
    logger.info(f"Full validation results - Loss: {final_loss:.4f}, Angular Error: {final_error:.4f}")
    
    wandb.log({
        "final_test/loss": final_loss,
        "final_test/angular_error": final_error
    })
    
    wandb.summary["final_test_loss"] = final_loss
    wandb.summary["final_test_angular_error"] = final_error
    
    return final_loss, final_error


def spherical_to_cartesian(theta_phi):  # can be changed to models/common
    if torch.isnan(theta_phi).any():
        logger.warning(f"theta_phi NaN check: {torch.isnan(theta_phi).any()}")

    # recored the range of theta and phi (input)
    # logger.debug(f"theta range: {theta_phi[:, 0].min().item()} to {theta_phi[:, 0].max().item()}, NaN: {torch.isnan(theta_phi[:, 0]).any()}")
    # logger.debug(f"phi range: {theta_phi[:, 1].min().item()} to {theta_phi[:, 1].max().item()}, NaN: {torch.isnan(theta_phi[:, 1]).any()}")

    '''the inputs are pytorch tensors, so we need to convert them to numpy arrays for the function'''
    # I guess there is a pitchtoyaw func in EVE lib for pytorch tensors directly (in angular.py ???)'''
    theta_phi_np = theta_phi.detach().cpu().numpy()
    vectors_np = pitchyaw_to_vector(theta_phi_np)  # EVE lib func
    vectors_torch = torch.from_numpy(vectors_np).to(theta_phi.device).type(theta_phi.dtype)

    # NaN check for output
    # if torch.isnan(vectors_torch).any():
    #     logger.warning(f"cartesian result NaN check: {torch.isnan(vectors_torch).any()}")
    #     logger.warning(f"x NaN check: {torch.isnan(vectors_torch[:, 0]).any()}")
    #     logger.warning(f"y NaN check: {torch.isnan(vectors_torch[:, 1]).any()}")
    #     logger.warning(f"z NaN check: {torch.isnan(vectors_torch[:, 2]).any()}")

    return vectors_torch


def angular_error(a, b):  # losses/angular error, can be changed to directly use the lib func
    """Differentiable PyTorch implementation of calculating angular error, using the logic in angular.py directly"""
    a_vec = pitchyaw_to_vector(a)  # is this func the pytorch version?
    b_vec = pitchyaw_to_vector(b)

    # calculate cosine similarity (which is error)
    sim = F.cosine_similarity(a_vec, b_vec, dim=1, eps=1e-8)
    sim = F.hardtanh_(sim, min_val=-1 + 1e-8,
                      max_val=1 - 1e-8)  # same as Anuglarloss #from losses.angular import AngularLoss

    # conver to angles
    to_degrees = 180. / torch.pi

    return torch.acos(sim) * to_degrees


# Custom exception class for NaN detection
class NaNDetectedException(Exception):
    """Exception raised when NaN values are detected"""
    pass


''' some doubts'''


def validate_model(model, valid_dl, loss_func, log_images=False, batch_idx=0):
    "Compute performance of the model on the validation dataset and log a wandb.Table"
    model.eval()
    val_loss = 0.0
    total_ang_error = 0.0

    with torch.inference_mode():
        for i, (Leyes, Reyes, faces, labels) in enumerate(valid_dl):
            Leyes, Reyes, faces, labels = Leyes.to(device), Reyes.to(device), faces.to(device), labels.to(device)

            # Forward pass
            outputs = model(Leyes, Reyes, faces)
            
            # val_loss += loss_func.calculate_mean_loss(outputs, labels).item() * labels.size(0)
            val_loss += loss_func.calculate_mean_loss(outputs, labels).item() * labels.size(0)
            batch_error = loss_func.calculate_loss(outputs, labels)  # Use calculate_loss to get the angle error of each sample
                
            total_ang_error += batch_error.sum().item()

            # Log one batch of images
            if i == batch_idx and log_images:
                log_image_table(Leyes, Reyes, faces, outputs, labels, outputs.softmax(dim=1))

    return val_loss / len(valid_dl.dataset), total_ang_error / len(valid_dl.dataset)


''' some doubts'''


def log_image_table(Leyes, Reyes, faces, predicted, labels, errors):
    "Log a wandb.Table with (img, pred, target, scores)"

    # predicted = spherical_to_cartesian(predicted)
    # labels = spherical_to_cartesian(labels)

    predicted = pitchyaw_to_vector(predicted)
    labels = pitchyaw_to_vector(labels)
    table = wandb.Table(
        columns=["left_eye", "right_eye", "face", "pred_x", "pred_y", "pred_z",
                 "target_x", "target_y", "target_z", "angular_error"]
    )

    ###############################changed !!!!!!!!!!!!!!!!!!###########################
    # gaze.py angular.py AngularLoss as
    ang_loss = AngularLoss()
    ################################changed !!!!!!!!!!!!!!!!!!!!#####################

    with torch.no_grad():  # gradient calc is not needed?????
        errors = ang_loss.calculate_loss(predicted, labels)  # calc angular error

    for left, right, face, pred, targ, err in zip(
            Leyes.to("cpu"), Reyes.to("cpu"), faces.to("cpu"), predicted.to("cpu"), labels.to("cpu"), errors.to("cpu")
    ):
        # img visualization
        left_img = wandb.Image(left.permute(1, 2, 0).numpy() * 255)
        right_img = wandb.Image(right.permute(1, 2, 0).numpy() * 255)
        face_img = wandb.Image(face.permute(1, 2, 0).numpy() * 255)

        # add one row of data to the table
        table.add_data(
            left_img, right_img, face_img,
            float(pred[0]), float(pred[1]), float(pred[2]),
            float(targ[0]), float(targ[1]), float(targ[2]),
            float(err.item())
        )

    wandb.log({"gaze_predictions": table}, commit=False)


# Backward hook function to monitor gradients during backpropagation
# def backward_hook(module, grad_input, grad_output):
#     for g in grad_input:
#         if g is not None and torch.isnan(g).any():
#             logger.error("NaN gradient detected in backward hook")
#             raise NaNDetectedException("NaN gradient detected in backward hook")
#     return None


# Function to check gradients for NaN values
# def check_grad_nan(grad, name):
#     if grad is not None and torch.isnan(grad).any():
#         logger.error(f"NaN gradient detected in parameter {name} during backward")
#         raise NaNDetectedException(f"NaN gradient detected in parameter {name}")
#     return grad


def train():
    logger.info("Starting training...")

    model = WholeModel().to(device)  # Load the model

    # Register hooks for model parameters to detect NaN during backpropagation
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         param.register_hook(lambda grad, name=name: check_grad_nan(grad, name))

    early_stopper = EarlyStopping(patience=5, min_delta=1e-3)

    '''Rohan'''
    # dataset_path = "/data/leuven/374/vsc37437/mango_to_vsc_test/OP"
    # label_excel = "/data/leuven/374/vsc37437/mango_to_vsc_test/OP"
    # validation_dataset_path = "/data/leuven/374/vsc37437/mango_to_vsc_test/OP-Val"
    '''Rohan'''

    '''Tau'''
    dataset_path = "/data/leuven/374/vsc37415/OP2/OP"
    label_excel = "/data/leuven/374/vsc37415/OP2/OP"
    validation_dataset_path = "/data/leuven/374/vsc37415/OP2/OP_val"
    
    dataset_path = "/scratch/leuven/374/vsc37415/EVE_large/train"
    label_excel = "/scratch/leuven/374/vsc37415/EVE_large/train"
    validation_dataset_path = "/scratch/leuven/374/vsc37415/EVE_large/val"
    '''Tau'''



    # Train your model and upload checkpoints
    # Launch 3 experiments, trying different dropout rates
    for _ in range(1):
        # initialise a wandb run
        wandb.init(
            project="pytorch-intro",
            config={
                "epochs": 10,
                "batch_size": 128,
                "lr": 1e-4,
                "dropout": random.uniform(0.4, 0.5)  # trying different dropout rates
            },
        )

        # Copy your config
        config = wandb.config

        # # Get the data
        # # combine different images from different folders
        # train_dl = get_dataloader(dataset_path, label_excel, batch_size=config.batch_size, shuffle=True)
        # '''!!!!!!!!!!!!'''
        # valid_dl = get_dataloader(validation_dataset_path, label_excel, batch_size=config.batch_size,
        #                           shuffle=False)  # no shuffle for validation, also 2 times batch size for faster validation?

        train_dl = get_dataloader(dataset_path, label_excel, batch_size=config.batch_size, shuffle=True)
        valid_dl = get_dataloader(validation_dataset_path, label_excel, 
                                 batch_size=config.batch_size,
                                 shuffle=False,
                                 is_validation=True)  # is_validation=True, to create a subset of the val set
        
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
        # loss_func = angular_error
        loss_func = AngularLoss()

        # optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=1e-4)  # added weight decay

        # Training
        example_ct = 0  # track the number of examples seen so far
        step_ct = 0  # track the number of steps taken so far

        for epoch in range(config.epochs):
            logger.info(f"Starting epoch {epoch + 1}/{config.epochs}")
            model.train()
            for step, (Leyes, Reyes, faces, labels) in enumerate(train_dl):
                try:
                    Leyes, Reyes, faces, labels = Leyes.to(device), Reyes.to(device), faces.to(device), labels.to(
                        device)

                    # Use detect_anomaly to wrap forward and backward passes
                    # with torch.autograd.detect_anomaly():
                    outputs = model(Leyes, Reyes, faces)
                    # if (torch.isnan(outputs).any()):
                    #     logger.error(f"Model outputs NaN check: {torch.isnan(outputs).any()}")
                    #     raise NaNDetectedException("NaN detected in model outputs")

                    # train_loss = loss_func(outputs, labels)
                    train_loss = loss_func.calculate_mean_loss(outputs, labels)
                    # if torch.isnan(train_loss).any():
                    #     logger.error(f"Loss NaN check: {torch.isnan(train_loss).any()}")
                    #     raise NaNDetectedException("NaN detected in loss calculation")

                    # batch_error = angular_error(outputs, labels) # angular error in degrees, base on batches
                    batch_error = loss_func.calculate_loss(outputs, labels)
                    # if torch.isnan(batch_error).any():
                    #     logger.error(f"NaN detected in batch error at epoch {epoch}, step {step}")
                    #     raise NaNDetectedException("NaN detected in batch error calculation")

                    optimizer.zero_grad()

                    # Calculate and check if mean is NaN
                    loss_mean = train_loss.mean()
                    # if torch.isnan(loss_mean):
                    #     logger.error("NaN detected in loss.mean()")
                    #     raise NaNDetectedException("NaN detected in loss.mean()")

                    # Execute backpropagation
                    loss_mean.backward()

                    # Check for NaN in gradients
                    # nan_params = []
                    # for name, param in model.named_parameters():
                    #     if param.grad is not None and torch.isnan(param.grad).any():
                    #         nan_params.append(name)
                    #         logger.error(f"NaN gradient detected in {name}")
                    #         # Print parameter and gradient statistics for diagnostics
                    #         if not torch.isnan(param).all():  # Ensure param is not all NaN
                    #             logger.error(f"  - Param stats: min={param.min().item()}, max={param.max().item()}, mean={param.mean().item()}")
                    #         if not torch.isnan(param.grad).all():  # Ensure grad is not all NaN
                    #             logger.error(f"  - Grad stats: min={param.grad.min().item()}, max={param.grad.max().item()}, mean={param.grad.mean().item()}")

                    optimizer.step()

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

                    # # Periodically print training info
                    # if step % 10 == 0:
                    #     logger.info(
                    #         f"Epoch {epoch + 1}, Step {step}: Loss = {train_loss.mean().item():.4f}, Error = {batch_error.mean().item():.4f}")

                    step_ct += 1

                except NaNDetectedException as e:
                    logger.error(f"NaN detected: {e}")
                    # We don't exit here to maintain compatibility with original code behavior
                    # Instead log the error and continue training as in the original

                except Exception as e:
                    logger.error(f"Error in training step {step} of epoch {epoch}: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
                    continue

            # validate model after each epoch by using subset
            val_loss, val_error = validate_model(
                model, valid_dl, loss_func, log_images=(epoch == (config.epochs - 1))
            )

            # Log train and validation metrics to wandb
            val_metrics = {"val/val_loss": val_loss,
                           "val/val_error": val_error}
            wandb.log({**metrics, **val_metrics})

            # Early stopping check
            if (early_stopper(val_error)):
                logger.info("Early stopping triggered")
                break

            # Save the model checkpoint to wandb
            torch.save(model, "my_model.pt")
            wandb.log_model(
                "./my_model.pt",
                "my_mnist_model",  # change?
                aliases=[f"epoch-{epoch + 1}_dropout-{round(wandb.config.dropout, 4)}"],
            )

            logger.info(f"Epoch: {epoch + 1} completed. Val Loss: {val_loss:.4f}, Val Error: {val_error:.4f}")

        # # If you had a test set, this is how you could log it as a Summary metric
        # wandb.summary["test_Val_error"] = val_error

        # test full validation set
        logger.info("Training completed. Starting full evaluation on the complete validation set...")
        final_loss, final_error = do_final_full_test(model, valid_dl, loss_func)

        wandb.summary["test_Val_error"] = val_error  # for subset
        wandb.summary["test_Val_loss"] = val_loss    # for subset

        wandb.summary["final_test_error"] = final_error  # for full dataset
        wandb.summary["final_test_loss"] = final_loss    # for full dataset

        wandb.finish()


if __name__ == "__main__":
    try:
        train()
    except NaNDetectedException as e:
        logger.error(f"Training terminated due to NaN detection: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Fatal error in training: {e}")
        import traceback

        logger.error(traceback.format_exc())