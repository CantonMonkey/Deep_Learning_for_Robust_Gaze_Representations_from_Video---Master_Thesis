import torch
from Integrated_model import WholeModel, SequentialWholeModel
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import pandas as pd
from torch.utils.data import Dataset, DataLoader, Subset
import torch.nn as nn
import wandb
import torchvision.transforms as transforms
import random
import math
from PIL import Image
import h5py
from EarlyStopping import EarlyStopping
import logging
import sys
import numpy as np
import gc
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from losses.angular import AngularLoss
from models.common import pitchyaw_to_vector
from core.gaze import angular_error as np_angular_error
import torch.nn.functional as F


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training_debug.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def salvage_memory():
    """Attempt to free available memory"""
    torch.cuda.empty_cache()
    gc.collect()


# Sequence dataset class
class GazeSequenceDataset(Dataset):
    def __init__(self, root_dir, label_path, transform=None, camera_dirs=None, is_validation=False, 
                 sequence_length=30, max_steps_per_folder=10):
        self.root_dir = root_dir
        self.label_path = label_path
        self.transform = transform
        self.is_validation = is_validation
        self.sequence_length = sequence_length  # sequence length

        if camera_dirs is None:
            self.camera_dirs = ['l', 'r', 'c']
        else:
            self.camera_dirs = camera_dirs
        
        # Get data folders
        self.data_folders = sorted([
            d for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d)) and (d.startswith('train') or d.startswith('test') or d.startswith('val'))
        ])
        
        # Filter folders
        if not is_validation:
            self.data_folders = [d for d in self.data_folders if d.startswith('train')]
        else:
            self.data_folders = [d for d in self.data_folders if d.startswith('val')]
            
        logger.info(f"Data folder used: {self.data_folders}")
        
        # Store sequence information
        self.sequences = []
        
        # Temporarily store all frame paths and labels
        self.all_frames = {}
        
        # Extract sequences from data folders
        for data_folder in self.data_folders:
            data_folder_path = os.path.join(root_dir, data_folder)
            # Get step folders
            step_folders = sorted([
                d for d in os.listdir(data_folder_path)
                if os.path.isdir(os.path.join(data_folder_path, d)) and d.startswith('step')
            ])
            
            # Limit the number of steps
            if len(step_folders) > max_steps_per_folder:
                random.shuffle(step_folders)
                step_folders = step_folders[:max_steps_per_folder]
                logger.info(f"Limit the number of steps used in the folder {data_folder} to {max_steps_per_folder}")
            
            for step_folder in step_folders:
                step_path = os.path.join(data_folder_path, step_folder)
                
                for camera_dir in self.camera_dirs:
                    camera_path = os.path.join(step_path, camera_dir)
                    
                    if not os.path.exists(camera_path):
                        continue
                    if not all(os.path.exists(os.path.join(camera_path, folder)) 
                            for folder in ['left_eye', 'right_eye', 'face']):
                        continue
                    
                    left_images = sorted(os.listdir(os.path.join(camera_path, 'left_eye')))
                    right_images = sorted(os.listdir(os.path.join(camera_path, 'right_eye')))
                    face_images = sorted(os.listdir(os.path.join(camera_path, 'face')))
                    
                    if not (left_images and right_images and face_images):
                        continue
                    
                    label_files = [l for l in os.listdir(camera_path) if l.endswith('.h5')]
                    if not label_files:
                        continue
                    
                    label_path_h5 = os.path.join(camera_path, label_files[0])
                    try:
                        with h5py.File(label_path_h5, 'r') as f:
                            labels = f['face_g_tobii/data'][:]
                            
                            # read validity
                            if 'face_g_tobii/validity' in f:
                                validity = f['face_g_tobii/validity'][:]
                            else:
                                # if not available, create a default validity array with all True values
                                validity = np.ones(len(labels), dtype=bool)
                    except Exception as e:
                        logger.error(f"Error reading h5 file {label_path_h5}: {e}")
                        continue
                    
                    if len(labels) != len(left_images) or len(validity) != len(labels):
                        logger.warning(f"Mismatch in data lengths for {camera_path}")
                        continue
                    
                    # Create sequence
                    sequence_id = f"{data_folder}_{step_folder}_{camera_dir}"
                    self.all_frames[sequence_id] = {
                        'left_eye_paths': [os.path.join(camera_path, 'left_eye', img) for img in left_images],
                        'right_eye_paths': [os.path.join(camera_path, 'right_eye', img) for img in right_images],
                        'face_paths': [os.path.join(camera_path, 'face', img) for img in face_images],
                        'labels': labels,
                        'validity': validity  # store validity
                    }
                    
                    # Split the sequence into fixed-length segments
                    num_frames = len(left_images)
                    for seq_start in range(0, num_frames, self.sequence_length):
                        seq_end = min(seq_start + self.sequence_length, num_frames)
                        if seq_end - seq_start >= 5:  # need at least 5 frames
                            self.sequences.append({
                                'sequence_id': sequence_id,
                                'start_idx': seq_start,
                                'end_idx': seq_end
                            })
        
        logger.info(f"Total sequences: {len(self.sequences)}")
        self.original_full_dataset = self
    
    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        """return a sequence of images and labels"""
        seq = self.sequences[idx]
        sequence_id = seq['sequence_id']
        start_idx = seq['start_idx']
        end_idx = seq['end_idx']
        
        # get the sequence data
        sequence_data = self.all_frames[sequence_id]
        
        # extract paths and labels for the sequence
        left_eye_paths = sequence_data['left_eye_paths'][start_idx:end_idx]
        right_eye_paths = sequence_data['right_eye_paths'][start_idx:end_idx]
        face_paths = sequence_data['face_paths'][start_idx:end_idx]
        labels = sequence_data['labels'][start_idx:end_idx]
        validity = sequence_data['validity'][start_idx:end_idx]  # extract validity
        
        left_seq = []
        right_seq = []
        face_seq = []
        labels_seq = []
        validity_seq = []
        
        # laod every frame in the sequence
        for i in range(len(left_eye_paths)):
            # load images
            left_img = Image.open(left_eye_paths[i]).convert("RGB")
            right_img = Image.open(right_eye_paths[i]).convert("RGB")
            face_img = Image.open(face_paths[i]).convert("RGB")
            
            if self.transform:
                left_img = self.transform(left_img)
                right_img = self.transform(right_img)
                face_img = self.transform(face_img)
            
            # add in the sequence
            left_seq.append(left_img)
            right_seq.append(right_img)
            face_seq.append(face_img)
            labels_seq.append(torch.tensor(labels[i], dtype=torch.float32))
            validity_seq.append(torch.tensor(validity[i], dtype=torch.bool))  #add validity to the sequence
        
        left_seq = torch.stack(left_seq)
        right_seq = torch.stack(right_seq)
        face_seq = torch.stack(face_seq)
        labels_seq = torch.stack(labels_seq)
        validity_seq = torch.stack(validity_seq)  # convert validity to tensor
        
        return {
            'left_eye': left_seq,         # [seq_len, C, H, W]
            'right_eye': right_seq,       # [seq_len, C, H, W]
            'face': face_seq,             # [seq_len, C, H, W]
            'gaze': labels_seq,           # [seq_len, 2]
            'gaze_validity': validity_seq  # [seq_len]
        }

# Get sequence data loader
def get_sequence_dataloader(folder_path, label_path, batch_size, sequence_length=30,
                           shuffle=True, is_validation=False, num_workers=8, max_steps_per_folder=10):
    """Create sequence data loader"""
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = GazeSequenceDataset(
        folder_path, 
        label_path, 
        transform=transform, 
        is_validation=is_validation,
        sequence_length=sequence_length,
        max_steps_per_folder=max_steps_per_folder
    )
    
    # Create subset for validation set
    if is_validation:
        dataset.original_full_dataset = dataset
        
        num_subset = 128  # reduce validation subset size
        if len(dataset) > num_subset:
            subset_indices = sorted(np.random.permutation(len(dataset))[:num_subset])
            subset = Subset(dataset, subset_indices)
            subset.original_full_dataset = dataset
            dataset = subset
            logger.info(f"Created validation subset with {len(subset)} sequences")
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True,
        num_workers=num_workers,
        drop_last=True if shuffle else False
    )
    
    return loader


def validate_sequence_model(model, valid_dl, loss_func):
    """Evaluate sequence model on validation set with proper validity handling"""
    model.eval()
    val_loss = 0.0
    total_samples = 0
    
    with torch.inference_mode():
        for batch_data in valid_dl:
            # Get batch data
            left_eyes = batch_data['left_eye'].to(device, non_blocking=True)     # [B, T, C, H, W]
            right_eyes = batch_data['right_eye'].to(device, non_blocking=True)   # [B, T, C, H, W]
            faces = batch_data['face'].to(device, non_blocking=True)             # [B, T, C, H, W]
            labels = batch_data['gaze'].to(device, non_blocking=True)            # [B, T, 2]
            labels_validity = batch_data['gaze_validity'].to(device, non_blocking=True)  # [B, T]
            
            # Forward pass
            outputs = model(left_eyes, right_eyes, faces)  # [B, T, 2]
            
            # Prepare reference dictionary
            reference_dict = {
                'gaze': labels,
                'gaze_validity': labels_validity
            }
            
            # Calculate loss using loss function that respects validity
            batch_loss = loss_func(outputs, 'gaze', reference_dict)
            val_loss += batch_loss.item()
            
            # # Calculate angular error respecting validity
            # batch_errors = []
            # for b in range(outputs.size(0)):
            #     valid_indices = labels_validity[b].bool()
            #     if valid_indices.sum() > 0:
            #         pred = outputs[b, valid_indices]
            #         gt = labels[b, valid_indices]
                    
            #         pred_vec = pitchyaw_to_vector(pred)
            #         gt_vec = pitchyaw_to_vector(gt)
                    
            #         sim = F.cosine_similarity(pred_vec, gt_vec, dim=1, eps=1e-8)
            #         sim = torch.clamp(sim, min=-1+1e-8, max=1-1e-8)
                    
            #         ang_error = torch.acos(sim) * (180.0 / np.pi)
            #         # Average error for this sample
            #         batch_errors.append(ang_error.mean().item())
            
            # # Add average error for this batch
            # if batch_errors:
            #     total_ang_error += sum(batch_errors) / len(batch_errors)
            
            total_samples += 1
    
    # Return average loss and angular error
    return val_loss / total_samples


def do_final_full_test(model, valid_dl, loss_func):
    """Test sequence model on the full validation dataset with proper validity handling"""
    logger.info("# Starting validation on the full validation dataset...")
    
    # Get original complete dataset
    if isinstance(valid_dl.dataset, Subset) and hasattr(valid_dl.dataset, 'original_full_dataset'):
        full_dataset = valid_dl.dataset.original_full_dataset
    else:
        full_dataset = valid_dl.dataset
    
    # Create new data loader for the complete dataset
    full_loader = DataLoader(
        full_dataset,
        batch_size=96,  # reduce batch size to fit GPU memory
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )
    
    logger.info(f"Created a full validation data loader containing {len(full_dataset)} sequences")
    
    model.eval()
    val_loss = 0.0
    total_samples = 0
    total_valid_frames = 0
    
    with torch.inference_mode():
        for batch_data in full_loader:
            # Get batch data
            left_eyes = batch_data['left_eye'].to(device, non_blocking=True)
            right_eyes = batch_data['right_eye'].to(device, non_blocking=True)
            faces = batch_data['face'].to(device, non_blocking=True)
            labels = batch_data['gaze'].to(device, non_blocking=True)
            labels_validity = batch_data['gaze_validity'].to(device, non_blocking=True)
            
            # Forward pass
            outputs = model(left_eyes, right_eyes, faces)
            
            # Prepare reference dictionary
            reference_dict = {
                'gaze': labels,
                'gaze_validity': labels_validity
            }
            
            # Calculate loss using validity-aware loss function
            batch_loss = loss_func(outputs, 'gaze', reference_dict)
            val_loss += batch_loss.item()
            
            # # Calculate angular error with proper validity handling
            # batch_errors = []
            # for b in range(outputs.size(0)):
            #     valid_indices = labels_validity[b].bool()
            #     num_valid = valid_indices.sum().item()
                
            #     if num_valid > 0:
            #         total_valid_frames += num_valid
                    
            #         pred = outputs[b, valid_indices]
            #         gt = labels[b, valid_indices]
                    
            #         pred_vec = pitchyaw_to_vector(pred)
            #         gt_vec = pitchyaw_to_vector(gt)
                    
            #         sim = F.cosine_similarity(pred_vec, gt_vec, dim=1, eps=1e-8)
            #         sim = torch.clamp(sim, min=-1+1e-8, max=1-1e-8)
                    
            #         ang_error = torch.acos(sim) * (180.0 / np.pi)
            #         # Average error for this sample
            #         batch_errors.append(ang_error.mean().item())
            
            # # Add average error for this batch
            # if batch_errors:
            #     total_ang_error += sum(batch_errors) / len(batch_errors)
            
            total_samples += 1
    
    final_loss = val_loss / total_samples
    
    logger.info(f"Full validation results - loss: {final_loss:.4f} degrees")
    logger.info(f"Total valid frames processed: {total_valid_frames}")
    
    wandb.log({
        "final_test/loss": final_loss,
        "final_test/valid_frames": total_valid_frames
    })
    
    wandb.summary["final_test_loss"] = final_loss
    wandb.summary["final_test_valid_frames"] = total_valid_frames
    
    return final_loss

def train():
    logger.info("Starting training with sequence data...")

    # Initialize base model
    base_model = WholeModel().to(device)
    
    # Create sequence model
    model = SequentialWholeModel(base_model).to(device)

    early_stopper = EarlyStopping(patience=5, min_delta=1e-3)

    '''Tau'''
    # dataset_path = "/scratch/leuven/374/vsc37415/EVE_large/train"  # dataset path
    # label_excel = "/scratch/leuven/374/vsc37415/EVE_large/train"  # label path
    # validation_dataset_path = "/scratch/leuven/374/vsc37415/EVE_large/val"

    dataset_path = "/scratch/leuven/374/vsc37415/EVE/train"  # dataset path
    label_excel = "/scratch/leuven/374/vsc37415/EVE/train"  # label path
    validation_dataset_path = "/scratch/leuven/374/vsc37415/EVE/val"
    '''Tau'''

    

    # Start training experiment
    for _ in range(1):
        # Initialize wandb
        wandb.init(
            project="pytorch-intro",
            config={
                "epochs": 20,
                "batch_size": 12,
                "lr": 2e-5,
                "weight_decay": 5e-6,  
                "num_workers" : 8,
                "dropout": 0.167,   # useless
                "sequence_length": 30,
                "max_steps_per_folder": 10,
                #"max_steps_per_folder": float('inf'), # no limit
                "warmup_steps_ratio": 0.15,  # Warmup for 10% of total training steps
                "warmup_start_lr": 1e-7  # Start with tiny non-zero learning rate
            },
        )

        # Get configuration
        config = wandb.config

        # Get sequence data loaders
        train_dl = get_sequence_dataloader(
            dataset_path, 
            label_excel, 
            batch_size=config.batch_size, 
            sequence_length=config.sequence_length,
            max_steps_per_folder=config.max_steps_per_folder,
            shuffle=True, 
            num_workers=config.num_workers
        )
        
        valid_dl = get_sequence_dataloader(
            validation_dataset_path, 
            label_excel, 
            batch_size=config.batch_size,
            sequence_length=config.sequence_length,
            max_steps_per_folder=config.max_steps_per_folder,
            shuffle=False,
            is_validation=True,
            num_workers=config.num_workers
        )
        
        # Calculate steps per epoch and total steps
        steps_per_epoch = len(train_dl)
        total_steps = steps_per_epoch * config.epochs
        
        # Define warmup parameters for step-based warmup
        warmup_steps = int(total_steps * config.warmup_steps_ratio)
        warmup_start_lr = config.warmup_start_lr
        initial_lr = config.lr
        
        logger.info(f"Total training steps: {total_steps}, Warmup steps: {warmup_steps}")

        # Use angular loss with validity handling
        loss_func = AngularLoss()

        # Use AdamW optimizer with warmup_start_lr
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=warmup_start_lr,  # Start with lower learning rate
            weight_decay=config.weight_decay
        )
        
        # Use ReduceLROnPlateau - this will only be used after warmup
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min',
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=True
        )
        
        # Initialize mixed precision training
        scaler = torch.amp.GradScaler('cuda')

        # Initialize training variables
        example_ct = 0
        global_step = 0
        best_val_loss = float('inf')

        # Store losses for each epoch
        epoch_train_losses = []
        epoch_val_losses = []

        for epoch in range(config.epochs):
            logger.info(f"Starting epoch {epoch + 1}/{config.epochs}")
            
            model.train()
            epoch_loss = 0.0
            samples_count = 0
            
            for step, batch_data in enumerate(train_dl):
                try:
                    # Apply step-based warmup
                    if global_step < warmup_steps:
                        # Linear warmup
                        warmup_factor = global_step / warmup_steps
                        current_lr = warmup_start_lr + (initial_lr - warmup_start_lr) * warmup_factor # initial_lr = config.lr
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = current_lr
                        
                        # Log warmup progress periodically
                        if global_step % 20 == 0:
                            logger.info(f"Warmup step {global_step}/{warmup_steps}, LR = {current_lr:.6f}")
                    
                    # Get batch data
                    left_eyes = batch_data['left_eye'].to(device, non_blocking=True)
                    right_eyes = batch_data['right_eye'].to(device, non_blocking=True)
                    faces = batch_data['face'].to(device, non_blocking=True)
                    labels = batch_data['gaze'].to(device, non_blocking=True)
                    labels_validity = batch_data['gaze_validity'].to(device, non_blocking=True)
                    
                    # Clear gradients
                    optimizer.zero_grad()
                    
                    # Use mixed precision training
                    with torch.amp.autocast('cuda'):
                        # Forward pass
                        outputs = model(left_eyes, right_eyes, faces)
                        
                        # Prepare reference dictionary
                        reference_dict = {
                            'gaze': labels,
                            'gaze_validity': labels_validity
                        }
                        
                        # Calculate loss
                        train_loss = loss_func(outputs, 'gaze', reference_dict)
                    
                    # Backward pass
                    scaler.scale(train_loss).backward()
                    
                    # Gradient clipping
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    # Update parameters
                    scaler.step(optimizer)
                    scaler.update()

                    # Update statistics
                    batch_size = left_eyes.size(0)
                    samples_count += batch_size
                    epoch_loss += train_loss.item() * batch_size
                    
                    example_ct += batch_size
                    global_step += 1

                    # Get current learning rate
                    current_lr = optimizer.param_groups[0]['lr']

                    # Record metrics
                    metrics = {
                        "train/train_loss": train_loss.item(),
                        "train/epoch": (step + 1 + (steps_per_epoch * epoch)) / steps_per_epoch,
                        "train/example_ct": example_ct,
                        "train/lr": current_lr,
                        "train/global_step": global_step
                    }

                    wandb.log(metrics)

                    # Print training information periodically
                    if step % 20 == 0:
                        logger.info(
                            f"Epoch {epoch + 1}, Step {step}, Global Step {global_step}: Loss = {train_loss.item():.4f}, "
                            f"LR = {current_lr:.6f}"
                        )

                except Exception as e:
                    logger.error(f"Error in training step {step} of epoch {epoch}: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
                    salvage_memory()
                    continue
            
            # Calculate average training loss
            avg_train_loss = epoch_loss / samples_count if samples_count > 0 else 0
            epoch_train_losses.append(avg_train_loss)
            
            # Validate model
            val_loss = validate_sequence_model(model, valid_dl, loss_func)
            epoch_val_losses.append(val_loss)

            # Update learning rate scheduler - only after warmup
            if global_step >= warmup_steps:
                scheduler.step(val_loss)

            # Get current learning rate
            current_lr = optimizer.param_groups[0]['lr']
            
            # Record metrics for each epoch
            epoch_metrics = {
                "epoch/train_loss": avg_train_loss,
                "epoch/val_loss": val_loss,
                "epoch/lr": current_lr,
                "epoch/number": epoch + 1
            }
            wandb.log(epoch_metrics)
            
            # Create train_loss vs val_loss chart
            train_vs_val_loss = wandb.plot.line_series(
                xs=[[i+1 for i in range(epoch+1)], [i+1 for i in range(epoch+1)]],
                ys=[epoch_train_losses, epoch_val_losses],
                keys=["Train Loss", "Val Loss"],
                title="Train Loss vs Val Loss",
                xname="Epoch",
                yname="angular error (degrees)",
            )
            wandb.log({"train_vs_val_loss": train_vs_val_loss})

            # Record validation metrics
            val_metrics = {
                "val/val_loss": val_loss,
                "val/lr": current_lr
            }
            wandb.log(val_metrics)

            # Check if this is the best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                # Save the best model
                best_model_path = f"best_model_epoch_{epoch}.pt"
                torch.save(model.state_dict(), best_model_path)
                logger.info(f"Saved best model with error {best_val_loss:.4f} to {best_model_path}")

            # Early stopping check
            if early_stopper(val_loss):
                logger.info("Early stopping triggered")
                break

            # Record information for each epoch
            logger.info(f"Epoch: {epoch + 1} completed. Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {current_lr:.6f}")
            
            # Free memory at the end of epoch
            salvage_memory()

        # Load the best model for final testing
        best_model_path = f"best_model_epoch_{epoch}.pt"
        if os.path.exists(best_model_path):
            model.load_state_dict(torch.load(best_model_path))
            logger.info(f"Loaded best model from {best_model_path} for final testing")

        # Test on the full validation set
        logger.info("Training completed. Starting full evaluation on the complete validation set...")
        final_loss = do_final_full_test(model, valid_dl, loss_func)

        # Record final results
        wandb.summary["best_val_loss"] = best_val_loss
        wandb.summary["best_model_path"] = best_model_path
        wandb.summary["final_test_loss"] = final_loss

        wandb.finish()


if __name__ == "__main__":
    try:
        # Set random seed for reproducibility
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        np.random.seed(42)
        
        train()
    except Exception as e:
        logger.error(f"Fatal error in training: {e}")
        import traceback
        logger.error(traceback.format_exc())
        salvage_memory()