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
            # self.camera_dirs = ['l', 'r', 'c']
            self.camera_dirs = ['l', 'r', 'c', 'basler']
        else:
            self.camera_dirs = camera_dirs
        
        # Get data folders
        self.data_folders = sorted([
            d for d in os.listdir(root_dir)
            # if os.path.isdir(os.path.join(root_dir, d)) and (d.startswith('train') or d.startswith('test') or d.startswith('val'))
            if os.path.isdir(os.path.join(root_dir, d)) and (d.startswith('train') or d.startswith('val'))
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
                    sequence_id = f"{data_folder}{step_folder}{camera_dir}"
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
        
        num_subset = float('inf') # reduce validation subset size
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


def validate_sequence_model(model, valid_dl, loss_func, current_step=None):
    """Evaluate sequence model on validation set with proper validity handling"""
    model.eval()
    val_loss = 0.0
    total_samples = 0
    chunk_size = 5
    
    # Collect validation metrics for a single log at the end
    all_val_metrics = {}
    
    with torch.inference_mode():
        for batch_idx, batch_data in enumerate(valid_dl):
            # Get batch data
            left_eyes = batch_data['left_eye'].to(device, non_blocking=True)     # [B, T, C, H, W]
            right_eyes = batch_data['right_eye'].to(device, non_blocking=True)   # [B, T, C, H, W]
            faces = batch_data['face'].to(device, non_blocking=True)             # [B, T, C, H, W]
            labels = batch_data['gaze'].to(device, non_blocking=True)            # [B, T, 2]
            labels_validity = batch_data['gaze_validity'].to(device, non_blocking=True)  # [B, T]
            hidden = None
            sequence_len = left_eyes.shape[1]

            chunk_losses = []
            for t in range(0, sequence_len, chunk_size):
                chunk_idx = t // chunk_size
                left_chunk = left_eyes[:, t:t+chunk_size]
                right_chunk = right_eyes[:, t:t+chunk_size]
                face_chunk = faces[:, t:t+chunk_size]
                label_chunk = labels[:, t:t+chunk_size]
                valid_chunk = labels_validity[:, t:t+chunk_size]

                outputs, hidden = model(left_chunk, right_chunk, face_chunk, hidden)
                
                # Prepare reference dictionary
                reference_dict = {
                    'gaze': label_chunk,
                    'gaze_validity': valid_chunk
                }
                    
                # Calculate loss (for 1 chunk)
                chunk_loss = loss_func(outputs, 'gaze', reference_dict) 
                chunk_losses.append(chunk_loss)

                logger.info(f"[Validation] Batch {batch_idx}, Chunk {chunk_idx+1}: Frames {t} to {min(t+chunk_size, sequence_len) - 1}, Loss = {chunk_loss.item():.4f}")

            batch_loss = torch.stack(chunk_losses).mean()
            val_loss += batch_loss.item()
            total_samples += 1  # tot sample is for the whole batch (number of sequences with 30 frames)
    
    # Calculate average validation loss (for 30 frmaes)
    avg_val_loss = val_loss / total_samples if total_samples > 0 else 0
    
    # Log validation metrics only once at the end
    all_val_metrics["val/avg_loss"] = avg_val_loss
    
    # Log all validation metrics without incrementing wandb's internal step counter
    # the input step is used to log the metrics at the current global step (avoid trancation)
    if current_step is not None:
        wandb.log(all_val_metrics, step=current_step, commit=False)
    else:
        wandb.log(all_val_metrics, commit=False)
    
    return avg_val_loss #(for 30 frames)


def train():
    logger.info("Starting training with sequence data...")

    base_model = WholeModel().to(device)
    
    model = SequentialWholeModel(base_model).to(device)

    early_stopper = EarlyStopping(patience=3, min_delta=1e-3)

    ''' Tau'''
    dataset_path = "/scratch/leuven/374/vsc37415/EVE/train"  # dataset path
    label_excel = "/scratch/leuven/374/vsc37415/EVE/train"  # label path
    validation_dataset_path = "/scratch/leuven/374/vsc37415/EVE/val"
    '''Tau'''

    for _ in range(1):
        wandb.init(
            project="pytorch-intro",
            config={
                "epochs": 10,
                "batch_size": 72,
                "lr": 1e-4,
                "weight_decay": 5e-6,  
                "num_workers": 6,
                "dropout": 0.167,  
                "sequence_length": 30,
                "train_max_steps_per_folder": 10,
                "val_max_steps_per_folder": 10,
                "warmup_steps_ratio": 0.0,
                "warmup_start_lr": 1e-7
            },
        )

        config = wandb.config

        train_dl = get_sequence_dataloader(
            dataset_path, 
            label_excel, 
            batch_size=config.batch_size, 
            sequence_length=config.sequence_length,
            max_steps_per_folder=config.train_max_steps_per_folder,
            shuffle=True, 
            num_workers=config.num_workers
        )
        
        valid_dl = get_sequence_dataloader(
            validation_dataset_path, 
            label_excel, 
            batch_size=config.batch_size,
            sequence_length=config.sequence_length,
            max_steps_per_folder=config.val_max_steps_per_folder,
            shuffle=False,
            is_validation=True,
            num_workers=config.num_workers
        )
        
        # Calculate steps per epoch and total steps
        steps_per_epoch = len(train_dl)
        total_steps = steps_per_epoch * config.epochs
        
        # we dont use warmup
        warmup_steps = int(total_steps * config.warmup_steps_ratio)
        warmup_start_lr = config.warmup_start_lr
        
        logger.info(f"Total training steps: {total_steps}, Warmup steps: {warmup_steps}")

        loss_func = AngularLoss()

        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=config.lr, 
            weight_decay=config.weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min',
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=True
        )
        
        # mixed precision training
        scaler = torch.amp.GradScaler('cuda')

        # Initialize counters for tracking progress (avoid truncation)
        global_step = 0
        global_chunk_step = 0  # New counter for individual chunks 
        example_ct = 0
        best_val_loss = float('inf')
        
        # Store losses for each epoch
        epoch_train_losses = []
        epoch_val_losses = []
        
        # Directory for best model
        best_model_dir = "/scratch/leuven/374/vsc37415/models_v100"
        os.makedirs(best_model_dir, exist_ok=True)




        for epoch in range(config.epochs):
            logger.info(f"Starting epoch {epoch + 1}/{config.epochs}")

            model.train()
            epoch_loss = 0.0
            samples_count = 0
            chunk_size = 5
            
            # Track epoch progress for logging
            batch_count = 0
            
            for batch_idx, batch_data in enumerate(train_dl):
                try:
                    # Get batch data
                    left_eyes = batch_data['left_eye'].to(device, non_blocking=True)
                    right_eyes = batch_data['right_eye'].to(device, non_blocking=True)
                    faces = batch_data['face'].to(device, non_blocking=True)
                    labels = batch_data['gaze'].to(device, non_blocking=True)
                    labels_validity = batch_data['gaze_validity'].to(device, non_blocking=True)
                    hidden = None
                    sequence_len = left_eyes.shape[1]
                    batch_size = left_eyes.size(0)
                    
                    # Clear gradients
                    optimizer.zero_grad()
                    batch_loss = 0.0
                    num_chunks = 0
                    
                    # Process each 5-frame chunk
                    for t in range(0, sequence_len, chunk_size):
                        chunk_idx = t // chunk_size
                        left_chunk = left_eyes[:, t:t+chunk_size]
                        right_chunk = right_eyes[:, t:t+chunk_size]
                        face_chunk = faces[:, t:t+chunk_size]
                        label_chunk = labels[:, t:t+chunk_size]
                        valid_chunk = labels_validity[:, t:t+chunk_size]
                        
                        # Use mixed precision training
                        with torch.amp.autocast('cuda'):
                            outputs, hidden = model(left_chunk, right_chunk, face_chunk, hidden)
                            
                            # Prepare reference dictionary
                            reference_dict = {
                                'gaze': label_chunk,
                                'gaze_validity': valid_chunk
                            }
                            
                            # Calculate loss for this chunk
                            chunk_loss = loss_func(outputs, 'gaze', reference_dict)
                            
                            # Calculate accurate epoch progress
                            batch_progress = batch_idx / steps_per_epoch
                            chunk_progress = chunk_idx / (sequence_len // chunk_size) / steps_per_epoch
                            current_progress = epoch + batch_progress + chunk_progress
                            
                            # Log metrics for this chunk  (for traning curve. The epoch wise train and val loss are logged based on 30 frames)
                            metrics = {
                                "train/chunk_loss": chunk_loss.item(),
                                "train/chunk_idx": chunk_idx,
                                "train/global_step": global_step,
                                "train/global_chunk_step": global_chunk_step,
                                "train/epoch": current_progress,
                                "train/lr": optimizer.param_groups[0]['lr']
                            }
                            wandb.log(metrics, step=global_chunk_step)
                            global_chunk_step += 1
                            
                            # Add to total loss (scaled by chunk size)
                            actual_chunk_size = left_chunk.size(1)
                            batch_loss += chunk_loss * (actual_chunk_size / chunk_size)
                            num_chunks += (actual_chunk_size / chunk_size)
                            
                            if batch_idx % 50 == 0:
                                logger.info(f"[Train Epoch {epoch+1} | Batch {batch_idx}/{steps_per_epoch}] Chunk {chunk_idx+1}: Frames {t} to {min(t+chunk_size, sequence_len) - 1}, Loss = {chunk_loss.item():.4f}")
                        
                    # Normalize the batch loss
                    batch_loss = batch_loss / num_chunks if num_chunks > 0 else 0
                    
                    # Backward pass
                    scaler.scale(batch_loss).backward()
                    
                    # Gradient clipping
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    # Update parameters
                    scaler.step(optimizer)
                    scaler.update()

                    # Update statistics
                    samples_count += batch_size
                    epoch_loss += batch_loss.item() * batch_size
                    example_ct += batch_size
                    global_step += 1
                    batch_count += 1
                
                except Exception as e:
                    logger.error(f"Error in training step {batch_idx} of epoch {epoch}: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
                    salvage_memory()
                    continue
            
            # Calculate average training loss for the epoch
            avg_train_loss = epoch_loss / samples_count if samples_count > 0 else 0
            epoch_train_losses.append(avg_train_loss)
            
            # Validate model and use global_chunk_step for consistent wandb logging
            val_loss = validate_sequence_model(model, valid_dl, loss_func, global_chunk_step)
            epoch_val_losses.append(val_loss)

            # Update learning rate scheduler
            if global_step >= warmup_steps:
                scheduler.step(val_loss)

            # Get current learning rate
            current_lr = optimizer.param_groups[0]['lr']
            
            # Record metrics for each epoch (30 frames)
            epoch_metrics = {
                "epoch/avg_train_loss": avg_train_loss,
                "epoch/avg_val_loss": val_loss,
                "epoch/lr": current_lr,
                "epoch/number": epoch + 1
            }
            wandb.log(epoch_metrics, step=global_chunk_step)
            
            # # Create train_loss vs val_loss chart (can be removed)
            # train_vs_val_loss = wandb.plot.line_series(
            #     xs=[[i+1 for i in range(len(epoch_train_losses))], [i+1 for i in range(len(epoch_val_losses))]],
            #     ys=[epoch_train_losses, epoch_val_losses],
            #     keys=["Avg Train Loss", "Avg Val Loss"],
            #     title="Average Train Loss vs Val Loss (across 5-frame chunks)",
            #     xname="Epoch"
            # )
            # wandb.log({"train_vs_val_loss": train_vs_val_loss}, step=global_chunk_step)

            # Record validation metrics
            val_metrics = {
                "val/avg_loss": val_loss,
                "val/lr": current_lr
            }
            wandb.log(val_metrics, step=global_chunk_step)

            # Check if this is the best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                # Save the best model
                best_model_path = os.path.join(best_model_dir, "best_model.pt")
                torch.save(model.state_dict(), best_model_path)
                logger.info(f"Saved new best model with error {best_val_loss:.4f} to {best_model_path}")

            # Early stopping check
            if early_stopper(val_loss):
                logger.info("Early stopping triggered")
                break

            # Record information for each epoch
            logger.info(f"Epoch: {epoch + 1}/{config.epochs} completed. Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {current_lr:.6f}")
            
            # Free memory at the end of epoch
            salvage_memory()

        # Record final results (kind of useless?)
        wandb.summary["best_val_loss"] = best_val_loss
        wandb.summary["best_model_path"] = os.path.join(best_model_dir, "best_model.pt")
        wandb.summary["total_epochs_completed"] = len(epoch_train_losses)
        wandb.summary["total_steps"] = global_step
        wandb.summary["total_chunk_steps"] = global_chunk_step
        wandb.summary["final_lr"] = optimizer.param_groups[0]['lr']

        wandb.finish()
        
        logger.info("Training completed successfully.")


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