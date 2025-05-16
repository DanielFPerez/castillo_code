import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, f1_score
from sklearn.metrics import confusion_matrix, precision_score, recall_score
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer, AutoModel
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
import time

from transformers import get_linear_schedule_with_warmup
from transformers import get_scheduler
import math

from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define paths
DATA_DIR = "path/to/your/data/"  # Update this to your data directory
MODELS_DIR = "saved_models/"
METRICS_DIR = "metrics/"

MODEL = "intfloat/multilingual-e5-large-instruct"
HIDDEN_SIZE = 1024  # Hidden size for the model

# Create directories if they don't exist
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)

# Load data
def load_data(data_dir):
    train_path = os.path.join(data_dir, "sanitized_train.json")
    val_path = os.path.join(data_dir, "sanitized_validate.json")
    test_path = os.path.join(data_dir, "sanitized_test.json")
    
    with open(train_path, 'r') as f:
        print(f"Loading training data from {train_path}")
        train_data = json.load(f)
    
    with open(val_path, 'r') as f:
        print(f"Loading validation data from {val_path}")
        val_data = json.load(f)
    
    with open(test_path, 'r') as f:
        print(f"Loading test data from {test_path}")
        test_data = json.load(f)
    print("Data loading complete.")
    return train_data, val_data, test_data

# Custom Dataset class
class ResponseLengthDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512, task='regression', bin_size=50, max_bin_threshold=2000):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.task = task
        self.bin_size = bin_size
        self.max_bin_threshold = max_bin_threshold  # New parameter for max bin threshold
        
        # Encode model names
        self.models = list(set([item['model'] for item in data]))
        self.model_encoder = {model: idx for idx, model in enumerate(self.models)}
        
        # Calculate normalization statistics for regression targets
        if task == 'regression':
            means = np.array([item['output_mean'] for item in data])
            stds = np.array([item['output_std'] for item in data])
            
            self.mean_target_mean = means.mean()
            self.std_target_mean = means.std()
            self.mean_target_std = stds.mean()
            self.std_target_std = stds.std()
            
            print(f"Target normalization stats:")
            print(f"  Mean output_mean: {self.mean_target_mean:.2f}, Std: {self.std_target_mean:.2f}")
            print(f"  Mean output_std: {self.mean_target_std:.2f}, Std: {self.std_target_std:.2f}")
        elif task == 'classification':
            # Count distribution of p99 values
            p99_values = [item['output_percentiles']['p99'] for item in data]
            below_threshold = sum(1 for p in p99_values if p < max_bin_threshold)
            above_threshold = sum(1 for p in p99_values if p >= max_bin_threshold)
            
            # Calculate number of classes (bins below threshold + 1 class for above threshold)
            self.num_bins_below_threshold = max_bin_threshold // bin_size
            self.num_classes = self.num_bins_below_threshold + 1
            
            print(f"Classification binning stats:")
            print(f"  Values below {max_bin_threshold}: {below_threshold} ({below_threshold/len(p99_values):.1%})")
            print(f"  Values above {max_bin_threshold}: {above_threshold} ({above_threshold/len(p99_values):.1%})")
            print(f"  Number of classes: {self.num_classes} ({self.num_bins_below_threshold} bins of {bin_size} tokens + 1 bin for ≥{max_bin_threshold})")
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Tokenize text
        encoding = self.tokenizer(
            item['prompt_text'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Extract features
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        
        # One-hot encode model
        model_idx = self.model_encoder[item['model']]
        model_tensor = torch.zeros(len(self.models))
        model_tensor[model_idx] = 1
        
        if self.task == 'regression':
            # Normalize targets for regression
            output_mean = (item['output_mean'] - self.mean_target_mean) / max(self.std_target_mean, 1e-9)
            output_std = (item['output_std'] - self.mean_target_std) / max(self.std_target_std, 1e-9)
            target = torch.tensor([output_mean, output_std], dtype=torch.float)
            
        else:  # Classification with custom binning
            # Get p99 percentile
            p99 = item['output_percentiles']['p99']
            
            # Apply custom binning logic
            if p99 < self.max_bin_threshold:
                # For values below threshold, use normal binning
                p99_class = int(p99 // self.bin_size)
            else:
                # For values above threshold, assign to the last class
                p99_class = self.num_bins_below_threshold
                
            target = torch.tensor(p99_class, dtype=torch.long)
            
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask, 
            'model': model_tensor,
            'target': target
        }

    # Helper methods for denormalizing predictions
    def denormalize_mean(self, normalized_value):
        return normalized_value * self.std_target_mean + self.mean_target_mean
    
    def denormalize_std(self, normalized_value):
        return normalized_value * self.std_target_std + self.mean_target_std
    
    # Helper method to convert class to token range
    def class_to_token_range(self, class_idx):
        if class_idx < self.num_bins_below_threshold:
            lower = class_idx * self.bin_size
            upper = (class_idx + 1) * self.bin_size
            return f"{lower}-{upper} tokens"
        else:
            return f"≥{self.max_bin_threshold} tokens"


# Custom Loss Function
class WeightedRegressionLoss(nn.Module):
    def __init__(self, mean_weight=0.7, std_weight=0.3, mse_weight=0.5, mae_weight=0.5):
        super(WeightedRegressionLoss, self).__init__()
        self.mse_loss = nn.MSELoss(reduction='none')
        self.mae_loss = nn.L1Loss(reduction='none')
        self.mean_weight = mean_weight  # Weight for output_mean prediction
        self.std_weight = std_weight    # Weight for output_std prediction
        self.mse_weight = mse_weight    # Weight for MSE loss component
        self.mae_weight = mae_weight    # Weight for MAE loss component
        
        print(f"Using weighted loss: mean_weight={mean_weight}, std_weight={std_weight}, "
              f"mse_weight={mse_weight}, mae_weight={mae_weight}")
    
    def forward(self, preds, targets):
        # Split predictions and targets
        mean_preds = preds[:, 0]
        std_preds = preds[:, 1]
        mean_targets = targets[:, 0]
        std_targets = targets[:, 1]
        
        # Calculate MSE for each component
        mean_mse = self.mse_loss(mean_preds, mean_targets).mean()
        std_mse = self.mse_loss(std_preds, std_targets).mean()
        
        # Calculate MAE for each component
        mean_mae = self.mae_loss(mean_preds, mean_targets).mean()
        std_mae = self.mae_loss(std_preds, std_targets).mean()
        
        # Combine losses
        mean_loss = self.mse_weight * mean_mse + self.mae_weight * mean_mae
        std_loss = self.mse_weight * std_mse + self.mae_weight * std_mae
        
        # Weight the components
        total_loss = self.mean_weight * mean_loss + self.std_weight * std_loss
        
        return total_loss


class TextEmbeddingModel(nn.Module):
    def __init__(self, pretrained_model_name=MODEL):
        super(TextEmbeddingModel, self).__init__()
        self.encoder = AutoModel.from_pretrained(pretrained_model_name)
        self.hidden_size = self.encoder.config.hidden_size
        print(f"Loaded model: {pretrained_model_name} with hidden size: {self.hidden_size}")
        
    def forward(self, input_ids, attention_mask):
        # For E5 models, it's often better to use the mean pooling of last_hidden_state
        # rather than just the CLS token
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        
        # Use mean pooling for better representation
        last_hidden_state = outputs.last_hidden_state
        # Create attention mask in proper shape [batch_size, seq_length, 1]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        # Sum the embeddings weighted by attention mask
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        # Sum the mask to get actual token count (excluding padding)
        sum_mask = input_mask_expanded.sum(1)
        # Avoid division by zero
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        # Get mean embedding
        mean_embeddings = sum_embeddings / sum_mask
        
        return mean_embeddings


class RegressionPredictor(nn.Module):
    def __init__(self, num_models, hidden_size=256, model_emb_dim=32):  # Increased hidden size
        super(RegressionPredictor, self).__init__()
        self.text_encoder = TextEmbeddingModel()
        # Get the actual embedding size correctly
        self.text_embedding_size = self.text_encoder.hidden_size
        self.model_emb_dim = model_emb_dim

        # Project one-hot model encoding into a dense vector
        self.model_embedding_layer = nn.Linear(num_models, model_emb_dim)
        
        # Deeper network with more capacity
        self.fc1 = nn.Linear(self.text_embedding_size + model_emb_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)  # Extra layer
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, 2)  # Output mean and std
        self.relu = nn.ReLU()
    
    def forward(self, input_ids, attention_mask, model_enc):
        text_embedding = self.text_encoder(input_ids, attention_mask)
        model_embedding = self.model_embedding_layer(model_enc)

        combined = torch.cat((text_embedding, model_embedding), dim=1)
        
        x = self.relu(self.fc1(combined))
        x = self.relu(self.fc2(x))  # Extra layer
        x = self.relu(self.fc3(x))
        output = self.fc4(x)
        
        return output

    

class ClassificationPredictor(nn.Module):
    def __init__(self, num_models, num_classes, hidden_size=128):
        super(ClassificationPredictor, self).__init__()
        self.text_encoder = TextEmbeddingModel()
        # Get the actual embedding size correctly
        self.text_embedding_size = self.text_encoder.hidden_size
        
        # Use the correct embedding size
        self.fc1 = nn.Linear(self.text_embedding_size + num_models, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, input_ids, attention_mask, model_enc):
        text_embedding = self.text_encoder(input_ids, attention_mask)
        combined = torch.cat((text_embedding, model_enc), dim=1)
        
        x = self.dropout(self.relu(self.fc1(combined)))
        x = self.dropout(self.relu(self.fc2(x)))
        logits = self.fc3(x)
        
        return logits

# Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler=None, num_epochs=5, task='regression'):
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    metrics = {'train_loss': [],
               'val_loss': [],
               'epoch': [],
               'learning_rates': []}  # Track learning rates
    
    if task == 'regression':
        metrics.update({
            'train_mse_mean': [], 'train_mae_mean': [],
            'val_mse_mean': [], 'val_mae_mean': [],
            'train_mse_std': [], 'train_mae_std': [],
            'val_mse_std': [], 'val_mae_std': []
            })
    else:  # classification
        metrics.update({
            'train_accuracy': [], 'train_f1': [],
            'val_accuracy': [], 'val_f1': []
            })
    
    # Get scheduler class name if it exists
    scheduler_type = None
    if scheduler is not None:
        scheduler_type = scheduler.__class__.__name__
    
    for epoch in range(num_epochs):
        t_start = time.time()
        model.train()
        train_loss = 0.0
        
        all_preds = []
        all_targets = []
        
        # Track current learning rates for each parameter group
        current_lr = [param_group['lr'] for param_group in optimizer.param_groups]
        metrics['learning_rates'].append(current_lr)
        print(f"Learning rates for epoch {epoch+1}: {current_lr}")
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            model_tensor = batch['model'].to(device)
            targets = batch['target'].to(device)
            
            # Forward pass
            outputs = model(input_ids, attention_mask, model_tensor)
            
            # Loss calculation
            loss = criterion(outputs, targets)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Step the warmup/decay scheduler if it exists and is a per-batch scheduler
            if scheduler_type in ['LambdaLR', 'OneCycleLR', 'LinearLR']:
                scheduler.step()
            
            train_loss += loss.item()
            
            # Store predictions and targets for metrics
            if task == 'regression':
                all_preds.extend(outputs.detach().cpu().numpy())
                all_targets.extend(targets.detach().cpu().numpy())
            else:  # classification
                _, predicted = torch.max(outputs, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        # Calculate average training loss
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                model_tensor = batch['model'].to(device)
                targets = batch['target'].to(device)
                
                # Forward pass
                outputs = model(input_ids, attention_mask, model_tensor)
                
                # Loss calculation
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                
                # Store predictions and targets for metrics
                if task == 'regression':
                    val_preds.extend(outputs.detach().cpu().numpy())
                    val_targets.extend(targets.detach().cpu().numpy())
                else:  # classification
                    _, predicted = torch.max(outputs, 1)
                    val_preds.extend(predicted.cpu().numpy())
                    val_targets.extend(targets.cpu().numpy())
        
        # Calculate average validation loss
        avg_val_loss = val_loss / len(val_loader)
        
        # Store losses for plotting
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        # Calculate and store metrics
        metrics['train_loss'].append(avg_train_loss)
        metrics['val_loss'].append(avg_val_loss)
        metrics['epoch'].append(epoch + 1)
        
        t_end = time.time() - t_start

        if task == 'regression':
            # Convert lists to numpy arrays for metric calculation
            all_preds_np = np.array(all_preds)
            all_targets_np = np.array(all_targets)
            val_preds_np = np.array(val_preds)
            val_targets_np = np.array(val_targets)
            
            # Denormalize predictions and targets for train set
            train_dataset = train_loader.dataset
            
            # Denormalize mean predictions for train
            denorm_train_preds_mean = train_dataset.denormalize_mean(all_preds_np[:, 0])
            denorm_train_targets_mean = train_dataset.denormalize_mean(all_targets_np[:, 0])
            
            # Denormalize std predictions for train
            denorm_train_preds_std = train_dataset.denormalize_std(all_preds_np[:, 1])
            denorm_train_targets_std = train_dataset.denormalize_std(all_targets_np[:, 1])
            
            # Denormalize predictions and targets for validation set
            val_dataset = val_loader.dataset
            
            # Denormalize mean predictions for validation
            denorm_val_preds_mean = val_dataset.denormalize_mean(val_preds_np[:, 0])
            denorm_val_targets_mean = val_dataset.denormalize_mean(val_targets_np[:, 0])
            
            # Denormalize std predictions for validation
            denorm_val_preds_std = val_dataset.denormalize_std(val_preds_np[:, 1])
            denorm_val_targets_std = val_dataset.denormalize_std(val_targets_np[:, 1])
            
            # Calculate MSE and MAE for denormalized mean predictions
            train_mse_mean = mean_squared_error(denorm_train_targets_mean, denorm_train_preds_mean)
            train_mae_mean = mean_absolute_error(denorm_train_targets_mean, denorm_train_preds_mean)
            val_mse_mean = mean_squared_error(denorm_val_targets_mean, denorm_val_preds_mean)
            val_mae_mean = mean_absolute_error(denorm_val_targets_mean, denorm_val_preds_mean)
            
            # Calculate MSE and MAE for denormalized std predictions
            train_mse_std = mean_squared_error(denorm_train_targets_std, denorm_train_preds_std)
            train_mae_std = mean_absolute_error(denorm_train_targets_std, denorm_train_preds_std)
            val_mse_std = mean_squared_error(denorm_val_targets_std, denorm_val_preds_std)
            val_mae_std = mean_absolute_error(denorm_val_targets_std, denorm_val_preds_std)
            
            # Store metrics for both mean and std predictions
            metrics['train_mse_mean'].append(train_mse_mean)
            metrics['train_mae_mean'].append(train_mae_mean)
            metrics['val_mse_mean'].append(val_mse_mean)
            metrics['val_mae_mean'].append(val_mae_mean)
            
            metrics['train_mse_std'].append(train_mse_std)
            metrics['train_mae_std'].append(train_mae_std)
            metrics['val_mse_std'].append(val_mse_std)
            metrics['val_mae_std'].append(val_mae_std)
            
            print(f"Epoch {epoch+1}/{num_epochs} - "
                  f"Train Loss: {avg_train_loss:.4f}, "
                  f"Train MSE Mean: {train_mse_mean:.4f}, "
                  f"Train MAE Mean: {train_mae_mean:.4f}, "
                  f"Train MSE Std: {train_mse_std:.4f}, "
                  f"Train MAE Std: {train_mae_std:.4f}, "
                  f"Val Loss: {avg_val_loss:.4f}, "
                  f"Val MSE Mean: {val_mse_mean:.4f}, "
                  f"Val MAE Mean: {val_mae_mean:.4f}, "
                  f"Val MSE Std: {val_mse_std:.4f}, "
                  f"Val MAE Std: {val_mae_std:.4f}, Time: {t_end:.2f}s")
            
        else:  # classification
            train_acc = accuracy_score(all_targets, all_preds)
            train_f1 = f1_score(all_targets, all_preds, average='weighted')
            val_acc = accuracy_score(val_targets, val_preds)
            val_f1 = f1_score(val_targets, val_preds, average='weighted')
            
            metrics['train_accuracy'].append(train_acc)
            metrics['train_f1'].append(train_f1)
            metrics['val_accuracy'].append(val_acc)
            metrics['val_f1'].append(val_f1)
            
            print(f"Epoch {epoch+1}/{num_epochs} - "
                  f"Train Loss: {avg_train_loss:.4f}, "
                  f"Train Acc: {train_acc:.4f}, "
                  f"Val Loss: {avg_val_loss:.4f}, "
                  f"Val Acc: {val_acc:.4f}, Time: {t_end:.2f}s")
        
        # Step the epoch-based schedulers
        if scheduler is not None:
            # ReduceLROnPlateau is a special case that requires validation loss
            if scheduler_type == 'ReduceLROnPlateau':
                scheduler.step(avg_val_loss)
            # Others like StepLR, CosineAnnealingLR are stepped at epoch end
            elif scheduler_type not in ['LambdaLR', 'OneCycleLR', 'LinearLR']:
                scheduler.step()
        
        # Save the best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            print(f"Saving best model with validation loss: {best_val_loss:.4f}")
            torch.save(model.state_dict(), os.path.join(MODELS_DIR, f"best_{task}_model.pt"))
    
    return model, metrics


# Evaluate the model on test set
def evaluate_model(model, test_loader, criterion, task='regression'):
    model.eval()
    test_loss = 0.0
    
    test_preds = []
    test_targets = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            model_tensor = batch['model'].to(device)
            targets = batch['target'].to(device)
            
            # Forward pass
            outputs = model(input_ids, attention_mask, model_tensor)
            
            # Loss calculation
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            
            # Store predictions and targets for metrics
            if task == 'regression':
                test_preds.extend(outputs.detach().cpu().numpy())
                test_targets.extend(targets.detach().cpu().numpy())
            else:  # classification
                _, predicted = torch.max(outputs, 1)
                test_preds.extend(predicted.cpu().numpy())
                test_targets.extend(targets.cpu().numpy())
    
    # Calculate average test loss
    avg_test_loss = test_loss / len(test_loader)
    
    metrics = {'test_loss': avg_test_loss}
    
    if task == 'regression':
        # Convert lists to numpy arrays for metric calculation
        test_preds_np = np.array(test_preds)
        test_targets_np = np.array(test_targets)
        
        # Denormalize predictions and targets
        dataset = test_loader.dataset
        
        # Denormalize mean predictions
        denorm_preds_mean = dataset.denormalize_mean(test_preds_np[:, 0])
        denorm_targets_mean = dataset.denormalize_mean(test_targets_np[:, 0])
        
        # Denormalize std predictions
        denorm_preds_std = dataset.denormalize_std(test_preds_np[:, 1])
        denorm_targets_std = dataset.denormalize_std(test_targets_np[:, 1])
        
        # Calculate MSE and MAE for denormalized mean predictions
        test_mse_mean = mean_squared_error(denorm_targets_mean, denorm_preds_mean)
        test_mae_mean = mean_absolute_error(denorm_targets_mean, denorm_preds_mean)
        
        # Calculate MSE and MAE for denormalized std predictions
        test_mse_std = mean_squared_error(denorm_targets_std, denorm_preds_std)
        test_mae_std = mean_absolute_error(denorm_targets_std, denorm_preds_std)
        
        metrics.update({
            'test_mse_mean': test_mse_mean, 'test_mae_mean': test_mae_mean,
            'test_mse_std': test_mse_std, 'test_mae_std': test_mae_std
        })
        
        print(f"Test results - "
              f"Loss: {avg_test_loss:.4f}, "
              f"MSE (mean): {test_mse_mean:.4f}, "
              f"MAE (mean): {test_mae_mean:.4f}, "
              f"MSE (std): {test_mse_std:.4f}, "
              f"MAE (std): {test_mae_std:.4f}")
        
    else:  # classification
        test_acc = accuracy_score(test_targets, test_preds)
        test_f1 = f1_score(test_targets, test_preds, average='weighted')
        
        metrics.update({
            'test_accuracy': test_acc, 'test_f1': test_f1
        })
        
        print(f"Test results - "
              f"Loss: {avg_test_loss:.4f}, "
              f"Accuracy: {test_acc:.4f}, "
              f"F1 Score: {test_f1:.4f}")
    
    return metrics


# Update the plot_metrics function to include MAE visualization
def plot_metrics(metrics, task='regression', save_path=None):
    # Create the figure
    if task == 'regression':
        fig = plt.figure(figsize=(14, 20))  # Increased height to accommodate 5 subplots
        
        # Plot 1: Training and validation loss
        ax1 = fig.add_subplot(5, 1, 1)
        ax1.plot(metrics['epoch'], metrics['train_loss'], 'b-', label='Train Loss')
        ax1.plot(metrics['epoch'], metrics['val_loss'], 'r-', label='Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title(f'{task.capitalize()} Model - Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Plot 2: MSE for mean predictions
        ax2 = fig.add_subplot(5, 1, 2)
        ax2.plot(metrics['epoch'], metrics['train_mse_mean'], 'g-', label='Train MSE')
        ax2.plot(metrics['epoch'], metrics['val_mse_mean'], 'm-', label='Validation MSE')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('MSE')
        ax2.set_title('Mean Squared Error (for Mean Prediction)')
        ax2.legend()
        ax2.grid(True)
        
        # Plot 3: MAE for mean predictions
        ax3 = fig.add_subplot(5, 1, 3)
        ax3.plot(metrics['epoch'], metrics['train_mae_mean'], 'c-', label='Train MAE')
        ax3.plot(metrics['epoch'], metrics['val_mae_mean'], 'y-', label='Validation MAE')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('MAE')
        ax3.set_title('Mean Absolute Error (for Mean Prediction)')
        ax3.legend()
        ax3.grid(True)
        
        # Plot 4: MSE for std predictions
        ax4 = fig.add_subplot(5, 1, 4)
        ax4.plot(metrics['epoch'], metrics['train_mse_std'], 'g-', label='Train MSE')
        ax4.plot(metrics['epoch'], metrics['val_mse_std'], 'm-', label='Validation MSE')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('MSE')
        ax4.set_title('Mean Squared Error (for Std Prediction)')
        ax4.legend()
        ax4.grid(True)
        
        # Plot 5: MAE for std predictions
        ax5 = fig.add_subplot(5, 1, 5)
        ax5.plot(metrics['epoch'], metrics['train_mae_std'], 'c-', label='Train MAE')
        ax5.plot(metrics['epoch'], metrics['val_mae_std'], 'y-', label='Validation MAE')
        ax5.set_xlabel('Epoch')
        ax5.set_ylabel('MAE')
        ax5.set_title('Mean Absolute Error (for Std Prediction)')
        ax5.legend()
        ax5.grid(True)
        
    else:  # classification
        fig = plt.figure(figsize=(14, 14))
        
        # Plot 1: Training and validation loss
        ax1 = fig.add_subplot(3, 1, 1)
        ax1.plot(metrics['epoch'], metrics['train_loss'], 'b-', label='Train Loss')
        ax1.plot(metrics['epoch'], metrics['val_loss'], 'r-', label='Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title(f'{task.capitalize()} Model - Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Plot 2: Accuracy for train and validation
        ax2 = fig.add_subplot(3, 1, 2)
        ax2.plot(metrics['epoch'], metrics['train_accuracy'], 'g-', label='Train Accuracy')
        ax2.plot(metrics['epoch'], metrics['val_accuracy'], 'm-', label='Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Classification Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        # Plot 3: F1 Score for train and validation
        ax3 = fig.add_subplot(3, 1, 3)
        ax3.plot(metrics['epoch'], metrics['train_f1'], 'c-', label='Train F1')
        ax3.plot(metrics['epoch'], metrics['val_f1'], 'y-', label='Validation F1')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('F1 Score')
        ax3.set_title('F1 Score (Weighted)')
        ax3.legend()
        ax3.grid(True)
    
    plt.tight_layout()
    
    # Save the figure if a path is provided
    if save_path:
        plt.savefig(os.path.join(save_path, f'{task}_training_metrics.png'))
    
    # Create learning rate figure
    lr_fig = plt.figure(figsize=(10, 6))
    for i, lr_group in enumerate(zip(*metrics['learning_rates'])):
        plt.plot(metrics['epoch'], lr_group, label=f'Group {i+1}')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')  # Log scale often helps visualize learning rate schedules
    
    # Save the learning rate figure if a path is provided
    if save_path:
        plt.savefig(os.path.join(save_path, f'{task}_learning_rates.png'))
    
    # Return the figures
    return fig, lr_fig

# Metric analysis function for classification
def analyze_classification_metrics(y_true, y_pred, dataset):
    """
    Analyze classification metrics and return detailed statistics
    
    Args:
        y_true: True class labels
        y_pred: Predicted class labels
        dataset: The dataset containing binning information
    
    Returns:
        Dictionary with various classification metrics
    """
    bin_size = dataset.bin_size
    max_bin_threshold = dataset.max_bin_threshold
    num_bins_below_threshold = max_bin_threshold // bin_size
    
    # Create class labels
    class_labels = []
    for i in range(num_bins_below_threshold):
        lower = i * bin_size
        upper = (i + 1) * bin_size
        class_labels.append(f"{lower}-{upper}")
    class_labels.append(f"≥{max_bin_threshold}")
    
    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    cm = confusion_matrix(y_true, y_pred)
    
    # Class distribution
    class_counts = np.bincount(y_true, minlength=len(class_labels))
    
    # Per-class accuracy
    class_accuracy = {}
    for i in range(len(class_labels)):
        if np.sum(y_true == i) > 0:
            class_accuracy[class_labels[i]] = np.mean(np.array(y_pred)[np.array(y_true) == i] == i)
    
    # Precision and recall per class
    precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
    
    # Create metrics dictionary
    metrics = {
        'accuracy': accuracy,
        'f1_score': f1,
        'confusion_matrix': cm,
        'class_labels': class_labels,
        'class_counts': class_counts,
        'class_accuracy': class_accuracy,
        'precision_per_class': dict(zip(class_labels, precision_per_class)),
        'recall_per_class': dict(zip(class_labels, recall_per_class)),
    }
    
    return metrics


# Plotting function for classification results
def plot_classification_results(metrics, save_path=None):
    """
    Create visualizations for classification results
    
    Args:
        metrics: Dictionary containing classification metrics from analyze_classification_metrics
        save_path: Optional path to save the plots
    
    Returns:
        List of figure objects
    """
    figures = []
    
    # 1. Confusion Matrix
    cm_fig = plt.figure(figsize=(12, 10))
    sns.heatmap(
        metrics['confusion_matrix'], 
        annot=True, 
        fmt='d', 
        cmap='Blues', 
        xticklabels=metrics['class_labels'], 
        yticklabels=metrics['class_labels']
    )
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix for P99 Classification')
    if save_path:
        plt.savefig(os.path.join(save_path, 'classification_confusion_matrix.png'))
    figures.append(cm_fig)
    
    # 2. Class Distribution
    dist_fig = plt.figure(figsize=(14, 6))
    plt.bar(metrics['class_labels'], metrics['class_counts'])
    plt.xlabel('Token Range')
    plt.ylabel('Count')
    plt.title('Distribution of P99 Classes')
    plt.xticks(rotation=45)
    if save_path:
        plt.savefig(os.path.join(save_path, 'classification_class_distribution.png'))
    figures.append(dist_fig)
    
    # 3. Per-class Accuracy
    acc_fig = plt.figure(figsize=(14, 6))
    plt.bar(list(metrics['class_accuracy'].keys()), list(metrics['class_accuracy'].values()))
    plt.xlabel('Token Range')
    plt.ylabel('Accuracy')
    plt.title('Classification Accuracy by Token Range')
    plt.xticks(rotation=45)
    if save_path:
        plt.savefig(os.path.join(save_path, 'classification_class_accuracy.png'))
    figures.append(acc_fig)
    
    # 4. Precision and Recall
    pr_fig = plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.bar(list(metrics['precision_per_class'].keys()), list(metrics['precision_per_class'].values()))
    plt.xlabel('Token Range')
    plt.ylabel('Precision')
    plt.title('Precision by Token Range')
    plt.xticks(rotation=90)
    
    plt.subplot(1, 2, 2)
    plt.bar(list(metrics['recall_per_class'].keys()), list(metrics['recall_per_class'].values()))
    plt.xlabel('Token Range')
    plt.ylabel('Recall')
    plt.title('Recall by Token Range')
    plt.xticks(rotation=90)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(os.path.join(save_path, 'classification_precision_recall.png'))
    figures.append(pr_fig)
    
    return figures


# Function to run the analysis
def run_classification_analysis(model, test_loader, criterion, save_path=None):
    """
    Run a comprehensive analysis of classification results
    
    Args:
        model: Trained classification model
        test_loader: DataLoader for test dataset
        criterion: Loss function
        save_path: Optional path to save results and plots
    
    Returns:
        Dictionary of metrics and list of figures
    """
    model.eval()
    all_preds = []
    all_targets = []
    test_loss = 0.0
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Analyzing classification results"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            model_tensor = batch['model'].to(device)
            targets = batch['target'].to(device)
            
            # Forward pass
            outputs = model(input_ids, attention_mask, model_tensor)
            
            # Loss calculation
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    # Calculate average test loss
    avg_test_loss = test_loss / len(test_loader)
    
    # Get detailed metrics
    metrics = analyze_classification_metrics(
        np.array(all_targets), 
        np.array(all_preds), 
        test_loader.dataset
    )
    
    # Add loss to metrics
    metrics['test_loss'] = avg_test_loss
    
    # Create visualizations
    figures = plot_classification_results(metrics, save_path)
    
    # Print summary
    print(f"Classification Analysis Summary:")
    print(f"  Loss: {avg_test_loss:.4f}")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  F1 Score (weighted): {metrics['f1_score']:.4f}")
    
    # Save metrics to file if path provided
    if save_path:
        # Convert numpy arrays to lists for JSON serialization
        serializable_metrics = {k: v if not isinstance(v, np.ndarray) else v.tolist() 
                               for k, v in metrics.items()}
        
        with open(os.path.join(save_path, 'classification_analysis.json'), 'w') as f:
            json.dump(serializable_metrics, f, indent=2)
    
    return metrics, figures


# Main execution function
def main():
    # Load data
    print("Loading data...")
    train_data, val_data, test_data = load_data(DATA_DIR)

    print(f"Training data size: {len(train_data)}")
    print(f"Validation data size: {len(val_data)}")
    print(f"Test data size: {len(test_data)}")
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    tokenizer.pad_token = tokenizer.eos_token if tokenizer.pad_token is None else tokenizer.pad_token
    
    # Setup for regression task
    print("\nSetting up regression predictor...")
    train_dataset_reg = ResponseLengthDataset(train_data, tokenizer, task='regression')
    val_dataset_reg = ResponseLengthDataset(val_data, tokenizer, task='regression')
    test_dataset_reg = ResponseLengthDataset(test_data, tokenizer, task='regression')
    
    batch_size = 16
    train_loader_reg = DataLoader(train_dataset_reg, batch_size=batch_size, shuffle=True)
    val_loader_reg = DataLoader(val_dataset_reg, batch_size=batch_size)
    test_loader_reg = DataLoader(test_dataset_reg, batch_size=batch_size)
    
    # Determine number of models
    num_models = len(train_dataset_reg.models)
    
    # Initialize regression model
    regression_model = RegressionPredictor(num_models=num_models, hidden_size=HIDDEN_SIZE).to(device)
    
    # Save model architecture info
    with open(os.path.join(MODELS_DIR, 'regression_model_info.json'), 'w') as f:
        json.dump({
                'num_models': num_models, 'model_encoder': train_dataset_reg.model_encoder,
                'models': train_dataset_reg.models,
                # Also save normalization parameters
                'mean_target_mean': float(train_dataset_reg.mean_target_mean), 'std_target_mean': float(train_dataset_reg.std_target_mean),
                'mean_target_std': float(train_dataset_reg.mean_target_std), 'std_target_std': float(train_dataset_reg.std_target_std)
            }, f, indent=2)
    
    # Group parameters by components for different learning rates
    encoder_params = list(regression_model.text_encoder.parameters())
    classifier_params = list(regression_model.fc1.parameters()) + \
                         list(regression_model.fc2.parameters()) + \
                         list(regression_model.fc3.parameters())
    
    # Create optimizer with parameter groups - lower LR for pretrained encoder
    regression_optimizer = optim.AdamW([
        {'params': encoder_params, 'lr': 1e-5},  # Lower rate for pretrained encoder
        {'params': classifier_params, 'lr': 5e-5}  # Higher rate for new layers
    ])
    
    # Training parameters
    regression_criterion = WeightedRegressionLoss(
                mean_weight=0.7,  
                std_weight=0.3,   
                mse_weight=0.3,   
                mae_weight=0.7   
    )
    
    # Calculate number of training steps for warmup
    num_epochs = 5
    num_training_steps = len(train_loader_reg) * num_epochs
    num_warmup_steps = int(0.1 * num_training_steps)  # 10% of total steps for warmup
    
    # Create scheduler with warmup
    regression_scheduler = get_scheduler(
            name="linear",  # linear schedule with warmup
            optimizer=regression_optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps*2
    )
    
    # Train regression model with scheduler
    print("\nTraining regression model...")
    trained_reg_model, reg_metrics = train_model(
        regression_model, 
        train_loader_reg, 
        val_loader_reg, 
        regression_criterion, 
        regression_optimizer,
        scheduler=regression_scheduler,
        num_epochs=num_epochs, 
        task='regression'
    )
    
    # Evaluate regression model
    print("\nEvaluating regression model on test set...")
    reg_test_metrics = evaluate_model(
        trained_reg_model,
        test_loader_reg,
        regression_criterion,
        task='regression'
    )
    
    # Save regression metrics
    with open(os.path.join(METRICS_DIR, 'regression_training_metrics.pkl'), 'wb') as f:
        pickle.dump(reg_metrics, f)
    
    with open(os.path.join(METRICS_DIR, 'regression_test_metrics.json'), 'w') as f:
        json.dump(reg_test_metrics, f)
    
    # Plot regression metrics
    plot_metrics(reg_metrics, task='regression')
    
    # Setup for classification task
    print("\nSetting up classification predictor...")
    
    # Define bin size and max threshold for classification
    bin_size = 50
    max_bin_threshold = 2000  # New parameter for max bin threshold
    
    # Find max p99 value for logging purposes
    max_p99 = 0
    for dataset in [train_data, val_data, test_data]:
        for item in dataset:
            p99 = item['output_percentiles']['p99']
            max_p99 = max(max_p99, p99)
    
    print(f"Maximum p99 value in dataset: {max_p99}")
    
    # Create datasets with custom binning
    train_dataset_cls = ResponseLengthDataset(
        train_data, tokenizer, task='classification', 
        bin_size=bin_size, max_bin_threshold=max_bin_threshold
    )
    val_dataset_cls = ResponseLengthDataset(
        val_data, tokenizer, task='classification', 
        bin_size=bin_size, max_bin_threshold=max_bin_threshold
    )
    test_dataset_cls = ResponseLengthDataset(
        test_data, tokenizer, task='classification', 
        bin_size=bin_size, max_bin_threshold=max_bin_threshold
    )
    
    # Number of classes is now determined within the dataset
    num_classes = train_dataset_cls.num_classes
    print(f"Using {num_classes} classes for classification")
    
    train_loader_cls = DataLoader(train_dataset_cls, batch_size=batch_size, shuffle=True)
    val_loader_cls = DataLoader(val_dataset_cls, batch_size=batch_size)
    test_loader_cls = DataLoader(test_dataset_cls, batch_size=batch_size)
    
    # Initialize classification model
    classification_model = ClassificationPredictor(num_models=num_models, num_classes=num_classes, hidden_size=HIDDEN_SIZE).to(device)
    
    with open(os.path.join(MODELS_DIR, 'classification_model_info.json'), 'w') as f:
        json.dump({
            'num_models': num_models,
            'num_classes': num_classes,
            'bin_size': bin_size,
            'max_bin_threshold': max_bin_threshold,
            'model_encoder': train_dataset_cls.model_encoder,
            'models': train_dataset_cls.models
        }, f, indent=2)
    
    # Group parameters for classification model too
    cls_encoder_params = list(classification_model.text_encoder.parameters())
    cls_classifier_params = list(classification_model.fc1.parameters()) + \
                           list(classification_model.fc2.parameters()) + \
                           list(classification_model.fc3.parameters())
    
    # Create optimizer with parameter groups for classification
    classification_optimizer = optim.AdamW([
        {'params': cls_encoder_params, 'lr': 1e-5},  # Lower rate for pretrained encoder
        {'params': cls_classifier_params, 'lr': 5e-5}  # Higher rate for new layers
    ])
    
    # Training parameters
    classification_criterion = nn.CrossEntropyLoss()
    
    # Calculate number of training steps for warmup
    cls_num_training_steps = len(train_loader_cls) * num_epochs
    cls_num_warmup_steps = int(0.1 * cls_num_training_steps)  # 10% of total steps for warmup
    
    # Create scheduler with warmup
    classification_scheduler = get_scheduler(
            name="linear",  # linear schedule with warmup
            optimizer=classification_optimizer,
            num_warmup_steps=cls_num_warmup_steps,
            num_training_steps=cls_num_training_steps*2
    )
    
    # Train classification model with scheduler
    print("\nTraining classification model...")
    trained_cls_model, cls_metrics = train_model(
        classification_model, 
        train_loader_cls, 
        val_loader_cls, 
        classification_criterion, 
        classification_optimizer,
        scheduler=classification_scheduler,
        num_epochs=num_epochs,
        task='classification'
    )
    
    # Standard evaluation
    print("\nBasic evaluation of classification model on test set...")
    cls_test_metrics = evaluate_model(
        trained_cls_model,
        test_loader_cls,
        classification_criterion,
        task='classification'
    )
    
    # Save standard metrics
    with open(os.path.join(METRICS_DIR, 'classification_training_metrics.pkl'), 'wb') as f:
        pickle.dump(cls_metrics, f)
    
    with open(os.path.join(METRICS_DIR, 'classification_test_metrics.json'), 'w') as f:
        json.dump(cls_test_metrics, f)
    
    # Plot standard training metrics
    plot_metrics(cls_metrics, task='classification', save_path=METRICS_DIR)
    
    # Run detailed classification analysis
    print("\nRunning detailed classification analysis...")
    detailed_cls_metrics, cls_figures = run_classification_analysis(
        trained_cls_model,
        test_loader_cls,
        classification_criterion,
        save_path=METRICS_DIR
    )
    
    print("\nTraining and evaluation complete!")
    print(f"Models saved to: {MODELS_DIR}")
    print(f"Metrics saved to: {METRICS_DIR}")

if __name__ == "__main__":
    main()
    

#     # Example usage of prediction
#     print("\nExample predictions:")
    
#     example_prompt = "Write a comprehensive essay about the impact of artificial intelligence on society."
#     example_model = "gpt-3.5-turbo"
    
#     reg_prediction = predict_response_length(example_prompt, example_model, task='regression')
#     print(f"Regression prediction: Expected response mean length: {reg_prediction['predicted_mean']:.2f} tokens, "
#           f"Expected std: {reg_prediction['predicted_std']:.2f} tokens")
    
#     cls_prediction = predict_response_length(example_prompt, example_model, task='classification')
#     print(f"Classification prediction: P99 token length class: {cls_prediction['predicted_p99_class']}, "
#           f"Range: {cls_prediction['predicted_p99_range']}")


# # Inference functions
# def load_regression_model(model_path):
#     # Load model info
#     with open(os.path.join(MODELS_DIR, 'regression_model_info.json'), 'r') as f:
#         model_info = json.load(f)
    
#     # Initialize model
#     model = RegressionPredictor(768, model_info['num_models']).to(device)
#     model.load_state_dict(torch.load(model_path))
#     model.eval()
    
#     return model, model_info

# def load_classification_model(model_path):
#     # Load model info
#     with open(os.path.join(MODELS_DIR, 'classification_model_info.json'), 'r') as f:
#         model_info = json.load(f)
    
#     # Initialize model
#     model = ClassificationPredictor(768, model_info['num_models'], model_info['num_classes']).to(device)
#     model.load_state_dict(torch.load(model_path))
#     model.eval()
    
#     return model, model_info

# def predict_response_length(prompt_text, model_name, task='regression'):
#     # Load appropriate model
#     if task == 'regression':
#         model_path = os.path.join(MODELS_DIR, 'best_regression_model.pt')
#         model, model_info = load_regression_model(model_path)
#         tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
#     else:  # classification
#         model_path = os.path.join(MODELS_DIR, 'best_classification_model.pt')
#         model, model_info = load_classification_model(model_path)
#         tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    
#     # Check if model_name exists in the training data
#     model_encoder = model_info['model_encoder']
#     if model_name not in model_encoder:
#         print(f"Warning: Model '{model_name}' not found in training data. Using fallback.")
#         model_name = list(model_encoder.keys())[0]
    
#     # Prepare inputs
#     encoding = tokenizer(
#         prompt_text,
#         max_length=512,
#         padding='max_length',
#         truncation=True,
#         return_tensors='pt'
#     )
    
#     input_ids = encoding['input_ids'].to(device)
#     attention_mask = encoding['attention_mask'].to(device)
    
#     # Prepare model one-hot encoding
#     model_tensor = torch.zeros(len(model_encoder)).to(device)
#     model_tensor[model_encoder[model_name]] = 1
#     model_tensor = model_tensor.unsqueeze(0)
    
#     # Prediction
#     with torch.no_grad():
#         output = model(input_ids, attention_mask, model_tensor)
        
#         if task == 'regression':
#             mean_pred, std_pred = output[0].cpu().numpy()
#             return {
#                 'predicted_mean': mean_pred,
#                 'predicted_std': std_pred
#             }
#         else:  # classification
#             _, predicted_class = torch.max(output, 1)
#             bin_size = model_info['bin_size']
#             lower_bound = predicted_class.item() * bin_size
#             upper_bound = (predicted_class.item() + 1) * bin_size
#             return {
#                 'predicted_p99_class': predicted_class.item(),
#                 'predicted_p99_range': f"{lower_bound}-{upper_bound} tokens"
#             }

    