import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LinearLR, CosineAnnealingLR, SequentialLR
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, f1_score, confusion_matrix, precision_score, recall_score
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
    def __init__(self, data, tokenizer, max_length=512, task='regression', bin_size=50, max_bin_threshold=2000, percentile=90):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.task = task
        self.bin_size = bin_size
        self.max_bin_threshold = max_bin_threshold  # New parameter for max bin threshold
        self.percentile = percentile    
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
            # Count distribution of percentile values
            if f"p{percentile}" in ["p25", "p50", "p75", "p99"]:
                p_values = [item['output_percentiles'][percentile] for item in data]
            else:
                p_values = [np.percentile(item['output_sizes'], percentile) for item in data]

            below_threshold = sum(1 for p in p_values if p < max_bin_threshold)
            above_threshold = sum(1 for p in p_values if p >= max_bin_threshold)
            
            # Calculate number of classes (bins below threshold + 1 class for above threshold)
            self.num_bins_below_threshold = max_bin_threshold // bin_size
            self.num_classes = self.num_bins_below_threshold + 1
            
            print(f"Classification binning stats:")
            print(f"  Values below {max_bin_threshold}: {below_threshold} ({below_threshold/len(p_values):.1%})")
            print(f"  Values above {max_bin_threshold}: {above_threshold} ({above_threshold/len(p_values):.1%})")
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
            if f"p{self.percentile}" in ["p25", "p50", "p75", "p99"]:
                p_val = item['output_percentiles'][self.percentile]
            else:
                p_val = np.percentile(item['output_sizes'], self.percentile)
            
            # Apply custom binning logic
            if p_val < self.max_bin_threshold:
                # For values below threshold, use normal binning
                p_val_class = int(p_val // self.bin_size)
            else:
                # For values above threshold, assign to the last class
                p_val_class = self.num_bins_below_threshold
                
            target = torch.tensor(p_val_class, dtype=torch.long)
            
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


# Define the text embedding model
class TextEmbeddingModel(nn.Module):
    def __init__(self, pretrained_model_name):
        super(TextEmbeddingModel, self).__init__()
        self.bert = AutoModel.from_pretrained(pretrained_model_name)
        self.hidden_size = self.bert.config.hidden_size
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Use the CLS token representation as the text embedding
        return outputs.last_hidden_state[:, 0, :]


class RegressionPredictor(nn.Module):
    def __init__(self, pretrained_model_name, num_models, hidden_size=256, model_emb_dim=32):
        super(RegressionPredictor, self).__init__()
        self.text_encoder = TextEmbeddingModel(pretrained_model_name)
        self.text_embedding_size = self.text_encoder.hidden_size

        self.model_embedding_layer = nn.Linear(num_models, model_emb_dim)
        self.fc1 = nn.Linear(self.text_embedding_size + model_emb_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, 2)  # [mean, std]
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, input_ids, attention_mask, model_enc):
        text_embedding = self.text_encoder(input_ids, attention_mask)
        model_embedding = self.model_embedding_layer(model_enc)
        combined = torch.cat((text_embedding, model_embedding), dim=1)

        x = self.dropout(self.relu(self.fc1(combined)))
        x = self.dropout(self.relu(self.fc2(x)))
        output = self.fc3(x)
        return output


class ClassificationPredictor(nn.Module):
    def __init__(self, pretrained_model_name, num_models, num_classes, hidden_size=256, model_emb_dim=32):
        super(ClassificationPredictor, self).__init__()
        self.text_encoder = TextEmbeddingModel(pretrained_model_name)
        self.text_embedding_size = self.text_encoder.hidden_size

        self.model_embedding_layer = nn.Linear(num_models, model_emb_dim)
        self.fc1 = nn.Linear(self.text_embedding_size + model_emb_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, input_ids, attention_mask, model_enc):
        text_embedding = self.text_encoder(input_ids, attention_mask)
        model_embedding = self.model_embedding_layer(model_enc)
        combined = torch.cat((text_embedding, model_embedding), dim=1)

        x = self.dropout(self.relu(self.fc1(combined)))
        x = self.dropout(self.relu(self.fc2(x)))
        logits = self.fc3(x)
        return logits


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
    num_epochs_recorded = len(metrics['epoch'])
    learning_rates = metrics.get('learning_rates', [])
    if len(learning_rates) > num_epochs_recorded:
        learning_rates = learning_rates[:num_epochs_recorded]
    elif len(learning_rates) < num_epochs_recorded:
        # Pad with last known learning rate
        if learning_rates:
            last_lr = learning_rates[-1]
            learning_rates += [last_lr] * (num_epochs_recorded - len(learning_rates))
        else:
            learning_rates = [[1e-5]] * num_epochs_recorded  # fallback default

    lr_fig = plt.figure(figsize=(10, 6))
    for i, lr_group in enumerate(zip(*learning_rates)):
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
def plot_classification_results(metrics, percentile, save_path=None):
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
    plt.title(f'Confusion Matrix for P{percentile} Classification')
    if save_path:
        plt.savefig(os.path.join(save_path, 'classification_confusion_matrix.png'))
    figures.append(cm_fig)
    
    # 2. Class Distribution
    dist_fig = plt.figure(figsize=(14, 6))
    plt.bar(metrics['class_labels'], metrics['class_counts'])
    plt.xlabel('Token Range')
    plt.ylabel('Count')
    plt.title(f'Distribution of P{percentile} Classes')
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
def run_classification_analysis(model, device, test_loader, criterion, save_path=None):
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
