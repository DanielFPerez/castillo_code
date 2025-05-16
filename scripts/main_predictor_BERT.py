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
import argparse

from transformers import get_linear_schedule_with_warmup
from transformers import get_scheduler
import math

from utils_bert import load_data, ResponseLengthDataset, WeightedRegressionLoss, RegressionPredictor, ClassificationPredictor, \
                        plot_metrics, analyze_classification_metrics, run_classification_analysis, plot_classification_results

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

MODEL = "distilbert/distilbert-base-multilingual-cased" # "distilbert-base-uncased"


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

    try:
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

            # Get scheduler class name if it exists
            scheduler_type = None if scheduler is None else scheduler.__class__.__name__
            
            for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")):
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
                
                # Step CosineAnnealingWarmRestarts after each batch
                if scheduler_type == 'CosineAnnealingWarmRestarts':
                    scheduler.step(epoch + batch_idx / len(train_loader))
                
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
            
            # Time for epoch
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
            
            # Save the best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                print(f"Saving best model with validation loss: {best_val_loss:.4f}")
                torch.save(model.state_dict(), os.path.join(MODELS_DIR, f"best_{task}_model.pt"))
    
    except KeyboardInterrupt:
        print("\n[!] Training interrupted by user. Saving intermediate results...")
        return model, metrics

    return model, metrics


# Evaluate the model on test set
def evaluate_model(model, test_loader, criterion, task='regression'):
    model.eval()
    test_loss = 0.0
    
    test_preds = []
    test_targets = []
    
    try:
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
    
    except KeyboardInterrupt:
        print("\n[!] Evaluation interrupted by user. Saving partial metrics...")
    
    
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


def run_regression_task(args):
    """Run regression task with given arguments"""
    global MODELS_DIR, METRICS_DIR, device
    
    # Load data
    print("Loading data...")
    train_data, val_data, test_data = load_data(args.data_src_dir)

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
    
    train_loader_reg = DataLoader(train_dataset_reg, batch_size=args.batch_size, shuffle=True)
    val_loader_reg = DataLoader(val_dataset_reg, batch_size=args.batch_size)
    test_loader_reg = DataLoader(test_dataset_reg, batch_size=args.batch_size)
    
    # Determine number of models
    num_models = len(train_dataset_reg.models)
    
    # Initialize regression model
    regression_model = RegressionPredictor(pretrained_model_name=MODEL, num_models=num_models, hidden_size=args.hidden_size).to(device)
    
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
                         list(regression_model.fc3.parameters()) + \
                        list(regression_model.model_embedding_layer.parameters())
    
    # Create optimizer with parameter groups - lower LR for pretrained encoder
    regression_optimizer = optim.AdamW([
        {'params': encoder_params, 'lr': 5e-5, 'weight_decay': 0.01},  # Lower rate for pretrained encoder
        {'params': classifier_params, 'lr': 1e-4, 'weight_decay': 0.01}  # Higher rate for new layers
    ])
    
    # Training parameters
    regression_criterion = WeightedRegressionLoss(
                mean_weight=0.6, std_weight=0.4,   
                mse_weight=0.6, mae_weight=0.4   
    )
    
    # Calculate number of training steps for warmup
    num_training_steps = len(train_loader_reg) * args.num_epochs
    num_warmup_steps = int(0.1 * num_training_steps)  # 10% of total steps for warmup
    
    # Create scheduler with warmup
    regression_scheduler = CosineAnnealingWarmRestarts(
        optimizer=regression_optimizer,
        T_0=len(train_loader_reg),  # Restart after 1 epoch initially
        T_mult=2,  # Double the restart period after each restart
        eta_min=1e-6  # Minimum learning rate
    )
    
    # Train regression model with scheduler
    print("\nTraining regression model...")
    trained_reg_model, reg_metrics = train_model(
        regression_model, train_loader_reg, val_loader_reg, regression_criterion, regression_optimizer,
        scheduler=regression_scheduler, num_epochs=args.num_epochs, 
        task='regression'
    )
    
    # Evaluate regression model
    print("\nEvaluating regression model on test set...")
    reg_test_metrics = evaluate_model(
        trained_reg_model, test_loader_reg,
        regression_criterion,
        task='regression'
    )
    
    # Save regression metrics
    with open(os.path.join(METRICS_DIR, 'regression_training_metrics.pkl'), 'wb') as f:
        pickle.dump(reg_metrics, f)
    
    with open(os.path.join(METRICS_DIR, 'regression_test_metrics.json'), 'w') as f:
        json.dump(reg_test_metrics, f, indent=2)
    
    # Plot regression metrics
    plot_metrics(reg_metrics, task='regression', save_path=METRICS_DIR)
    
    print("\nRegression task completed!")
    print(f"Models saved to: {MODELS_DIR}")
    print(f"Metrics saved to: {METRICS_DIR}")


def run_classification_task(args):
    """Run classification task with given arguments"""
    global MODELS_DIR, METRICS_DIR, device
    
    # Load data
    print("Loading data...")
    train_data, val_data, test_data = load_data(args.data_src_dir)

    print(f"Training data size: {len(train_data)}")
    print(f"Validation data size: {len(val_data)}")
    print(f"Test data size: {len(test_data)}")
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    tokenizer.pad_token = tokenizer.eos_token if tokenizer.pad_token is None else tokenizer.pad_token
    
    # Setup for classification task
    print("\nSetting up classification predictor...")
    
    # Define bin size and max threshold for classification
    bin_size = 100
    max_bin_threshold = 1000  # New parameter for max bin threshold
    
    # Find max percentile value for logging purposes
    max_p_val = 0
    for dataset in [train_data, val_data, test_data]:
        for item in dataset:
            if f"p{args.percentile}" in ["p25", "p50", "p75", "p99"]:
                p_val = item['output_percentiles'][args.percentile]
            else:
                p_val = np.percentile(item['output_sizes'], args.percentile)
            max_p_val = max(max_p_val, p_val)
    
    print(f"Maximum p{args.percentile} value in dataset: {max_p_val}")
    
    # Create datasets with custom binning
    train_dataset_cls = ResponseLengthDataset(
        train_data, tokenizer, task='classification', 
        bin_size=bin_size, max_bin_threshold=max_bin_threshold, percentile=args.percentile
    )
    val_dataset_cls = ResponseLengthDataset(
        val_data, tokenizer, task='classification', 
        bin_size=bin_size, max_bin_threshold=max_bin_threshold, percentile=args.percentile
    )
    test_dataset_cls = ResponseLengthDataset(
        test_data, tokenizer, task='classification', 
        bin_size=bin_size, max_bin_threshold=max_bin_threshold, percentile=args.percentile
    )
    
    # Number of classes is now determined within the dataset
    num_classes = train_dataset_cls.num_classes
    print(f"Using {num_classes} classes for classification")
    
    train_loader_cls = DataLoader(train_dataset_cls, batch_size=args.batch_size, shuffle=True)
    val_loader_cls = DataLoader(val_dataset_cls, batch_size=args.batch_size)
    test_loader_cls = DataLoader(test_dataset_cls, batch_size=args.batch_size)
    
    # Determine number of models
    num_models = len(train_dataset_cls.models)
    
    # Initialize classification model
    classification_model = ClassificationPredictor(
        pretrained_model_name=MODEL, 
        num_models=num_models, 
        num_classes=num_classes, 
        hidden_size=args.hidden_size
    ).to(device)
    
    # Save model architecture info
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
                           list(classification_model.fc3.parameters()) + \
                           list(classification_model.model_embedding_layer.parameters())
    
    # Create optimizer with parameter groups for classification
    classification_optimizer = optim.AdamW([
        {'params': cls_encoder_params, 'lr': 5e-5, 'weight_decay': 0.01},  # Lower rate for pretrained encoder
        {'params': cls_classifier_params, 'lr': 1e-4, 'weight_decay': 0.01}  # Higher rate for new layers
    ])
    
    # Training parameters
    classification_criterion = nn.CrossEntropyLoss()
    
    # Calculate number of training steps for warmup
    cls_num_training_steps = len(train_loader_cls) * args.num_epochs
    cls_num_warmup_steps = int(0.1 * cls_num_training_steps)  # 10% of total steps for warmup
    
    # Create scheduler with warmup
    classification_scheduler = CosineAnnealingWarmRestarts(
        optimizer=classification_optimizer,
        T_0=len(train_loader_cls),  # Restart after 1 epoch initially
        T_mult=2,  # Double the restart period after each restart
        eta_min=1e-6  # Minimum learning rate
    )
    
    # Train classification model with scheduler
    print("\nTraining classification model...")
    trained_cls_model, cls_metrics = train_model(
        classification_model, train_loader_cls, val_loader_cls, 
        classification_criterion, classification_optimizer,
        scheduler=classification_scheduler, num_epochs=args.num_epochs,
        task='classification'
    )
    
    # Standard evaluation
    print("\nBasic evaluation of classification model on test set...")
    cls_test_metrics = evaluate_model(
        trained_cls_model, test_loader_cls,
        classification_criterion,
        task='classification'
    )
    
    # Save standard metrics
    with open(os.path.join(METRICS_DIR, 'classification_training_metrics.pkl'), 'wb') as f:
        pickle.dump(cls_metrics, f)
    
    with open(os.path.join(METRICS_DIR, 'classification_test_metrics.json'), 'w') as f:
        json.dump(cls_test_metrics, f, indent=2)
    
    # Plot standard training metrics
    plot_metrics(cls_metrics, task='classification', save_path=METRICS_DIR)
    
    # Run detailed classification analysis
    print("\nRunning detailed classification analysis...")
    detailed_cls_metrics, cls_figures = run_classification_analysis(
        trained_cls_model, args.device,
        test_loader_cls,
        classification_criterion,
        save_path=METRICS_DIR
    )
    
    print("\nClassification task completed!")
    print(f"Models saved to: {MODELS_DIR}")
    print(f"Metrics saved to: {METRICS_DIR}")


def main():
    global MODELS_DIR, METRICS_DIR, device
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Train and evaluate BERT-based predictor models')
    parser.add_argument('--task', type=str, choices=['regression', 'classification'], required=True,
                        help='Task to run: regression or classification')
    parser.add_argument('--data_src_dir', type=str, required=True,
                        help='Source directory containing the dataset')
    parser.add_argument('--dst_dir', type=str, required=True,
                        help='Destination directory for saving models and metrics')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to use (e.g., "cuda:0", "cpu")')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=20,
                        help='Number of training epochs')
    parser.add_argument('--hidden_size', type=int, default=512,
                        help='Hidden size for the model')
    parser.add_argument('--percentile', type=int, default=90,
                        help='Hidden size for the model')
    
    args = parser.parse_args()
    
    # Set global variables
    global device
    device = args.device
    print(f"Using device: {device}")
    
    # Set up directories
    MODELS_DIR = os.path.join(args.dst_dir, 'saved_models')
    METRICS_DIR = os.path.join(args.dst_dir, 'metrics')
    
    # Create directories if they don't exist
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(METRICS_DIR, exist_ok=True)
    
    print(f"Models will be saved to: {MODELS_DIR}")
    print(f"Metrics will be saved to: {METRICS_DIR}")
    
    # Run the selected task
    if args.task == 'regression':
        run_regression_task(args)
    else:  # classification
        run_classification_task(args)


if __name__ == "__main__":
    main()