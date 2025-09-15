"""Temperature calibration for neural encoder models.

This module provides temperature scaling for model calibration,
which improves the reliability of confidence scores.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import LBFGS
import numpy as np
from tqdm import tqdm


class TemperatureScaling(nn.Module):
    """Temperature scaling for model calibration.
    
    Temperature scaling is a simple post-processing technique that improves
    model calibration by scaling logits with a learned temperature parameter.
    This is typically fitted on a validation set after training.
    
    Args:
        initial_temperature: Initial temperature value
        learnable: If True, temperature is a learnable parameter
        device: Device for optimization
    """
    
    def __init__(
        self,
        initial_temperature: float = 1.0,
        learnable: bool = True,
        device: str = 'cpu'
    ):
        super().__init__()
        
        self.device = device
        
        if learnable:
            # Temperature as learnable parameter
            self.temperature = nn.Parameter(torch.tensor(initial_temperature))
        else:
            # Fixed temperature
            self.register_buffer('temperature', torch.tensor(initial_temperature))
    
    def forward(self, logits: torch.FloatTensor) -> torch.FloatTensor:
        """Apply temperature scaling to logits.
        
        Args:
            logits: Input logits [B, T, V] or [B, V]
            
        Returns:
            Scaled logits
        """
        return logits / self.temperature.clamp(min=1e-8)
    
    def fit(
        self,
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        criterion: Optional[nn.Module] = None,
        max_iter: int = 50,
        tolerance: float = 1e-5
    ) -> float:
        """Fit temperature on validation data.
        
        Args:
            model: The neural encoder model
            dataloader: Validation dataloader
            criterion: Loss function (default: NLLLoss)
            max_iter: Maximum optimization iterations
            tolerance: Convergence tolerance
            
        Returns:
            Optimal temperature value
        """
        if criterion is None:
            criterion = nn.NLLLoss(reduction='mean')
        
        # Collect model predictions
        model.eval()
        all_logits = []
        all_labels = []
        all_lengths = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Collecting predictions"):
                # Move batch to device
                if hasattr(batch, 'to'):
                    batch = batch.to(self.device)
                
                # Get model predictions
                emissions = model(batch)
                
                # Store predictions and labels
                all_logits.append(emissions.log_probs.cpu())
                if batch.y is not None:
                    all_labels.append(batch.y.cpu())
                    all_lengths.append(batch.y_lens.cpu())
        
        if not all_labels:
            print("Warning: No labels found for temperature calibration")
            return 1.0
        
        # Concatenate all predictions
        all_logits = torch.cat(all_logits, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        all_lengths = torch.cat(all_lengths, dim=0)
        
        # Move to optimization device
        all_logits = all_logits.to(self.device)
        all_labels = all_labels.to(self.device)
        
        # Optimize temperature
        self.temperature.data = torch.tensor(1.0).to(self.device)
        
        optimizer = LBFGS(
            [self.temperature],
            lr=0.01,
            max_iter=max_iter,
            tolerance_change=tolerance
        )
        
        def eval_loss():
            optimizer.zero_grad()
            
            # Apply temperature scaling
            scaled_logits = all_logits / self.temperature.clamp(min=1e-8)
            
            # Compute loss (simplified - you may need to handle CTC properly)
            loss = 0.0
            for i in range(len(all_logits)):
                seq_len = all_lengths[i]
                seq_logits = scaled_logits[i, :seq_len]
                seq_labels = all_labels[i, :seq_len]
                
                # Simple cross-entropy loss (for CTC you'd use CTCLoss)
                seq_loss = F.cross_entropy(
                    seq_logits.view(-1, seq_logits.size(-1)),
                    seq_labels.view(-1),
                    ignore_index=0  # Ignore padding
                )
                loss += seq_loss
            
            loss = loss / len(all_logits)
            loss.backward()
            return loss
        
        print("Optimizing temperature...")
        optimizer.step(eval_loss)
        
        optimal_temp = self.temperature.item()
        print(f"Optimal temperature: {optimal_temp:.4f}")
        
        return optimal_temp
    
    def set_temperature(self, temperature: float):
        """Manually set temperature value.
        
        Args:
            temperature: New temperature value
        """
        if isinstance(self.temperature, nn.Parameter):
            self.temperature.data = torch.tensor(temperature)
        else:
            self.temperature = torch.tensor(temperature)


class PlattScaling(nn.Module):
    """Platt scaling for binary or multi-class calibration.
    
    Platt scaling fits a sigmoid (binary) or softmax (multi-class) function
    to map model outputs to calibrated probabilities.
    
    Args:
        num_classes: Number of classes
        use_bias: If True, use bias term
    """
    
    def __init__(self, num_classes: int, use_bias: bool = True):
        super().__init__()
        
        self.num_classes = num_classes
        
        if num_classes == 2:
            # Binary case: single weight and bias
            self.weight = nn.Parameter(torch.ones(1))
            self.bias = nn.Parameter(torch.zeros(1)) if use_bias else None
        else:
            # Multi-class: linear transformation
            self.weight = nn.Parameter(torch.eye(num_classes))
            self.bias = nn.Parameter(torch.zeros(num_classes)) if use_bias else None
    
    def forward(self, logits: torch.FloatTensor) -> torch.FloatTensor:
        """Apply Platt scaling.
        
        Args:
            logits: Input logits [..., num_classes]
            
        Returns:
            Scaled logits
        """
        if self.num_classes == 2:
            # Binary scaling
            scaled = logits * self.weight
            if self.bias is not None:
                scaled = scaled + self.bias
        else:
            # Multi-class scaling
            scaled = torch.matmul(logits, self.weight.T)
            if self.bias is not None:
                scaled = scaled + self.bias
        
        return scaled


class EnsembleTemperature(nn.Module):
    """Temperature scaling for ensemble models.
    
    When ensembling multiple models, each model can have its own temperature
    for better calibration.
    
    Args:
        num_models: Number of models in ensemble
        initial_temperature: Initial temperature for all models
        shared: If True, use same temperature for all models
    """
    
    def __init__(
        self,
        num_models: int,
        initial_temperature: float = 1.0,
        shared: bool = False
    ):
        super().__init__()
        
        self.num_models = num_models
        self.shared = shared
        
        if shared:
            self.temperature = nn.Parameter(torch.tensor(initial_temperature))
        else:
            self.temperatures = nn.ParameterList([
                nn.Parameter(torch.tensor(initial_temperature))
                for _ in range(num_models)
            ])
    
    def forward(
        self,
        logits_list: list[torch.FloatTensor]
    ) -> list[torch.FloatTensor]:
        """Apply temperature scaling to ensemble predictions.
        
        Args:
            logits_list: List of logits from each model
            
        Returns:
            List of scaled logits
        """
        scaled_logits = []
        
        for i, logits in enumerate(logits_list):
            if self.shared:
                temp = self.temperature
            else:
                temp = self.temperatures[i]
            
            scaled_logits.append(logits / temp.clamp(min=1e-8))
        
        return scaled_logits
    
    def get_temperatures(self) -> list[float]:
        """Get temperature values for all models.
        
        Returns:
            List of temperature values
        """
        if self.shared:
            return [self.temperature.item()] * self.num_models
        else:
            return [t.item() for t in self.temperatures]


def compute_ece(
    probs: torch.FloatTensor,
    labels: torch.LongTensor,
    num_bins: int = 10
) -> float:
    """Compute Expected Calibration Error (ECE).
    
    ECE measures the difference between confidence and accuracy across bins.
    
    Args:
        probs: Predicted probabilities [N, C]
        labels: True labels [N]
        num_bins: Number of bins for calibration
        
    Returns:
        ECE value
    """
    # Get confidence and predictions
    confidences, predictions = torch.max(probs, dim=1)
    accuracies = predictions.eq(labels)
    
    # Create bins
    bin_boundaries = torch.linspace(0, 1, num_bins + 1)
    
    ece = 0.0
    for i in range(num_bins):
        # Select samples in bin
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        prop_in_bin = in_bin.float().mean()
        
        if prop_in_bin > 0:
            # Compute accuracy and confidence in bin
            accuracy_in_bin = accuracies[in_bin].float().mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            
            # Add to ECE
            ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return ece.item()


def compute_mce(
    probs: torch.FloatTensor,
    labels: torch.LongTensor,
    num_bins: int = 10
) -> float:
    """Compute Maximum Calibration Error (MCE).
    
    MCE is the maximum difference between confidence and accuracy across bins.
    
    Args:
        probs: Predicted probabilities [N, C]
        labels: True labels [N]
        num_bins: Number of bins for calibration
        
    Returns:
        MCE value
    """
    # Get confidence and predictions
    confidences, predictions = torch.max(probs, dim=1)
    accuracies = predictions.eq(labels)
    
    # Create bins
    bin_boundaries = torch.linspace(0, 1, num_bins + 1)
    
    max_calibration_error = 0.0
    for i in range(num_bins):
        # Select samples in bin
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        
        if in_bin.sum() > 0:
            # Compute accuracy and confidence in bin
            accuracy_in_bin = accuracies[in_bin].float().mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            
            # Update max calibration error
            calibration_error = torch.abs(avg_confidence_in_bin - accuracy_in_bin)
            max_calibration_error = max(max_calibration_error, calibration_error.item())
    
    return max_calibration_error
