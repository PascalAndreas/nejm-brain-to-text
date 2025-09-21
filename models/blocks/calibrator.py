"""CTC-aware temperature calibration for neural encoder models.

This module provides CTC-specific temperature scaling for model calibration,
which improves the reliability of confidence scores by minimizing CTC loss
on a held-out validation set.

For CTC models, prefer CTCTemperatureCalibrator (below). The legacy
TemperatureScaling/PlattScaling target framewise classification
and are not alignment-aware.
"""

import math
import os
from typing import Optional, Dict, Any
import torch
from ..base import Batch
import torch.nn as nn
import torch.nn.functional as F


# Legacy calibrator removed - temperature is now handled by CTCHead directly


def framewise_ece(
    log_probs: torch.Tensor, 
    targets: torch.Tensor, 
    input_lengths: torch.LongTensor, 
    num_bins: int = 10
) -> float:
    """
    Framewise ECE on valid frames only.
    
    Args:
        log_probs: Log probabilities [B,T,V]
        targets: Target sequences [B,T] with pad idx to ignore
        input_lengths: Valid sequence lengths [B]
        num_bins: Number of bins for calibration
        
    Returns:
        Expected Calibration Error
    """
    B, T, V = log_probs.shape
    device = log_probs.device
    t = torch.arange(T, device=device).view(1, T)
    mask = (t < input_lengths.view(B, 1)).unsqueeze(-1)   # [B,T,1]
    probs = log_probs.exp()
    conf, pred = probs.max(dim=-1)                       # [B,T]
    
    # filter valid frames + valid labels if you have pad in targets
    conf = conf[mask.squeeze(-1)]
    if targets.dim() == 2:
        lab = targets[mask.squeeze(-1)]
        acc = (pred[mask.squeeze(-1)] == lab)
    else:
        # if you don't have aligned targets, skip accuracy computation
        return float("nan")
    
    # Create bins on same device/dtype
    bin_boundaries = torch.linspace(0, 1, num_bins + 1, device=device, dtype=conf.dtype)
    
    ece = 0.0
    for i in range(num_bins):
        # Select samples in bin - all bins use (a, b] for consistency
        in_bin = (conf > bin_boundaries[i]) & (conf <= bin_boundaries[i + 1])
        prop_in_bin = in_bin.float().mean()
        
        if prop_in_bin > 0:
            # Compute accuracy and confidence in bin
            accuracy_in_bin = acc[in_bin].float().mean()
            avg_confidence_in_bin = conf[in_bin].mean()
            
            # Add to ECE
            ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return ece.item()


def compute_ece(
    probs: torch.FloatTensor,
    labels: torch.LongTensor,
    num_bins: int = 10
) -> float:
    """Compute Expected Calibration Error (ECE) - legacy framewise version.
    
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
    
    # Create bins on same device/dtype as inputs
    bin_boundaries = torch.linspace(0, 1, num_bins + 1, device=probs.device, dtype=probs.dtype)
    
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
    """Compute Maximum Calibration Error (MCE) - legacy framewise version.
    
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
    
    # Create bins on same device/dtype as inputs
    bin_boundaries = torch.linspace(0, 1, num_bins + 1, device=probs.device, dtype=probs.dtype)
    
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


class FitCTCTemperature:
    """
    Optimize the CTCHead's internal temperature on a validation loader.

    Expects: model.eval() and a calibration forward that returns:
      {'logits': [B,T,V], 'input_lengths': [B], 'targets': 1D, 'target_lengths': [B]}
    """

    def __init__(self, blank_idx: int = 0):
        self.blank_idx = blank_idx

    @torch.no_grad()
    def _move_batch_to(self, batch, device: str):
        """Move batch to device and convert dict to Batch object if needed."""
        if isinstance(batch, dict):
            # Convert dict to Batch object first
            batch = Batch.from_dataset_batch(batch)
        if hasattr(batch, "to"):
            return batch.to(device)
        return batch

    def run(
        self,
        model: nn.Module,
        dataloader,
        *,
        device: str = "auto",
        epochs: int = 5,
        lr: float = 0.05,
        use_amp: bool = False,
    ) -> float:
        """
        Fit temperature parameter by minimizing CTC loss on validation data.
        
        Args:
            model: The neural encoder model
            dataloader: Validation dataloader
            device: Device for optimization ("auto" detects from model)
            epochs: Number of epochs to run (temperature usually converges fast)
            lr: Learning rate for temperature optimization
            use_amp: Whether to use automatic mixed precision
            
        Returns:
            Optimal temperature value
        """
        model.eval()
        
        # Auto-detect device from model if needed
        if device == "auto":
            device = str(next(model.parameters()).device)
        
        # Handle MPS limitations - CTC Loss not supported natively on MPS
        if device == "mps":
            fallback_enabled = os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") == "1"
            if fallback_enabled:
                print("⚠️  MPS detected with fallback enabled - CTC Loss will use CPU fallback")
                print("   This may be slower than native CPU/CUDA but allows MPS training")
            else:
                print("⚠️  MPS device detected but CTC Loss is not supported on MPS")
                print("   Options:")
                print("   1. Set PYTORCH_ENABLE_MPS_FALLBACK=1 before importing torch")
                print("   2. Use CPU: model = model.to('cpu')")
                print("   3. Use CUDA if available")
                raise RuntimeError(
                    "CTC Loss not supported on MPS device. Please enable MPS fallback "
                    "(PYTORCH_ENABLE_MPS_FALLBACK=1) or use CPU/CUDA for calibration."
                )
        
        ctc = nn.CTCLoss(blank=self.blank_idx, zero_infinity=True).to(device)

        # Grab head
        head = model.ctc_head if hasattr(model, "ctc_head") else model.head

        # 1) Materialize classwise τ vector (if used) BEFORE building optimizer
        first = next(iter(dataloader))
        first = self._move_batch_to(first, device)
        with torch.no_grad():
            out0 = model.forward_for_calibration(first)
            logits0 = out0["logits"]
            _ = head._apply_temperature(logits0)  # creates _tau_vec if needed

        # 2) Build optimizer ONLY over temperature params
        params = [head._tau_vec] if head.classwise_temp else [head._rho]
        
        # Guard against accidental gradients in projection weights
        for p in head.projection.parameters():
            p.requires_grad = False
        head.enable_temp_training(True)
        
        opt = torch.optim.Adam(params, lr=lr)
        # Device-agnostic GradScaler
        device_type = 'cuda' if device.startswith('cuda') else device
        scaler = torch.amp.GradScaler(device_type, enabled=use_amp)
        
        print(f"Initial temperature: {head.temperature():.4f}")

        for epoch in range(epochs):
            opt.zero_grad(set_to_none=True)
            total_loss = 0.0
            batch_count = 0

            for batch in dataloader:
                batch = self._move_batch_to(batch, device)

                # No grads through model; only temp
                with torch.no_grad():
                    out = model.forward_for_calibration(batch)
                    logits = out["logits"]             # [B,T,V] (raw)
                    in_lens = out["input_lengths"]     # [B]
                    targets = out["targets"]           # 1D concat
                    tgt_lens = out["target_lengths"]   # [B]

                with torch.amp.autocast(device_type, enabled=use_amp):
                    logits_tau = head._apply_temperature(logits)
                    logp = F.log_softmax(logits_tau, dim=-1).transpose(0, 1)  # [T,B,V]
                    loss = ctc(logp, targets, in_lens, tgt_lens)

                scaler.scale(loss).backward()
                total_loss += float(loss.detach().cpu().item())
                batch_count += 1

            scaler.step(opt)
            scaler.update()
            
            # Print progress for each epoch
            avg_loss = total_loss / max(batch_count, 1)
            temp = head.temperature()
            print(f"[epoch {epoch + 1}/{epochs}] avg CTC = {avg_loss:.4f}, τ = {temp:.4f}")

        head.enable_temp_training(False)
        final_temp = head.temperature()
        print(f"Final temperature: {final_temp:.4f}")
        return final_temp