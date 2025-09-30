"""Comprehensive model diagnostic toolkit for analyzing information bottlenecks and layer importance.

This module implements the diagnostic methods suggested by GPT for analyzing:
1. Per-parameter/per-layer importance (Fisher, Taylor saliency, curvature)
2. Information bottleneck analysis (activation entropy, effective rank, CKA)
3. Jacobian spectrum analysis for information throughput
4. Noise-throughput tests for layer sensitivity
5. Linear probes for information preservation

Based on the methodology from the GPT recommendation for post-hoc analysis.
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pandas as pd
from collections import defaultdict
import pickle

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.lightning_module import BrainToTextLightningModule
from dataset.dataset import BrainToTextDataset, collate_fn
from models.base import Batch


class ModelDiagnostics:
    """Comprehensive diagnostic toolkit for neural encoder models."""
    
    def __init__(self, model: nn.Module, device: str = 'auto'):
        """Initialize diagnostics toolkit.
        
        Args:
            model: The neural encoder model to analyze
            device: Device to run analysis on ('auto' to detect from model)
        """
        self.model = model
        if device == 'auto':
            self.device = str(next(model.parameters()).device)
        else:
            self.device = device
            
        self.model.eval()
        
        # Storage for intermediate activations and gradients
        self.activations = {}
        self.gradients = {}
        self.hooks = []
        
        # Results storage
        self.results = {
            'parameter_importance': {},
            'bottleneck_analysis': {},
            'jacobian_analysis': {},
            'noise_sensitivity': {},
            'layer_info': {}
        }
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
        
    def cleanup(self):
        """Remove all hooks and clear stored activations."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        self.activations.clear()
        self.gradients.clear()
    
    def _prepare_batch(self, batch):
        """Convert batch dict to Batch object and move to device."""
        if isinstance(batch, dict):
            batch = Batch.from_dataset_batch(batch)
        return batch.to(self.device)
        
    def _register_hooks(self, layer_names: Optional[List[str]] = None):
        """Register forward and backward hooks for specified layers.
        
        Args:
            layer_names: List of layer names to hook. If None, hooks all named modules.
        """
        self.cleanup()  # Remove existing hooks
        
        def make_forward_hook(name):
            def hook(module, input, output):
                # Store activations (detach to avoid gradient tracking)
                
                # Handle different output types
                if hasattr(output, 'data') and hasattr(output, 'batch_sizes'):
                    # This is a PackedSequence from GRU layers, unpack it
                    from torch.nn.utils.rnn import pad_packed_sequence
                    activation, lengths = pad_packed_sequence(output, batch_first=True)
                elif isinstance(output, tuple):
                    # For modules that return tuples (e.g., some RNN variants)
                    activation = output[0]
                    # Check if the first element is also a PackedSequence
                    if hasattr(activation, 'data') and hasattr(activation, 'batch_sizes'):
                        from torch.nn.utils.rnn import pad_packed_sequence
                        activation, lengths = pad_packed_sequence(activation, batch_first=True)
                    else:
                        activation = activation.detach()
                else:
                    # Regular tensor output
                    activation = output.detach()
                
                self.activations[name] = activation
            return hook
            
        def make_backward_hook(name):
            def hook(module, grad_input, grad_output):
                # Store gradients
                if grad_output[0] is not None:
                    self.gradients[name] = grad_output[0].detach()
            return hook
        
        # Register hooks for specified layers
        for name, module in self.model.named_modules():
            if layer_names is None or name in layer_names:
                if len(list(module.children())) == 0:  # Only leaf modules
                    # Forward hook
                    fwd_hook = module.register_forward_hook(make_forward_hook(name))
                    self.hooks.append(fwd_hook)
                    
                    # Backward hook
                    bwd_hook = module.register_full_backward_hook(make_backward_hook(name))
                    self.hooks.append(bwd_hook)

    def compute_parameter_importance(self, dataloader: DataLoader, 
                                   num_batches: int = 10) -> Dict[str, Any]:
        """Compute per-parameter and per-layer importance metrics.
        
        Args:
            dataloader: DataLoader for computing importance
            num_batches: Number of batches to use for estimation
            
        Returns:
            Dictionary with importance metrics per layer
        """
        print("Computing parameter importance metrics...")
        
        # Storage for gradients and parameters
        param_grads = defaultdict(list)
        param_values = {}
        
        # Collect parameter names and values
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param_values[name] = param.detach().clone()
        
        # Compute gradients over multiple batches
        total_loss = 0.0
        num_samples = 0
        
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Computing gradients", total=num_batches)):
            if batch_idx >= num_batches:
                break
                
            # Move batch to device
            batch = self._prepare_batch(batch)
            
            # Forward pass
            self.model.zero_grad()
            emissions = self.model(batch)
            
            # Compute CTC loss
            log_probs = emissions.log_probs
            input_lengths = emissions.out_lens
            target_lengths = batch.y_lens
            
            # Properly flatten targets for CTC loss
            from models.blocks.utils import flatten_ctc_targets
            targets = flatten_ctc_targets(batch.y, batch.y_lens)
            
            loss = F.ctc_loss(
                log_probs.transpose(0, 1),  # CTC expects [T, B, C]
                targets,
                input_lengths,
                target_lengths,
                blank=0,
                reduction='mean'
            )
            
            # Backward pass
            loss.backward()
            
            # Collect gradients
            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    param_grads[name].append(param.grad.detach().clone())
            
            total_loss += loss.item()
            num_samples += batch.x.size(0)
        
        avg_loss = total_loss / num_batches
        
        # Compute importance metrics
        importance_results = {}
        
        for name in param_values:
            if name not in param_grads or len(param_grads[name]) == 0:
                continue
                
            param = param_values[name]
            grads = torch.stack(param_grads[name])  # [num_batches, ...]
            
            # A) Empirical Fisher Information (diagonal)
            fisher_diag = torch.mean(grads ** 2, dim=0)  # Average over batches
            
            # B) Taylor saliency (first-order)
            avg_grad = torch.mean(grads, dim=0)
            taylor_saliency = torch.abs(param * avg_grad)
            
            # Aggregate statistics
            layer_results = {
                'num_params': param.numel(),
                'param_norm': torch.norm(param).item(),
                'grad_norm': torch.norm(avg_grad).item(),
                
                # Fisher Information
                'fisher_mean': torch.mean(fisher_diag).item(),
                'fisher_std': torch.std(fisher_diag).item(),
                'fisher_max': torch.max(fisher_diag).item(),
                
                # Taylor saliency
                'taylor_mean': torch.mean(taylor_saliency).item(),
                'taylor_std': torch.std(taylor_saliency).item(),
                'taylor_sum': torch.sum(taylor_saliency).item(),
                
                # Normalized metrics
                'fisher_per_param': torch.mean(fisher_diag).item(),
                'taylor_per_param': torch.mean(taylor_saliency).item(),
                
                # Scale-normalized metrics
                'fisher_normalized': torch.mean(fisher_diag * param ** 2).item(),
                'taylor_normalized': torch.mean(taylor_saliency / (torch.abs(param) + 1e-8)).item(),
            }
            
            importance_results[name] = layer_results
        
        # Group by layer type for summary
        layer_summary = self._group_by_layer_type(importance_results)
        
        self.results['parameter_importance'] = {
            'per_parameter': importance_results,
            'layer_summary': layer_summary,
            'avg_loss': avg_loss,
            'num_batches': num_batches,
            'num_samples': num_samples
        }
        
        return self.results['parameter_importance']
    
    def compute_bottleneck_analysis(self, dataloader: DataLoader, 
                                  num_batches: int = 5) -> Dict[str, Any]:
        """Compute information bottleneck metrics.
        
        Args:
            dataloader: DataLoader for computing activations
            num_batches: Number of batches to use
            
        Returns:
            Dictionary with bottleneck analysis results
        """
        print("Computing information bottleneck analysis...")
        
        # Get layer names to analyze - include all important layers
        layer_names = []
        for name, module in self.model.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules only
                # Include all key components
                if any(x in name.lower() for x in ['smoother', 'prenet', 'backbone', 'gru', 'ctc_head', 'linear', 'projection']):
                    layer_names.append(name)
        
        print(f"Analyzing layers: {layer_names}")
        
        # Register hooks
        self._register_hooks(layer_names)
        
        # Collect activations
        all_activations = defaultdict(list)
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc="Collecting activations", total=num_batches)):
                if batch_idx >= num_batches:
                    break
                    
                batch = self._prepare_batch(batch)
                
                # Forward pass
                _ = self.model(batch)
                
                # Store activations
                for name, activation in self.activations.items():
                    # Flatten spatial/temporal dimensions, keep batch and feature dims
                    if activation.dim() >= 3:
                        # [B, T, F] -> [B*T, F]
                        act_flat = activation.view(-1, activation.size(-1))
                    else:
                        act_flat = activation
                    
                    all_activations[name].append(act_flat.cpu())
        
        # Analyze each layer
        bottleneck_results = {}
        
        for name in layer_names:
            if name not in all_activations or len(all_activations[name]) == 0:
                continue
                
            # Concatenate all activations for this layer
            activations = torch.cat(all_activations[name], dim=0)  # [N, F]
            
            if activations.numel() == 0:
                continue
                
            # E) Activation entropy & effective rank
            try:
                # Compute covariance matrix
                activations_centered = activations - torch.mean(activations, dim=0, keepdim=True)
                cov_matrix = torch.mm(activations_centered.t(), activations_centered) / (activations.size(0) - 1)
                
                # Add small regularization for numerical stability
                cov_matrix += torch.eye(cov_matrix.size(0)) * 1e-6
                
                # Compute eigenvalues
                eigenvals = torch.linalg.eigvals(cov_matrix).real
                eigenvals = torch.clamp(eigenvals, min=1e-10)  # Ensure positive
                
                # Effective rank (participation ratio)
                sum_eigs = torch.sum(eigenvals)
                sum_eigs_sq = torch.sum(eigenvals ** 2)
                effective_rank = (sum_eigs ** 2) / sum_eigs_sq
                
                # Activation entropy (log determinant approximation)
                log_det = torch.sum(torch.log(eigenvals))
                activation_entropy = 0.5 * log_det
                
                # Sparsity metrics
                activation_std = torch.std(activations)
                activation_mean_abs = torch.mean(torch.abs(activations))
                
                # Dead unit analysis (for ReLU-like activations)
                zero_fraction = torch.mean((torch.abs(activations) < 1e-6).float())
                
                # Saturation analysis (for bounded activations)
                if 'tanh' in name.lower() or 'sigmoid' in name.lower():
                    saturation_threshold = 0.95
                    saturated_fraction = torch.mean((torch.abs(activations) > saturation_threshold).float())
                else:
                    saturated_fraction = 0.0
                
                layer_results = {
                    'num_features': activations.size(1),
                    'num_samples': activations.size(0),
                    'effective_rank': effective_rank.item(),
                    'activation_entropy': activation_entropy.item(),
                    'eigenvalue_sum': sum_eigs.item(),
                    'max_eigenvalue': torch.max(eigenvals).item(),
                    'min_eigenvalue': torch.min(eigenvals).item(),
                    'condition_number': (torch.max(eigenvals) / torch.min(eigenvals)).item(),
                    'activation_std': activation_std.item(),
                    'activation_mean_abs': activation_mean_abs.item(),
                    'zero_fraction': zero_fraction.item(),
                    'saturated_fraction': saturated_fraction,
                    'rank_ratio': (effective_rank / activations.size(1)).item(),  # Normalized by feature dim
                }
                
                bottleneck_results[name] = layer_results
                
            except Exception as e:
                print(f"Error analyzing layer {name}: {e}")
                continue
        
        # Compute CKA similarities between adjacent layers
        cka_results = self._compute_cka_similarities(all_activations, layer_names)
        
        self.results['bottleneck_analysis'] = {
            'layer_metrics': bottleneck_results,
            'cka_similarities': cka_results,
            'num_batches': num_batches
        }
        
        return self.results['bottleneck_analysis']
    
    def compute_jacobian_analysis(self, dataloader: DataLoader, 
                                num_batches: int = 3, 
                                num_directions: int = 5) -> Dict[str, Any]:
        """Compute Jacobian spectrum analysis for information throughput.
        
        Args:
            dataloader: DataLoader for computing Jacobians
            num_batches: Number of batches to use
            num_directions: Number of random directions for spectrum estimation
            
        Returns:
            Dictionary with Jacobian analysis results
        """
        print("Computing Jacobian spectrum analysis...")
        
        # Get layer names to analyze
        layer_names = []
        for name, module in self.model.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules only
                if any(x in name.lower() for x in ['gru', 'linear', 'projection']):
                    layer_names.append(name)
        
        jacobian_results = {}
        
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Computing Jacobians", total=min(num_batches, len(dataloader)))):
            if batch_idx >= num_batches:
                break
                
            batch = self._prepare_batch(batch)
            
            # Register hooks for this batch
            self._register_hooks(layer_names)
            
            # Forward pass to get activations
            self.model.zero_grad()
            emissions = self.model(batch)
            
            # Analyze each layer
            for name in layer_names:
                if name not in self.activations:
                    continue
                    
                activation = self.activations[name]
                
                if activation.numel() == 0:
                    continue
                
                # Flatten activation for Jacobian computation
                if activation.dim() >= 3:
                    # [B, T, F] -> [B*T, F]
                    act_flat = activation.view(-1, activation.size(-1))
                else:
                    act_flat = activation
                
                # Skip if too large (memory constraints)
                if act_flat.numel() > 1e6:
                    continue
                
                try:
                    # Compute Jacobian spectrum using random projections
                    jacobian_norms = []
                    
                    for _ in range(num_directions):
                        # Random direction
                        v = torch.randn_like(act_flat)
                        v = v / torch.norm(v)
                        
                        # Compute Jacobian-vector product
                        if act_flat.requires_grad:
                            jvp = torch.autograd.grad(
                                outputs=act_flat,
                                inputs=batch.x,
                                grad_outputs=v,
                                retain_graph=True,
                                create_graph=False,
                                only_inputs=True
                            )[0]
                            
                            jacobian_norms.append(torch.norm(jvp).item())
                    
                    if jacobian_norms:
                        layer_result = {
                            'mean_jacobian_norm': np.mean(jacobian_norms),
                            'std_jacobian_norm': np.std(jacobian_norms),
                            'max_jacobian_norm': np.max(jacobian_norms),
                            'min_jacobian_norm': np.min(jacobian_norms),
                        }
                        
                        if name not in jacobian_results:
                            jacobian_results[name] = []
                        jacobian_results[name].append(layer_result)
                        
                except Exception as e:
                    print(f"Error computing Jacobian for layer {name}: {e}")
                    continue
        
        # Average results across batches
        final_results = {}
        for name, batch_results in jacobian_results.items():
            if batch_results:
                final_results[name] = {
                    'mean_jacobian_norm': np.mean([r['mean_jacobian_norm'] for r in batch_results]),
                    'std_jacobian_norm': np.mean([r['std_jacobian_norm'] for r in batch_results]),
                    'max_jacobian_norm': np.mean([r['max_jacobian_norm'] for r in batch_results]),
                    'min_jacobian_norm': np.mean([r['min_jacobian_norm'] for r in batch_results]),
                    'num_batches': len(batch_results)
                }
        
        self.results['jacobian_analysis'] = final_results
        return self.results['jacobian_analysis']
    
    def compute_noise_sensitivity(self, dataloader: DataLoader, 
                                noise_levels: List[float] = [1e-4, 1e-3, 1e-2, 1e-1],
                                num_batches: int = 3) -> Dict[str, Any]:
        """Compute noise sensitivity analysis.
        
        Args:
            dataloader: DataLoader for testing
            noise_levels: List of noise standard deviations to test
            num_batches: Number of batches to use
            
        Returns:
            Dictionary with noise sensitivity results
        """
        print("Computing noise sensitivity analysis...")
        
        # Get layer names to analyze
        layer_names = []
        for name, module in self.model.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules only
                if any(x in name.lower() for x in ['gru', 'linear', 'projection', 'smoother']):
                    layer_names.append(name)
        
        sensitivity_results = {}
        
        # Baseline performance (no noise)
        baseline_losses = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc="Computing baseline", total=min(num_batches, len(dataloader)))):
                if batch_idx >= num_batches:
                    break
                    
                batch = self._prepare_batch(batch)
                emissions = self.model(batch)
                
                # Compute loss
                loss = self._compute_ctc_loss(emissions, batch)
                baseline_losses.append(loss.item())
        
        baseline_loss = np.mean(baseline_losses)
        
        # Test each noise level
        for noise_std in noise_levels:
            print(f"Testing noise level: {noise_std}")
            
            # Test each layer
            for layer_name in layer_names:
                layer_losses = []
                
                # Create hook to inject noise
                def make_noise_hook(std):
                    def hook(module, input, output):
                        if isinstance(output, tuple):
                            # For RNN outputs
                            noisy_output = output[0] + torch.randn_like(output[0]) * std
                            return (noisy_output,) + output[1:]
                        else:
                            return output + torch.randn_like(output) * std
                    return hook
                
                # Find the module and register hook
                target_module = None
                for name, module in self.model.named_modules():
                    if name == layer_name:
                        target_module = module
                        break
                
                if target_module is None:
                    continue
                
                noise_hook = target_module.register_forward_hook(make_noise_hook(noise_std))
                
                try:
                    with torch.no_grad():
                        for batch_idx, batch in enumerate(dataloader):
                            if batch_idx >= num_batches:
                                break
                                
                            batch = self._prepare_batch(batch)
                            emissions = self.model(batch)
                            
                            # Compute loss
                            loss = self._compute_ctc_loss(emissions, batch)
                            layer_losses.append(loss.item())
                    
                    # Store results
                    if layer_name not in sensitivity_results:
                        sensitivity_results[layer_name] = {}
                    
                    avg_loss = np.mean(layer_losses)
                    sensitivity_results[layer_name][noise_std] = {
                        'avg_loss': avg_loss,
                        'loss_increase': avg_loss - baseline_loss,
                        'relative_increase': (avg_loss - baseline_loss) / baseline_loss if baseline_loss > 0 else 0,
                        'std_loss': np.std(layer_losses)
                    }
                    
                finally:
                    noise_hook.remove()
        
        self.results['noise_sensitivity'] = {
            'baseline_loss': baseline_loss,
            'layer_results': sensitivity_results,
            'noise_levels': noise_levels,
            'num_batches': num_batches
        }
        
        return self.results['noise_sensitivity']
    
    def _compute_ctc_loss(self, emissions, batch):
        """Helper to compute CTC loss."""
        log_probs = emissions.log_probs
        input_lengths = emissions.out_lens
        target_lengths = batch.y_lens
        
        # Properly flatten targets for CTC loss
        from models.blocks.utils import flatten_ctc_targets
        targets = flatten_ctc_targets(batch.y, target_lengths)
        
        return F.ctc_loss(
            log_probs.transpose(0, 1),  # CTC expects [T, B, C]
            targets,
            input_lengths,
            target_lengths,
            blank=0,
            reduction='mean'
        )
    
    def _compute_cka_similarities(self, all_activations: Dict, layer_names: List[str]) -> Dict[str, float]:
        """Compute Centered Kernel Alignment (CKA) between adjacent layers."""
        cka_results = {}
        
        # Sort layer names by their position in the model architecture
        # Define expected order based on model architecture
        layer_order = [
            'smoother', 'prenet.day_adapter', 'prenet.activation',
            'backbone.layers.0', 'backbone.layers.1', 'backbone.layers.2', 'backbone.layers.3', 'backbone.layers.4',
            'ctc_head.projection.dropout', 'ctc_head.projection.projection'
        ]
        
        # Sort available layers by architecture order
        sorted_layers = []
        for expected_name in layer_order:
            if expected_name in layer_names and expected_name in all_activations:
                sorted_layers.append(expected_name)
        
        # Add any remaining layers not in the expected order
        for name in layer_names:
            if name not in sorted_layers and name in all_activations:
                sorted_layers.append(name)
        
        print(f"CKA analysis order: {sorted_layers}")
        
        # Compute CKA between adjacent layers
        for i in range(len(sorted_layers) - 1):
            layer1, layer2 = sorted_layers[i], sorted_layers[i + 1]
            
            if layer1 in all_activations and layer2 in all_activations:
                try:
                    # Concatenate activations
                    act1 = torch.cat(all_activations[layer1], dim=0)
                    act2 = torch.cat(all_activations[layer2], dim=0)
                    
                    # Ensure same number of samples
                    min_samples = min(act1.size(0), act2.size(0))
                    act1 = act1[:min_samples]
                    act2 = act2[:min_samples]
                    
                    # Compute CKA
                    cka_score = self._linear_cka(act1, act2)
                    cka_results[f"{layer1}_to_{layer2}"] = cka_score
                    
                except Exception as e:
                    print(f"Error computing CKA between {layer1} and {layer2}: {e}")
        
        return cka_results
    
    def _linear_cka(self, X, Y):
        """Compute linear CKA between two activation matrices."""
        # Center the matrices
        X = X - torch.mean(X, dim=0, keepdim=True)
        Y = Y - torch.mean(Y, dim=0, keepdim=True)
        
        # Compute gram matrices
        K = torch.mm(X, X.t())
        L = torch.mm(Y, Y.t())
        
        # Compute CKA
        numerator = torch.trace(torch.mm(K, L))
        denominator = torch.sqrt(torch.trace(torch.mm(K, K)) * torch.trace(torch.mm(L, L)))
        
        if denominator == 0:
            return 0.0
        
        return (numerator / denominator).item()
    
    def _group_by_layer_type(self, param_results: Dict) -> Dict[str, Dict]:
        """Group parameter importance results by layer type."""
        layer_groups = defaultdict(list)
        
        for param_name, results in param_results.items():
            # Determine layer type from parameter name
            if 'smoother' in param_name:
                layer_type = 'smoother'
            elif 'prenet' in param_name:
                layer_type = 'prenet'
            elif 'backbone' in param_name:
                layer_type = 'backbone'
            elif 'ctc_head' in param_name:
                layer_type = 'ctc_head'
            elif 'aux_head' in param_name:
                layer_type = 'aux_head'
            else:
                layer_type = 'other'
            
            layer_groups[layer_type].append(results)
        
        # Aggregate statistics for each group
        group_summary = {}
        for layer_type, layer_list in layer_groups.items():
            if not layer_list:
                continue
                
            # Compute weighted averages (by number of parameters)
            total_params = sum(r['num_params'] for r in layer_list)
            
            if total_params > 0:
                group_summary[layer_type] = {
                    'total_params': total_params,
                    'num_layers': len(layer_list),
                    'avg_fisher_per_param': sum(r['fisher_per_param'] * r['num_params'] for r in layer_list) / total_params,
                    'avg_taylor_per_param': sum(r['taylor_per_param'] * r['num_params'] for r in layer_list) / total_params,
                    'total_taylor_sum': sum(r['taylor_sum'] for r in layer_list),
                    'avg_param_norm': sum(r['param_norm'] * r['num_params'] for r in layer_list) / total_params,
                    'avg_grad_norm': sum(r['grad_norm'] * r['num_params'] for r in layer_list) / total_params,
                }
        
        return group_summary
    
    def generate_report(self, output_dir: str = "analysis/diagnostic_results"):
        """Generate comprehensive diagnostic report with visualizations."""
        print("Generating diagnostic report...")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save raw results
        with open(f"{output_dir}/diagnostic_results.pkl", 'wb') as f:
            pickle.dump(self.results, f)
        
        # Generate visualizations
        self._plot_parameter_importance(output_dir)
        self._plot_bottleneck_analysis(output_dir)
        self._plot_noise_sensitivity(output_dir)
        
        # Generate summary report
        self._generate_text_report(output_dir)
        
        print(f"Diagnostic report saved to {output_dir}/")
    
    def _plot_parameter_importance(self, output_dir: str):
        """Plot parameter importance metrics."""
        if 'parameter_importance' not in self.results:
            return
            
        results = self.results['parameter_importance']
        
        # Layer summary plot
        if 'layer_summary' in results:
            layer_summary = results['layer_summary']
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Parameter Importance by Layer Type', fontsize=16)
            
            layer_types = list(layer_summary.keys())
            
            # Fisher importance
            fisher_values = [layer_summary[lt]['avg_fisher_per_param'] for lt in layer_types]
            axes[0, 0].bar(layer_types, fisher_values)
            axes[0, 0].set_title('Average Fisher Information per Parameter')
            axes[0, 0].set_ylabel('Fisher Value')
            axes[0, 0].tick_params(axis='x', rotation=45)
            
            # Taylor saliency
            taylor_values = [layer_summary[lt]['avg_taylor_per_param'] for lt in layer_types]
            axes[0, 1].bar(layer_types, taylor_values)
            axes[0, 1].set_title('Average Taylor Saliency per Parameter')
            axes[0, 1].set_ylabel('Taylor Value')
            axes[0, 1].tick_params(axis='x', rotation=45)
            
            # Parameter count
            param_counts = [layer_summary[lt]['total_params'] for lt in layer_types]
            axes[1, 0].bar(layer_types, param_counts)
            axes[1, 0].set_title('Total Parameters by Layer Type')
            axes[1, 0].set_ylabel('Number of Parameters')
            axes[1, 0].tick_params(axis='x', rotation=45)
            
            # Gradient norms
            grad_norms = [layer_summary[lt]['avg_grad_norm'] for lt in layer_types]
            axes[1, 1].bar(layer_types, grad_norms)
            axes[1, 1].set_title('Average Gradient Norm by Layer Type')
            axes[1, 1].set_ylabel('Gradient Norm')
            axes[1, 1].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            plt.savefig(f"{output_dir}/parameter_importance_summary.png", dpi=300, bbox_inches='tight')
            plt.close()
    
    def _plot_bottleneck_analysis(self, output_dir: str):
        """Plot bottleneck analysis results."""
        if 'bottleneck_analysis' not in self.results:
            return
            
        results = self.results['bottleneck_analysis']
        
        if 'layer_metrics' in results:
            layer_metrics = results['layer_metrics']
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Information Bottleneck Analysis', fontsize=16)
            
            layer_names = list(layer_metrics.keys())
            
            # Effective rank
            effective_ranks = [layer_metrics[ln]['effective_rank'] for ln in layer_names]
            rank_ratios = [layer_metrics[ln]['rank_ratio'] for ln in layer_names]
            
            axes[0, 0].bar(range(len(layer_names)), effective_ranks)
            axes[0, 0].set_title('Effective Rank by Layer')
            axes[0, 0].set_ylabel('Effective Rank')
            axes[0, 0].set_xticks(range(len(layer_names)))
            axes[0, 0].set_xticklabels([ln.split('.')[-1] for ln in layer_names], rotation=45)
            
            # Rank ratio (normalized)
            axes[0, 1].bar(range(len(layer_names)), rank_ratios)
            axes[0, 1].set_title('Rank Ratio (Effective Rank / Feature Dim)')
            axes[0, 1].set_ylabel('Rank Ratio')
            axes[0, 1].set_xticks(range(len(layer_names)))
            axes[0, 1].set_xticklabels([ln.split('.')[-1] for ln in layer_names], rotation=45)
            
            # Activation entropy
            entropies = [layer_metrics[ln]['activation_entropy'] for ln in layer_names]
            axes[1, 0].bar(range(len(layer_names)), entropies)
            axes[1, 0].set_title('Activation Entropy by Layer')
            axes[1, 0].set_ylabel('Entropy')
            axes[1, 0].set_xticks(range(len(layer_names)))
            axes[1, 0].set_xticklabels([ln.split('.')[-1] for ln in layer_names], rotation=45)
            
            # Zero fraction (sparsity)
            zero_fractions = [layer_metrics[ln]['zero_fraction'] for ln in layer_names]
            axes[1, 1].bar(range(len(layer_names)), zero_fractions)
            axes[1, 1].set_title('Zero Fraction (Sparsity) by Layer')
            axes[1, 1].set_ylabel('Zero Fraction')
            axes[1, 1].set_xticks(range(len(layer_names)))
            axes[1, 1].set_xticklabels([ln.split('.')[-1] for ln in layer_names], rotation=45)
            
            plt.tight_layout()
            plt.savefig(f"{output_dir}/bottleneck_analysis.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        # CKA similarity plot
        if 'cka_similarities' in results:
            cka_results = results['cka_similarities']
            
            if cka_results:
                plt.figure(figsize=(12, 6))
                
                pairs = list(cka_results.keys())
                similarities = list(cka_results.values())
                
                plt.bar(range(len(pairs)), similarities)
                plt.title('CKA Similarities Between Adjacent Layers')
                plt.ylabel('CKA Similarity')
                plt.xlabel('Layer Pairs')
                plt.xticks(range(len(pairs)), [p.replace('_to_', ' → ') for p in pairs], rotation=45)
                plt.axhline(y=0.8, color='r', linestyle='--', alpha=0.7, label='High similarity threshold')
                plt.axhline(y=0.5, color='orange', linestyle='--', alpha=0.7, label='Medium similarity threshold')
                plt.legend()
                
                plt.tight_layout()
                plt.savefig(f"{output_dir}/cka_similarities.png", dpi=300, bbox_inches='tight')
                plt.close()
    
    def _plot_noise_sensitivity(self, output_dir: str):
        """Plot noise sensitivity results."""
        if 'noise_sensitivity' not in self.results:
            return
            
        results = self.results['noise_sensitivity']
        
        if 'layer_results' in results:
            layer_results = results['layer_results']
            noise_levels = results['noise_levels']
            
            plt.figure(figsize=(15, 8))
            
            for layer_name in layer_results:
                relative_increases = []
                for noise_level in noise_levels:
                    if noise_level in layer_results[layer_name]:
                        relative_increases.append(layer_results[layer_name][noise_level]['relative_increase'])
                    else:
                        relative_increases.append(0)
                
                plt.plot(noise_levels, relative_increases, marker='o', label=layer_name.split('.')[-1])
            
            plt.xscale('log')
            plt.xlabel('Noise Standard Deviation')
            plt.ylabel('Relative Loss Increase')
            plt.title('Noise Sensitivity by Layer')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f"{output_dir}/noise_sensitivity.png", dpi=300, bbox_inches='tight')
            plt.close()
    
    def _generate_text_report(self, output_dir: str):
        """Generate comprehensive text report."""
        report_lines = []
        report_lines.append("# Model Diagnostic Report")
        report_lines.append("=" * 50)
        report_lines.append("")
        
        # Parameter importance summary
        if 'parameter_importance' in self.results:
            results = self.results['parameter_importance']
            report_lines.append("## Parameter Importance Analysis")
            report_lines.append("")
            
            if 'layer_summary' in results:
                layer_summary = results['layer_summary']
                
                report_lines.append("### Layer Type Summary:")
                for layer_type, metrics in layer_summary.items():
                    report_lines.append(f"\n**{layer_type.upper()}:**")
                    report_lines.append(f"  - Total parameters: {metrics['total_params']:,}")
                    report_lines.append(f"  - Number of layers: {metrics['num_layers']}")
                    report_lines.append(f"  - Avg Fisher per param: {metrics['avg_fisher_per_param']:.6f}")
                    report_lines.append(f"  - Avg Taylor per param: {metrics['avg_taylor_per_param']:.6f}")
                    report_lines.append(f"  - Total Taylor sum: {metrics['total_taylor_sum']:.6f}")
            
            report_lines.append(f"\nAnalysis based on {results['num_batches']} batches, {results['num_samples']} samples")
            report_lines.append(f"Average loss: {results['avg_loss']:.4f}")
            report_lines.append("")
        
        # Bottleneck analysis summary
        if 'bottleneck_analysis' in self.results:
            results = self.results['bottleneck_analysis']
            report_lines.append("## Information Bottleneck Analysis")
            report_lines.append("")
            
            if 'layer_metrics' in results:
                layer_metrics = results['layer_metrics']
                
                # Find potential bottlenecks
                bottleneck_layers = []
                for layer_name, metrics in layer_metrics.items():
                    rank_ratio = metrics['rank_ratio']
                    zero_fraction = metrics['zero_fraction']
                    
                    if rank_ratio < 0.5 or zero_fraction > 0.5:
                        bottleneck_layers.append((layer_name, rank_ratio, zero_fraction))
                
                if bottleneck_layers:
                    report_lines.append("### Potential Bottleneck Layers:")
                    for layer_name, rank_ratio, zero_fraction in bottleneck_layers:
                        report_lines.append(f"  - {layer_name}: rank_ratio={rank_ratio:.3f}, sparsity={zero_fraction:.3f}")
                else:
                    report_lines.append("### No obvious bottleneck layers detected")
                
                report_lines.append("")
            
            if 'cka_similarities' in results:
                cka_results = results['cka_similarities']
                report_lines.append("### CKA Similarities:")
                for pair, similarity in cka_results.items():
                    report_lines.append(f"  - {pair.replace('_to_', ' → ')}: {similarity:.3f}")
                report_lines.append("")
        
        # Noise sensitivity summary
        if 'noise_sensitivity' in self.results:
            results = self.results['noise_sensitivity']
            report_lines.append("## Noise Sensitivity Analysis")
            report_lines.append("")
            
            baseline_loss = results['baseline_loss']
            report_lines.append(f"Baseline loss: {baseline_loss:.4f}")
            report_lines.append("")
            
            if 'layer_results' in results:
                layer_results = results['layer_results']
                
                # Find most sensitive layers
                most_sensitive = []
                for layer_name, noise_data in layer_results.items():
                    # Use highest noise level for sensitivity ranking
                    max_noise = max(results['noise_levels'])
                    if max_noise in noise_data:
                        relative_increase = noise_data[max_noise]['relative_increase']
                        most_sensitive.append((layer_name, relative_increase))
                
                most_sensitive.sort(key=lambda x: x[1], reverse=True)
                
                report_lines.append("### Most Noise-Sensitive Layers (at highest noise level):")
                for layer_name, sensitivity in most_sensitive[:5]:
                    report_lines.append(f"  - {layer_name}: {sensitivity:.3f} relative increase")
                report_lines.append("")
        
        # Recommendations
        report_lines.append("## Recommendations")
        report_lines.append("")
        
        # Analyze results for recommendations
        recommendations = []
        
        if 'parameter_importance' in self.results:
            layer_summary = self.results['parameter_importance'].get('layer_summary', {})
            
            # Check if backbone dominates parameters
            if 'backbone' in layer_summary:
                backbone_params = layer_summary['backbone']['total_params']
                total_params = sum(ls['total_params'] for ls in layer_summary.values())
                backbone_ratio = backbone_params / total_params
                
                if backbone_ratio > 0.8:
                    recommendations.append(f"🔍 **Backbone dominates parameters** ({backbone_ratio:.1%} of total). Consider reducing hidden size or layers.")
            
            # Check Fisher importance ratios
            if len(layer_summary) > 1:
                fisher_values = [ls['avg_fisher_per_param'] for ls in layer_summary.values()]
                max_fisher = max(fisher_values)
                min_fisher = min(fisher_values)
                
                if max_fisher / min_fisher > 100:
                    recommendations.append("⚠️ **Large Fisher importance variation** between layer types. Some layers may be under/over-parameterized.")
        
        if 'bottleneck_analysis' in self.results:
            layer_metrics = self.results['bottleneck_analysis'].get('layer_metrics', {})
            
            # Check for severe bottlenecks
            severe_bottlenecks = []
            for layer_name, metrics in layer_metrics.items():
                if metrics['rank_ratio'] < 0.3:
                    severe_bottlenecks.append(layer_name)
            
            if severe_bottlenecks:
                recommendations.append(f"🚨 **Severe information bottlenecks detected** in: {', '.join(severe_bottlenecks)}. Consider increasing hidden dimensions.")
            
            # Check CKA drops
            cka_results = self.results['bottleneck_analysis'].get('cka_similarities', {})
            low_cka = [pair for pair, sim in cka_results.items() if sim < 0.3]
            
            if low_cka:
                recommendations.append(f"📉 **Low CKA similarities** in: {', '.join(low_cka)}. Information may be lost between these layers.")
        
        if recommendations:
            for rec in recommendations:
                report_lines.append(f"- {rec}")
        else:
            report_lines.append("- No major issues detected. Model architecture appears well-balanced.")
        
        report_lines.append("")
        report_lines.append("---")
        report_lines.append("Generated by Model Diagnostics Toolkit")
        
        # Save report
        with open(f"{output_dir}/diagnostic_report.md", 'w') as f:
            f.write('\n'.join(report_lines))


def main():
    """Main function to run diagnostics on trained model."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run model diagnostics')
    parser.add_argument('--checkpoint', type=str, default='models/checkpoints/trained.ckpt',
                       help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='pipeline/config.yaml',
                       help='Path to config file')
    parser.add_argument('--output_dir', type=str, default='analysis/diagnostic_results',
                       help='Output directory for results')
    parser.add_argument('--num_batches', type=int, default=10,
                       help='Number of batches to use for analysis')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to run on (auto, cpu, cuda, mps)')
    
    args = parser.parse_args()
    
    print("Loading model and data...")
    
    # Load config
    import yaml
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load model
    lightning_module = BrainToTextLightningModule.load_from_checkpoint(
        args.checkpoint,
        config=config,
        strict=False
    )
    
    # Use EMA model if available
    if hasattr(lightning_module, 'ema') and lightning_module.ema is not None:
        print("Using EMA model for diagnostics")
        lightning_module.ema.apply_shadow()
        model = lightning_module.model
    else:
        print("Using regular model (no EMA available)")
        model = lightning_module.model
    
    # Set device
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    else:
        device = args.device
    
    model = model.to(device)
    
    # Load validation dataset
    dataset_config = config['dataset']
    val_dataset = BrainToTextDataset(
        data_root=dataset_config['data_root'],
        split='val',
        corpus_filter=dataset_config.get('corpus_filter'),
        bad_trials_dict=dataset_config.get('bad_trials_dict'),
        random_seed=dataset_config.get('random_seed', 42)
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=4,  # Smaller batch size for diagnostics
        shuffle=False,
        num_workers=0,  # Avoid multiprocessing issues
        pin_memory=False,
        collate_fn=collate_fn
    )
    
    print(f"Running diagnostics on {len(val_dataset)} validation samples...")
    
    # Run diagnostics
    with ModelDiagnostics(model, device=device) as diagnostics:
        # Parameter importance analysis
        diagnostics.compute_parameter_importance(val_loader, num_batches=args.num_batches)
        
        # Bottleneck analysis
        diagnostics.compute_bottleneck_analysis(val_loader, num_batches=min(5, args.num_batches))
        
        # Jacobian analysis (computationally expensive) - skip due to gradient issues
        try:
            diagnostics.compute_jacobian_analysis(val_loader, num_batches=min(2, args.num_batches))
        except Exception as e:
            print(f"Jacobian analysis failed: {e}")
            print("Continuing with other analyses...")
        
        # Noise sensitivity analysis
        diagnostics.compute_noise_sensitivity(val_loader, num_batches=min(3, args.num_batches))
        
        # Generate report
        diagnostics.generate_report(args.output_dir)
    
    print(f"Diagnostics complete! Results saved to {args.output_dir}/")


if __name__ == '__main__':
    main()
