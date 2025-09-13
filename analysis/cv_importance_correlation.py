#!/usr/bin/env python3
"""
Correlate feature importance with coefficient of variation (CV) to see
if the model is effectively ignoring high-variability features.
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import pearsonr, spearmanr
import h5py
from tqdm import tqdm

# Add parent directory to path
sys.path.append('..')

def load_feature_cvs():
    """Load feature CVs from our previous analysis"""
    print("Computing feature CVs from data...")
    
    data_dir = '../data/t15_copyTask_neuralData/hdf5_data_final'
    sessions = []
    for item in os.listdir(data_dir):
        if item.startswith('t15.') and os.path.isdir(os.path.join(data_dir, item)):
            sessions.append(item)
    
    sessions.sort()
    
    # Load data from multiple sessions
    session_data = {}
    for session in tqdm(sessions[:15], desc="Loading sessions"):  # First 15 sessions
        session_path = os.path.join(data_dir, session)
        train_file = os.path.join(session_path, 'data_train.hdf5')
        
        if os.path.exists(train_file):
            try:
                with h5py.File(train_file, 'r') as f:
                    trials = list(f.keys())[:20]  # First 20 trials per session
                    
                    neural_features = []
                    for trial_key in trials:
                        trial_data = f[trial_key]['input_features'][:]
                        neural_features.append(trial_data)
                    
                    if neural_features:
                        all_data = np.concatenate(neural_features, axis=0)
                        session_data[session] = all_data
                        
            except Exception as e:
                continue
    
    # Compute CVs
    session_means = []
    for session, data in session_data.items():
        feature_means = np.mean(data, axis=0)  # [512]
        session_means.append(feature_means)
    
    session_means = np.array(session_means)  # [sessions, 512]
    
    # Compute CV for each feature across sessions
    feature_cvs = np.std(session_means, axis=0) / (np.abs(np.mean(session_means, axis=0)) + 1e-8)
    
    return feature_cvs

def load_importance_metrics():
    """Load importance metrics from the previous analysis"""
    print("Loading importance metrics from model...")
    
    # This is a simplified version - in practice you'd save/load the actual metrics
    # For now, we'll recompute the key weight-based metrics
    
    try:
        from omegaconf import OmegaConf
        import torch
        sys.path.append('../model_training')
        from rnn_model import GRUDecoder
        
        model_path = '../data/t15_pretrained_rnn_baseline/t15_pretrained_rnn_baseline'
        model_args = OmegaConf.load(os.path.join(model_path, 'checkpoint/args.yaml'))
        device = torch.device('cpu')
        
        model = GRUDecoder(
            neural_dim=model_args['model']['n_input_features'],
            n_units=model_args['model']['n_units'], 
            n_days=len(model_args['dataset']['sessions']),
            n_classes=model_args['dataset']['n_classes'],
            rnn_dropout=model_args['model']['rnn_dropout'],
            input_dropout=model_args['model']['input_network']['input_layer_dropout'],
            n_layers=model_args['model']['n_layers'],
            patch_size=model_args['model']['patch_size'],
            patch_stride=model_args['model']['patch_stride'],
        )
        
        checkpoint = torch.load(os.path.join(model_path, 'checkpoint/best_checkpoint'), 
                              map_location=device, weights_only=False)
        
        state_dict = checkpoint['model_state_dict']
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key.replace("module.", "").replace("_orig_mod.", "")
            new_state_dict[new_key] = value
        
        model.load_state_dict(new_state_dict)
        
        # Extract importance metrics
        n_days = len(model_args['dataset']['sessions'])
        day_weights = []
        
        for i in range(n_days):
            weights = model.day_weights[i].detach().cpu().numpy()
            day_weights.append(weights)
        
        day_weights = np.array(day_weights)  # [days, 512, 512]
        
        # Compute key importance metrics
        input_importance = np.mean(np.abs(day_weights), axis=(0, 2))  # [512]
        self_connection = np.mean(np.abs(np.diagonal(day_weights, axis1=1, axis2=2)), axis=0)  # [512]
        
        return {
            'input_importance': input_importance,
            'self_connection': self_connection
        }
        
    except Exception as e:
        print(f"Error loading importance metrics: {e}")
        return None

def get_electrode_group_name(feature_idx):
    """Get the electrode group name for a feature index"""
    if 0 <= feature_idx < 64:
        return "ventral_6v_thresh"
    elif 65 <= feature_idx < 128:
        return "area_4_thresh"
    elif 129 <= feature_idx < 192:
        return "55b_thresh"
    elif 193 <= feature_idx < 256:
        return "dorsal_6v_thresh"
    elif 257 <= feature_idx < 320:
        return "ventral_6v_power"
    elif 321 <= feature_idx < 384:
        return "area_4_power"
    elif 385 <= feature_idx < 448:
        return "55b_power"
    elif 449 <= feature_idx < 512:
        return "dorsal_6v_power"
    else:
        return "unknown"

def get_electrode_group_color(feature_idx):
    """Get color for electrode group"""
    colors = {
        'ventral_6v_thresh': 'red',
        'area_4_thresh': 'orange', 
        '55b_thresh': 'yellow',
        'dorsal_6v_thresh': 'green',
        'ventral_6v_power': 'cyan',
        'area_4_power': 'blue',
        '55b_power': 'purple',
        'dorsal_6v_power': 'pink'
    }
    
    group_name = get_electrode_group_name(feature_idx)
    return colors.get(group_name, 'black')

def analyze_cv_importance_correlation(feature_cvs, importance_metrics):
    """Analyze correlation between CV and importance"""
    print("\n=== CV vs IMPORTANCE CORRELATION ANALYSIS ===")
    
    results = {}
    
    for metric_name, importance_values in importance_metrics.items():
        # Compute correlations
        pearson_r, pearson_p = pearsonr(feature_cvs, importance_values)
        spearman_r, spearman_p = spearmanr(feature_cvs, importance_values)
        
        results[metric_name] = {
            'pearson_r': pearson_r,
            'pearson_p': pearson_p,
            'spearman_r': spearman_r,
            'spearman_p': spearman_p
        }
        
        print(f"{metric_name}:")
        print(f"  Pearson correlation: r={pearson_r:.4f}, p={pearson_p:.4f}")
        print(f"  Spearman correlation: r={spearman_r:.4f}, p={spearman_p:.4f}")
    
    return results

def analyze_extreme_features(feature_cvs, importance_metrics):
    """Analyze extreme CV features and their importance"""
    print("\n=== EXTREME FEATURE ANALYSIS ===")
    
    # Define thresholds
    high_cv_threshold = np.percentile(feature_cvs, 95)  # Top 5% most variable
    low_cv_threshold = np.percentile(feature_cvs, 5)    # Bottom 5% least variable
    
    high_cv_features = np.where(feature_cvs > high_cv_threshold)[0]
    low_cv_features = np.where(feature_cvs < low_cv_threshold)[0]
    
    print(f"High CV features (top 5%): {len(high_cv_features)} features")
    print(f"Low CV features (bottom 5%): {len(low_cv_features)} features")
    
    # Analyze importance of extreme features
    for metric_name, importance_values in importance_metrics.items():
        high_cv_importance = importance_values[high_cv_features]
        low_cv_importance = importance_values[low_cv_features]
        median_importance = np.median(importance_values)
        
        print(f"\n{metric_name}:")
        print(f"  High CV features - Mean importance: {np.mean(high_cv_importance):.6f}")
        print(f"  Low CV features - Mean importance: {np.mean(low_cv_importance):.6f}")
        print(f"  Overall median importance: {median_importance:.6f}")
        
        # Check if high CV features are below median importance
        high_cv_below_median = np.sum(high_cv_importance < median_importance)
        low_cv_below_median = np.sum(low_cv_importance < median_importance)
        
        print(f"  High CV features below median: {high_cv_below_median}/{len(high_cv_features)} ({high_cv_below_median/len(high_cv_features)*100:.1f}%)")
        print(f"  Low CV features below median: {low_cv_below_median}/{len(low_cv_features)} ({low_cv_below_median/len(low_cv_features)*100:.1f}%)")
    
    return high_cv_features, low_cv_features

def analyze_by_electrode_groups(feature_cvs, importance_metrics):
    """Analyze CV vs importance by electrode groups"""
    print("\n=== ELECTRODE GROUP CV vs IMPORTANCE ===")
    
    electrode_groups = {
        'ventral_6v_thresh': (0, 64),
        'area_4_thresh': (65, 128), 
        '55b_thresh': (129, 192),
        'dorsal_6v_thresh': (193, 256),
        'ventral_6v_power': (257, 320),
        'area_4_power': (321, 384),
        '55b_power': (385, 448),
        'dorsal_6v_power': (449, 512)
    }
    
    group_analysis = {}
    
    for group_name, (start, end) in electrode_groups.items():
        group_cvs = feature_cvs[start:end]
        group_analysis[group_name] = {'cv_mean': np.mean(group_cvs)}
        
        for metric_name, importance_values in importance_metrics.items():
            group_importance = importance_values[start:end]
            group_analysis[group_name][f'{metric_name}_mean'] = np.mean(group_importance)
            
            # Correlation within group
            if len(group_cvs) > 5:  # Need enough points for correlation
                try:
                    r, p = pearsonr(group_cvs, group_importance)
                    group_analysis[group_name][f'{metric_name}_cv_corr'] = r
                except:
                    group_analysis[group_name][f'{metric_name}_cv_corr'] = np.nan
        
        print(f"{group_name}:")
        print(f"  Mean CV: {group_analysis[group_name]['cv_mean']:.4f}")
        for metric_name in importance_metrics.keys():
            mean_imp = group_analysis[group_name][f'{metric_name}_mean']
            corr = group_analysis[group_name].get(f'{metric_name}_cv_corr', np.nan)
            print(f"  {metric_name}: {mean_imp:.6f} (CV corr: {corr:.3f})")
    
    return group_analysis

def create_cv_importance_visualizations(feature_cvs, importance_metrics, correlation_results, high_cv_features, low_cv_features):
    """Create visualizations for CV vs importance analysis"""
    
    plt.figure(figsize=(20, 12))
    
    # 1. CV vs Input Importance scatter plot
    plt.subplot(3, 4, 1)
    if 'input_importance' in importance_metrics:
        colors = [get_electrode_group_color(i) for i in range(512)]
        plt.scatter(feature_cvs, importance_metrics['input_importance'], 
                   c=colors, alpha=0.6, s=20)
        plt.xlabel('Feature CV')
        plt.ylabel('Input Importance')
        plt.title('CV vs Input Importance')
        plt.xscale('log')
        plt.yscale('log')
        
        # Add correlation info
        if 'input_importance' in correlation_results:
            r = correlation_results['input_importance']['pearson_r']
            plt.text(0.05, 0.95, f'r={r:.3f}', transform=plt.gca().transAxes, 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # 2. CV vs Self Connection scatter plot
    plt.subplot(3, 4, 2)
    if 'self_connection' in importance_metrics:
        colors = [get_electrode_group_color(i) for i in range(512)]
        plt.scatter(feature_cvs, importance_metrics['self_connection'], 
                   c=colors, alpha=0.6, s=20)
        plt.xlabel('Feature CV')
        plt.ylabel('Self Connection Strength')
        plt.title('CV vs Self Connection')
        plt.xscale('log')
        
        # Add correlation info
        if 'self_connection' in correlation_results:
            r = correlation_results['self_connection']['pearson_r']
            plt.text(0.05, 0.95, f'r={r:.3f}', transform=plt.gca().transAxes,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # 3. High CV vs Low CV features comparison
    plt.subplot(3, 4, 3)
    if 'input_importance' in importance_metrics:
        high_cv_imp = importance_metrics['input_importance'][high_cv_features]
        low_cv_imp = importance_metrics['input_importance'][low_cv_features]
        
        plt.boxplot([high_cv_imp, low_cv_imp], labels=['High CV\n(top 5%)', 'Low CV\n(bottom 5%)'])
        plt.ylabel('Input Importance')
        plt.title('Importance: High vs Low CV Features')
        plt.yscale('log')
    
    # 4. Feature CV distribution with importance overlay
    plt.subplot(3, 4, 4)
    plt.hist(feature_cvs, bins=50, alpha=0.7, color='lightblue', edgecolor='black')
    plt.xlabel('Feature CV')
    plt.ylabel('Count')
    plt.title('Feature CV Distribution')
    plt.xscale('log')
    
    # Highlight extreme features
    plt.axvline(np.percentile(feature_cvs, 95), color='red', linestyle='--', label='95th percentile')
    plt.axvline(np.percentile(feature_cvs, 5), color='blue', linestyle='--', label='5th percentile')
    plt.legend()
    
    # 5. Electrode group CV vs Importance
    plt.subplot(3, 4, 5)
    electrode_groups = ['ventral_6v_thresh', 'area_4_thresh', '55b_thresh', 'dorsal_6v_thresh',
                       'ventral_6v_power', 'area_4_power', '55b_power', 'dorsal_6v_power']
    
    group_cvs = []
    group_importances = []
    
    for group_name in electrode_groups:
        start, end = [(0, 64), (65, 128), (129, 192), (193, 256), 
                     (257, 320), (321, 384), (385, 448), (449, 512)][electrode_groups.index(group_name)]
        group_cvs.append(np.mean(feature_cvs[start:end]))
        if 'input_importance' in importance_metrics:
            group_importances.append(np.mean(importance_metrics['input_importance'][start:end]))
    
    if group_importances:
        colors_groups = ['red', 'orange', 'yellow', 'green', 'cyan', 'blue', 'purple', 'pink']
        plt.scatter(group_cvs, group_importances, c=colors_groups, s=100, alpha=0.8)
        
        # Add labels
        for i, group in enumerate(electrode_groups):
            plt.annotate(group.replace('_', '\n'), (group_cvs[i], group_importances[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        plt.xlabel('Mean Group CV')
        plt.ylabel('Mean Group Importance')
        plt.title('Electrode Group: CV vs Importance')
    
    # 6. Top 10 highest CV features
    plt.subplot(3, 4, 6)
    top_cv_indices = np.argsort(feature_cvs)[-10:][::-1]
    top_cv_values = feature_cvs[top_cv_indices]
    colors_top = [get_electrode_group_color(idx) for idx in top_cv_indices]
    
    plt.bar(range(10), top_cv_values, color=colors_top, alpha=0.7)
    plt.xlabel('Rank')
    plt.ylabel('CV')
    plt.title('Top 10 Highest CV Features')
    plt.yscale('log')
    
    # Add feature indices as labels
    for i, idx in enumerate(top_cv_indices):
        plt.text(i, top_cv_values[i], str(idx), ha='center', va='bottom', fontsize=8)
    
    # 7. Bottom 10 lowest CV features
    plt.subplot(3, 4, 7)
    bottom_cv_indices = np.argsort(feature_cvs)[:10]
    bottom_cv_values = feature_cvs[bottom_cv_indices]
    colors_bottom = [get_electrode_group_color(idx) for idx in bottom_cv_indices]
    
    plt.bar(range(10), bottom_cv_values, color=colors_bottom, alpha=0.7)
    plt.xlabel('Rank')
    plt.ylabel('CV')
    plt.title('Bottom 10 Lowest CV Features')
    
    # Add feature indices as labels
    for i, idx in enumerate(bottom_cv_indices):
        plt.text(i, bottom_cv_values[i], str(idx), ha='center', va='bottom', fontsize=8)
    
    # 8. CV by feature index with importance overlay
    plt.subplot(3, 4, 8)
    plt.plot(feature_cvs, alpha=0.7, label='CV')
    if 'input_importance' in importance_metrics:
        # Normalize importance to same scale as CV for visualization
        norm_importance = importance_metrics['input_importance'] * np.max(feature_cvs) / np.max(importance_metrics['input_importance'])
        plt.plot(norm_importance, alpha=0.7, label='Importance (scaled)')
    
    plt.xlabel('Feature Index')
    plt.ylabel('Value')
    plt.title('CV and Importance by Feature Index')
    plt.yscale('log')
    plt.legend()
    
    # Add electrode group boundaries
    boundaries = [64, 128, 192, 256, 320, 384, 448]
    for b in boundaries:
        plt.axvline(b, color='gray', linestyle='--', alpha=0.3)
    
    # 9. Correlation summary
    plt.subplot(3, 4, 9)
    if correlation_results:
        metrics = list(correlation_results.keys())
        correlations = [correlation_results[m]['pearson_r'] for m in metrics]
        p_values = [correlation_results[m]['pearson_p'] for m in metrics]
        
        bars = plt.bar(range(len(metrics)), correlations, alpha=0.7)
        plt.xlabel('Importance Metric')
        plt.ylabel('Correlation with CV')
        plt.title('CV-Importance Correlations')
        plt.xticks(range(len(metrics)), [m.replace('_', '\n') for m in metrics])
        
        # Color bars by significance
        for i, (bar, p) in enumerate(zip(bars, p_values)):
            if p < 0.001:
                bar.set_color('red')
            elif p < 0.01:
                bar.set_color('orange')
            elif p < 0.05:
                bar.set_color('yellow')
            else:
                bar.set_color('gray')
        
        plt.axhline(0, color='black', linestyle='-', alpha=0.5)
    
    # 10. Feature importance rank vs CV rank
    plt.subplot(3, 4, 10)
    if 'input_importance' in importance_metrics:
        cv_ranks = np.argsort(np.argsort(feature_cvs))  # Rank of CV (0 = lowest CV)
        importance_ranks = np.argsort(np.argsort(importance_metrics['input_importance']))  # Rank of importance
        
        colors = [get_electrode_group_color(i) for i in range(512)]
        plt.scatter(cv_ranks, importance_ranks, c=colors, alpha=0.6, s=10)
        plt.xlabel('CV Rank (0 = lowest CV)')
        plt.ylabel('Importance Rank (0 = lowest importance)')
        plt.title('Feature Ranks: CV vs Importance')
        
        # Add diagonal line (perfect correlation)
        plt.plot([0, 511], [0, 511], 'k--', alpha=0.5, label='Perfect correlation')
        plt.legend()
    
    # 11. Extreme features analysis
    plt.subplot(3, 4, 11)
    if 'input_importance' in importance_metrics:
        # Create categories
        categories = ['High CV\nHigh Imp', 'High CV\nLow Imp', 'Low CV\nHigh Imp', 'Low CV\nLow Imp']
        
        high_imp_threshold = np.percentile(importance_metrics['input_importance'], 75)
        low_imp_threshold = np.percentile(importance_metrics['input_importance'], 25)
        
        high_cv_high_imp = np.sum((feature_cvs > np.percentile(feature_cvs, 75)) & 
                                 (importance_metrics['input_importance'] > high_imp_threshold))
        high_cv_low_imp = np.sum((feature_cvs > np.percentile(feature_cvs, 75)) & 
                                (importance_metrics['input_importance'] < low_imp_threshold))
        low_cv_high_imp = np.sum((feature_cvs < np.percentile(feature_cvs, 25)) & 
                                (importance_metrics['input_importance'] > high_imp_threshold))
        low_cv_low_imp = np.sum((feature_cvs < np.percentile(feature_cvs, 25)) & 
                               (importance_metrics['input_importance'] < low_imp_threshold))
        
        counts = [high_cv_high_imp, high_cv_low_imp, low_cv_high_imp, low_cv_low_imp]
        colors_cat = ['red', 'orange', 'blue', 'green']
        
        plt.bar(range(4), counts, color=colors_cat, alpha=0.7)
        plt.xlabel('Feature Category')
        plt.ylabel('Number of Features')
        plt.title('Extreme Features Analysis')
        plt.xticks(range(4), categories)
    
    # 12. Model's handling of extreme CV features
    plt.subplot(3, 4, 12)
    if 'input_importance' in importance_metrics:
        # Show how model importance changes with CV
        cv_bins = np.logspace(np.log10(np.min(feature_cvs)), np.log10(np.max(feature_cvs)), 20)
        bin_centers = (cv_bins[:-1] + cv_bins[1:]) / 2
        
        binned_importance = []
        for i in range(len(cv_bins) - 1):
            mask = (feature_cvs >= cv_bins[i]) & (feature_cvs < cv_bins[i+1])
            if np.sum(mask) > 0:
                binned_importance.append(np.mean(importance_metrics['input_importance'][mask]))
            else:
                binned_importance.append(np.nan)
        
        plt.plot(bin_centers, binned_importance, 'o-', alpha=0.7)
        plt.xlabel('CV Bin Center')
        plt.ylabel('Mean Importance in Bin')
        plt.title('Model Importance vs CV Bins')
        plt.xscale('log')
        plt.yscale('log')
    
    plt.tight_layout()
    plt.savefig('cv_importance_correlation.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Main CV vs importance correlation analysis"""
    print("Starting CV vs Importance correlation analysis...")
    
    # Load data
    feature_cvs = load_feature_cvs()
    importance_metrics = load_importance_metrics()
    
    if importance_metrics is None:
        print("Failed to load importance metrics!")
        return
    
    print(f"Loaded CVs for {len(feature_cvs)} features")
    print(f"Loaded importance metrics: {list(importance_metrics.keys())}")
    
    # Analyze correlations
    correlation_results = analyze_cv_importance_correlation(feature_cvs, importance_metrics)
    
    # Analyze extreme features
    high_cv_features, low_cv_features = analyze_extreme_features(feature_cvs, importance_metrics)
    
    # Analyze by electrode groups
    group_analysis = analyze_by_electrode_groups(feature_cvs, importance_metrics)
    
    # Create visualizations
    create_cv_importance_visualizations(feature_cvs, importance_metrics, correlation_results, 
                                      high_cv_features, low_cv_features)
    
    # Final conclusions
    print(f"\n{'='*80}")
    print("CV vs IMPORTANCE CORRELATION CONCLUSIONS")
    print(f"{'='*80}")
    
    # Check if model is effectively ignoring high CV features
    if 'input_importance' in correlation_results:
        r = correlation_results['input_importance']['pearson_r']
        p = correlation_results['input_importance']['pearson_p']
        
        print(f"Overall CV vs Importance correlation: r={r:.4f}, p={p:.4f}")
        
        if r < -0.3 and p < 0.05:
            print("🎯 STRONG NEGATIVE CORRELATION - Model effectively ignores high CV features!")
        elif r < -0.1 and p < 0.05:
            print("📊 MODERATE NEGATIVE CORRELATION - Model somewhat downweights high CV features")
        elif abs(r) < 0.1:
            print("🤷 NO CORRELATION - Model treats all features equally regardless of CV")
        elif r > 0.1 and p < 0.05:
            print("⚠️  POSITIVE CORRELATION - Model actually uses high CV features more!")
        else:
            print("📈 WEAK/INSIGNIFICANT CORRELATION")
    
    # Analyze extreme features
    if 'input_importance' in importance_metrics:
        high_cv_imp = importance_metrics['input_importance'][high_cv_features]
        low_cv_imp = importance_metrics['input_importance'][low_cv_features]
        median_imp = np.median(importance_metrics['input_importance'])
        
        high_cv_below_median = np.sum(high_cv_imp < median_imp) / len(high_cv_imp) * 100
        low_cv_below_median = np.sum(low_cv_imp < median_imp) / len(low_cv_imp) * 100
        
        print(f"\nExtreme feature analysis:")
        print(f"  High CV features below median importance: {high_cv_below_median:.1f}%")
        print(f"  Low CV features below median importance: {low_cv_below_median:.1f}%")
        
        if high_cv_below_median > 70:
            print("  ✅ Model successfully downweights most high CV features")
        elif high_cv_below_median > 50:
            print("  📊 Model moderately downweights high CV features")
        else:
            print("  ⚠️  Model does not effectively downweight high CV features")
    
    print(f"\nAnalysis complete! Check 'cv_importance_correlation.png' for detailed visualizations.")

if __name__ == "__main__":
    main()
