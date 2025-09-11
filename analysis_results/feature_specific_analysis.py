#!/usr/bin/env python3
"""
Feature-specific analysis to understand:
1. Is the extreme CV (26247) from one outlier feature or systematic?
2. Which specific features are most problematic?
3. Are there patterns by electrode group and feature type?
4. Analysis of the RNN architecture for feature-specific processing
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import h5py
from tqdm import tqdm

# Add parent directory to path
sys.path.append('..')
sys.path.append('../model_training')

def load_comprehensive_session_data(sessions, data_dir, max_sessions=20):
    """Load comprehensive data for feature analysis"""
    print(f"Loading comprehensive data from {max_sessions} sessions...")
    
    session_data = {}
    
    for session in tqdm(sessions[:max_sessions], desc="Loading sessions"):
        session_path = os.path.join(data_dir, session)
        
        # Try to load training data
        train_file = os.path.join(session_path, 'data_train.hdf5')
        if os.path.exists(train_file):
            try:
                with h5py.File(train_file, 'r') as f:
                    trials = list(f.keys())[:30]  # More trials for better statistics
                    
                    neural_features = []
                    for trial_key in trials:
                        trial_data = f[trial_key]['input_features'][:]
                        neural_features.append(trial_data)
                    
                    if neural_features:
                        # Concatenate all trials
                        all_data = np.concatenate(neural_features, axis=0)
                        
                        session_data[session] = {
                            'raw_data': all_data,
                            'n_trials': len(neural_features),
                            'n_timepoints': all_data.shape[0]
                        }
                        
            except Exception as e:
                print(f"Error loading {session}: {e}")
                continue
    
    print(f"Successfully loaded {len(session_data)} sessions")
    return session_data

def analyze_feature_outliers(session_data):
    """Detailed analysis of feature outliers and extreme CV values"""
    print("\n=== FEATURE OUTLIER ANALYSIS ===")
    
    sessions = list(session_data.keys())
    n_features = 512
    
    # Collect detailed feature statistics
    all_feature_means = []  # [sessions, features]
    all_feature_stds = []
    all_feature_medians = []
    
    for session in sessions:
        data = session_data[session]['raw_data']  # [time, features]
        
        feature_means = np.mean(data, axis=0)
        feature_stds = np.std(data, axis=0)
        feature_medians = np.median(data, axis=0)
        
        all_feature_means.append(feature_means)
        all_feature_stds.append(feature_stds)
        all_feature_medians.append(feature_medians)
    
    all_feature_means = np.array(all_feature_means)  # [sessions, features]
    all_feature_stds = np.array(all_feature_stds)
    all_feature_medians = np.array(all_feature_medians)
    
    # Compute CV for each feature across sessions
    feature_cvs = np.std(all_feature_means, axis=0) / (np.abs(np.mean(all_feature_means, axis=0)) + 1e-8)
    
    # Find extreme outliers
    extreme_threshold = 1000  # CV > 1000
    extreme_features = np.where(feature_cvs > extreme_threshold)[0]
    
    print(f"Features with extreme CV (> {extreme_threshold}): {len(extreme_features)}")
    
    # Analyze the most extreme features
    top_10_extreme = np.argsort(feature_cvs)[-10:][::-1]  # Top 10 most variable
    
    print(f"\nTop 10 most variable features:")
    for i, feat_idx in enumerate(top_10_extreme):
        cv_val = feature_cvs[feat_idx]
        mean_val = np.mean(all_feature_means[:, feat_idx])
        std_val = np.std(all_feature_means[:, feat_idx])
        
        # Determine electrode group
        group_name = get_electrode_group_name(feat_idx)
        
        print(f"  {i+1:2d}. Feature {feat_idx:3d} ({group_name}): CV={cv_val:.2f}, mean={mean_val:.6f}, std={std_val:.6f}")
        
        # Check if this is due to near-zero means
        session_means = all_feature_means[:, feat_idx]
        near_zero_sessions = np.sum(np.abs(session_means) < 1e-6)
        print(f"      Sessions with near-zero mean: {near_zero_sessions}/{len(sessions)}")
    
    # Analyze distribution of CV values
    cv_percentiles = np.percentile(feature_cvs, [50, 75, 90, 95, 99, 99.9])
    print(f"\nCV Distribution Percentiles:")
    print(f"  50th: {cv_percentiles[0]:.2f}")
    print(f"  75th: {cv_percentiles[1]:.2f}")
    print(f"  90th: {cv_percentiles[2]:.2f}")
    print(f"  95th: {cv_percentiles[3]:.2f}")
    print(f"  99th: {cv_percentiles[4]:.2f}")
    print(f"  99.9th: {cv_percentiles[5]:.2f}")
    
    return {
        'feature_cvs': feature_cvs,
        'all_feature_means': all_feature_means,
        'all_feature_stds': all_feature_stds,
        'extreme_features': extreme_features,
        'top_10_extreme': top_10_extreme,
        'sessions': sessions
    }

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

def analyze_electrode_group_patterns(outlier_results):
    """Analyze patterns by electrode group and feature type"""
    print("\n=== ELECTRODE GROUP PATTERN ANALYSIS ===")
    
    feature_cvs = outlier_results['feature_cvs']
    
    # Define electrode groups
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
        
        group_stats = {
            'mean_cv': np.mean(group_cvs),
            'median_cv': np.median(group_cvs),
            'max_cv': np.max(group_cvs),
            'min_cv': np.min(group_cvs),
            'std_cv': np.std(group_cvs),
            'n_extreme': np.sum(group_cvs > 100),  # Features with CV > 100
            'n_high': np.sum(group_cvs > 10),      # Features with CV > 10
            'n_moderate': np.sum(group_cvs > 1),   # Features with CV > 1
            'worst_feature_idx': start + np.argmax(group_cvs),
            'worst_feature_cv': np.max(group_cvs)
        }
        
        group_analysis[group_name] = group_stats
        
        print(f"{group_name}:")
        print(f"  Mean CV: {group_stats['mean_cv']:.2f}")
        print(f"  Median CV: {group_stats['median_cv']:.2f}")
        print(f"  Max CV: {group_stats['max_cv']:.2f} (feature {group_stats['worst_feature_idx']})")
        print(f"  Features with CV > 100: {group_stats['n_extreme']}/64")
        print(f"  Features with CV > 10: {group_stats['n_high']}/64")
        print(f"  Features with CV > 1: {group_stats['n_moderate']}/64")
    
    # Compare threshold vs power features
    thresh_groups = ['ventral_6v_thresh', 'area_4_thresh', '55b_thresh', 'dorsal_6v_thresh']
    power_groups = ['ventral_6v_power', 'area_4_power', '55b_power', 'dorsal_6v_power']
    
    thresh_cvs = np.concatenate([feature_cvs[electrode_groups[g][0]:electrode_groups[g][1]] for g in thresh_groups])
    power_cvs = np.concatenate([feature_cvs[electrode_groups[g][0]:electrode_groups[g][1]] for g in power_groups])
    
    print(f"\nThreshold vs Power Feature Comparison:")
    print(f"  Threshold features - Mean CV: {np.mean(thresh_cvs):.2f}, Median CV: {np.median(thresh_cvs):.2f}")
    print(f"  Power features - Mean CV: {np.mean(power_cvs):.2f}, Median CV: {np.median(power_cvs):.2f}")
    
    return group_analysis

def analyze_rnn_architecture():
    """Analyze the RNN architecture for feature-specific processing"""
    print("\n=== RNN ARCHITECTURE ANALYSIS ===")
    
    try:
        from rnn_model import GRUDecoder
        
        # Create a dummy model to inspect architecture
        model = GRUDecoder(
            neural_dim=512,
            n_units=768,
            n_days=45,
            n_classes=41,
            n_layers=5
        )
        
        print("RNN Model Architecture:")
        print(f"  Input dimension: {model.neural_dim}")
        print(f"  Hidden units per layer: {model.n_units}")
        print(f"  Number of layers: {model.n_layers}")
        print(f"  Output classes: {model.n_classes}")
        print(f"  Number of day-specific transformations: {len(model.day_weights)}")
        
        # Analyze day-specific layers
        day_layer_shape = model.day_weights[0].shape
        print(f"\nDay-specific transformation:")
        print(f"  Weight matrix shape: {day_layer_shape}")
        print(f"  This is a FULL {day_layer_shape[0]}x{day_layer_shape[1]} transformation")
        print(f"  Each feature can affect every other feature!")
        
        # Check if there are feature-specific heads
        output_layer_shape = model.out.weight.shape
        print(f"\nOutput layer:")
        print(f"  Weight shape: {output_layer_shape}")
        print(f"  This is a single shared output head for all features")
        
        # Analysis
        print(f"\n🔍 ARCHITECTURE INSIGHTS:")
        print(f"  ❌ NO feature-specific heads - single output layer processes all features together")
        print(f"  ❌ Day transformations are FULL matrices - any feature can be transformed using any other feature")
        print(f"  ⚠️  This allows the model to 'route around' bad features using good ones")
        print(f"  ⚠️  Model can learn complex feature interactions to compensate for recording artifacts")
        
        return {
            'has_feature_specific_heads': False,
            'day_transformation_type': 'full_matrix',
            'input_dim': model.neural_dim,
            'output_dim': model.n_classes,
            'day_weight_shape': day_layer_shape,
            'output_weight_shape': output_layer_shape
        }
        
    except Exception as e:
        print(f"Error analyzing RNN architecture: {e}")
        return None

def create_feature_analysis_visualizations(outlier_results, group_analysis):
    """Create comprehensive visualizations"""
    
    plt.figure(figsize=(20, 15))
    
    feature_cvs = outlier_results['feature_cvs']
    
    # 1. CV distribution by feature index
    plt.subplot(3, 4, 1)
    plt.plot(feature_cvs, alpha=0.7)
    plt.xlabel('Feature Index')
    plt.ylabel('Coefficient of Variation')
    plt.title('CV by Feature Index')
    plt.yscale('log')
    
    # Add electrode group boundaries
    boundaries = [64, 128, 192, 256, 320, 384, 448, 512]
    colors = ['red', 'orange', 'yellow', 'green', 'cyan', 'blue', 'purple', 'pink']
    for i, b in enumerate(boundaries[:-1]):
        plt.axvline(b, color=colors[i], linestyle='--', alpha=0.5)
    
    # 2. CV histogram (log scale)
    plt.subplot(3, 4, 2)
    plt.hist(feature_cvs[feature_cvs > 0], bins=50, alpha=0.7, edgecolor='black')
    plt.xlabel('Coefficient of Variation')
    plt.ylabel('Number of Features')
    plt.title('CV Distribution (Log Scale)')
    plt.xscale('log')
    
    # 3. Box plot by electrode group
    plt.subplot(3, 4, 3)
    group_names = list(group_analysis.keys())
    group_data = []
    
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
    
    for group_name in group_names:
        start, end = electrode_groups[group_name]
        group_cvs = feature_cvs[start:end]
        group_data.append(group_cvs)
    
    box_plot = plt.boxplot(group_data, labels=[g.replace('_', '\n') for g in group_names])
    plt.yscale('log')
    plt.ylabel('CV (log scale)')
    plt.title('CV by Electrode Group')
    plt.xticks(rotation=45)
    
    # 4. Top 20 most variable features
    plt.subplot(3, 4, 4)
    top_20 = np.argsort(feature_cvs)[-20:][::-1]
    top_20_cvs = feature_cvs[top_20]
    colors_top = [get_electrode_group_color(idx) for idx in top_20]
    
    plt.bar(range(20), top_20_cvs, color=colors_top, alpha=0.7)
    plt.xlabel('Rank')
    plt.ylabel('CV')
    plt.title('Top 20 Most Variable Features')
    plt.yscale('log')
    
    # Add legend for colors
    group_colors = {
        'ventral_6v_thresh': 'red',
        'area_4_thresh': 'orange', 
        '55b_thresh': 'yellow',
        'dorsal_6v_thresh': 'green',
        'ventral_6v_power': 'cyan',
        'area_4_power': 'blue',
        '55b_power': 'purple',
        'dorsal_6v_power': 'pink'
    }
    
    # 5. Feature means across sessions heatmap (subset)
    plt.subplot(3, 4, 5)
    feature_means = outlier_results['all_feature_means']
    # Show every 8th feature for visibility
    subset_means = feature_means[:, ::8]
    im = plt.imshow(subset_means.T, aspect='auto', cmap='viridis')
    plt.colorbar(im)
    plt.xlabel('Session Index')
    plt.ylabel('Feature Index (every 8th)')
    plt.title('Feature Means Across Sessions')
    
    # 6. Threshold vs Power comparison
    plt.subplot(3, 4, 6)
    thresh_groups = ['ventral_6v_thresh', 'area_4_thresh', '55b_thresh', 'dorsal_6v_thresh']
    power_groups = ['ventral_6v_power', 'area_4_power', '55b_power', 'dorsal_6v_power']
    
    thresh_cvs = np.concatenate([feature_cvs[electrode_groups[g][0]:electrode_groups[g][1]] for g in thresh_groups])
    power_cvs = np.concatenate([feature_cvs[electrode_groups[g][0]:electrode_groups[g][1]] for g in power_groups])
    
    plt.hist([thresh_cvs, power_cvs], bins=50, alpha=0.7, label=['Threshold', 'Power'], color=['blue', 'red'])
    plt.xlabel('CV')
    plt.ylabel('Count')
    plt.title('Threshold vs Power Features')
    plt.legend()
    plt.xscale('log')
    
    # 7. Session correlation matrix (subset)
    plt.subplot(3, 4, 7)
    # Compute correlation between first 10 sessions
    n_sessions_to_show = min(10, feature_means.shape[0])
    session_corr = np.corrcoef(feature_means[:n_sessions_to_show])
    
    im = plt.imshow(session_corr, cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar(im)
    plt.xlabel('Session Index')
    plt.ylabel('Session Index')
    plt.title(f'Session Correlation Matrix (first {n_sessions_to_show})')
    
    # 8. Feature stability over sessions
    plt.subplot(3, 4, 8)
    # Show how the most variable feature changes over sessions
    worst_feature_idx = np.argmax(feature_cvs)
    worst_feature_values = feature_means[:, worst_feature_idx]
    
    plt.plot(worst_feature_values, 'o-', alpha=0.7)
    plt.xlabel('Session Index')
    plt.ylabel('Feature Value')
    plt.title(f'Most Variable Feature ({worst_feature_idx}) Over Sessions')
    
    # 9. CV vs mean scatter
    plt.subplot(3, 4, 9)
    overall_means = np.mean(feature_means, axis=0)
    plt.scatter(np.abs(overall_means), feature_cvs, alpha=0.5, s=10)
    plt.xlabel('Absolute Mean Value')
    plt.ylabel('CV')
    plt.title('CV vs Mean Magnitude')
    plt.xscale('log')
    plt.yscale('log')
    
    # 10. Number of problematic features per group
    plt.subplot(3, 4, 10)
    group_names_short = [g.replace('_', '\n') for g in group_names]
    extreme_counts = [group_analysis[g]['n_extreme'] for g in group_names]
    high_counts = [group_analysis[g]['n_high'] for g in group_names]
    
    x = np.arange(len(group_names))
    width = 0.35
    
    plt.bar(x - width/2, extreme_counts, width, label='CV > 100', alpha=0.7)
    plt.bar(x + width/2, high_counts, width, label='CV > 10', alpha=0.7)
    
    plt.xlabel('Electrode Group')
    plt.ylabel('Number of Features')
    plt.title('Problematic Features by Group')
    plt.xticks(x, group_names_short, rotation=45)
    plt.legend()
    
    # 11. Feature index vs CV colored by group
    plt.subplot(3, 4, 11)
    colors_all = [get_electrode_group_color(i) for i in range(512)]
    plt.scatter(range(512), feature_cvs, c=colors_all, alpha=0.6, s=10)
    plt.xlabel('Feature Index')
    plt.ylabel('CV')
    plt.title('CV by Feature (colored by group)')
    plt.yscale('log')
    
    # 12. Cumulative distribution of CV
    plt.subplot(3, 4, 12)
    sorted_cvs = np.sort(feature_cvs)
    cumulative = np.arange(1, len(sorted_cvs) + 1) / len(sorted_cvs)
    plt.plot(sorted_cvs, cumulative)
    plt.xlabel('CV Threshold')
    plt.ylabel('Fraction of Features Below')
    plt.title('Cumulative CV Distribution')
    plt.xscale('log')
    
    plt.tight_layout()
    plt.savefig('feature_specific_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

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

def main():
    """Main feature-specific analysis"""
    print("Starting comprehensive feature-specific analysis...")
    
    # Load data
    data_dir = '../data/t15_copyTask_neuralData/hdf5_data_final'
    
    # Get sessions
    sessions = []
    for item in os.listdir(data_dir):
        if item.startswith('t15.') and os.path.isdir(os.path.join(data_dir, item)):
            sessions.append(item)
    
    sessions.sort()
    
    # Load comprehensive data
    session_data = load_comprehensive_session_data(sessions, data_dir)
    
    if len(session_data) < 2:
        print("Not enough sessions!")
        return
    
    # Analyze feature outliers
    outlier_results = analyze_feature_outliers(session_data)
    
    # Analyze electrode group patterns
    group_analysis = analyze_electrode_group_patterns(outlier_results)
    
    # Analyze RNN architecture
    architecture_analysis = analyze_rnn_architecture()
    
    # Create visualizations
    create_feature_analysis_visualizations(outlier_results, group_analysis)
    
    # Final conclusions
    print(f"\n{'='*80}")
    print("FEATURE-SPECIFIC ANALYSIS CONCLUSIONS")
    print(f"{'='*80}")
    
    feature_cvs = outlier_results['feature_cvs']
    max_cv = np.max(feature_cvs)
    max_cv_idx = np.argmax(feature_cvs)
    
    print(f"Maximum CV: {max_cv:.2f} (Feature {max_cv_idx} - {get_electrode_group_name(max_cv_idx)})")
    
    # Check if extreme CVs are systematic or outliers
    extreme_features = np.sum(feature_cvs > 1000)
    very_high_features = np.sum(feature_cvs > 100)
    high_features = np.sum(feature_cvs > 10)
    
    print(f"\nFeature variability breakdown:")
    print(f"  CV > 1000 (extreme): {extreme_features} features ({extreme_features/512*100:.1f}%)")
    print(f"  CV > 100 (very high): {very_high_features} features ({very_high_features/512*100:.1f}%)")
    print(f"  CV > 10 (high): {high_features} features ({high_features/512*100:.1f}%)")
    
    if extreme_features > 10:
        print("🚨 SYSTEMATIC EXTREME VARIABILITY - Not just outliers!")
    elif extreme_features > 0:
        print("⚠️  Some extreme outlier features present")
    
    # Architecture conclusions
    if architecture_analysis:
        print(f"\nRNN Architecture Assessment:")
        if not architecture_analysis['has_feature_specific_heads']:
            print("❌ NO feature-specific processing heads")
            print("   Recommendation: Add separate processing paths for different electrode groups")
        
        if architecture_analysis['day_transformation_type'] == 'full_matrix':
            print("⚠️  Full matrix day transformations allow complex feature interactions")
            print("   This can mask individual feature problems but reduces interpretability")
    
    print(f"\nRecommendations for Kaggle competition:")
    print(f"1. 🎯 Keep day-specific transformations - they're necessary given the data quality")
    print(f"2. 🔧 Consider feature-specific processing heads for different electrode groups")
    print(f"3. 📊 Add feature importance analysis to identify which features are most reliable")
    print(f"4. 🧹 Consider feature selection or robust preprocessing to handle extreme outliers")
    print(f"5. 🔄 Experiment with different normalization strategies per electrode group")
    
    print(f"\nAnalysis complete! Check 'feature_specific_analysis.png' for detailed visualizations.")

if __name__ == "__main__":
    main()
