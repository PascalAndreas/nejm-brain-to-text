#!/usr/bin/env python3
"""
More detailed analysis of what's causing the day-specific transformation dependency.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import h5py
from tqdm import tqdm

def load_detailed_session_data(sessions, data_dir, max_sessions=15):
    """Load more detailed data from sessions"""
    print(f"Loading detailed data from {max_sessions} sessions...")
    
    session_data = {}
    
    for session in tqdm(sessions[:max_sessions], desc="Loading sessions"):
        session_path = os.path.join(data_dir, session)
        
        # Try to load training data
        train_file = os.path.join(session_path, 'data_train.hdf5')
        if os.path.exists(train_file):
            try:
                with h5py.File(train_file, 'r') as f:
                    trials = list(f.keys())[:20]  # First 20 trials
                    
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

def analyze_feature_distributions(session_data):
    """Analyze the actual distribution of features across sessions"""
    print("\n=== FEATURE DISTRIBUTION ANALYSIS ===")
    
    sessions = list(session_data.keys())
    n_features = 512
    
    # Collect feature statistics
    feature_stats = {}
    
    for session in sessions:
        data = session_data[session]['raw_data']  # [time, features]
        
        feature_stats[session] = {
            'means': np.mean(data, axis=0),
            'stds': np.std(data, axis=0),
            'medians': np.median(data, axis=0),
            'mins': np.min(data, axis=0),
            'maxs': np.max(data, axis=0),
            'q25': np.percentile(data, 25, axis=0),
            'q75': np.percentile(data, 75, axis=0)
        }
    
    # Analyze cross-session statistics
    session_means = np.array([feature_stats[s]['means'] for s in sessions])  # [sessions, features]
    session_stds = np.array([feature_stats[s]['stds'] for s in sessions])
    
    # Compute variability metrics
    feature_mean_cv = np.std(session_means, axis=0) / (np.abs(np.mean(session_means, axis=0)) + 1e-8)
    feature_std_cv = np.std(session_stds, axis=0) / (np.abs(np.mean(session_stds, axis=0)) + 1e-8)
    
    # Print summary
    print(f"Cross-session feature mean variability:")
    print(f"  Mean CV: {np.mean(feature_mean_cv):.4f}")
    print(f"  Median CV: {np.median(feature_mean_cv):.4f}")
    print(f"  Max CV: {np.max(feature_mean_cv):.4f}")
    print(f"  Features with CV > 0.1: {np.sum(feature_mean_cv > 0.1)} / {n_features}")
    print(f"  Features with CV > 0.5: {np.sum(feature_mean_cv > 0.5)} / {n_features}")
    
    return {
        'feature_stats': feature_stats,
        'session_means': session_means,
        'session_stds': session_stds,
        'feature_mean_cv': feature_mean_cv,
        'feature_std_cv': feature_std_cv,
        'sessions': sessions
    }

def analyze_signal_scaling_patterns(session_data):
    """Analyze if the differences are primarily scaling/offset issues"""
    print("\n=== SIGNAL SCALING ANALYSIS ===")
    
    sessions = list(session_data.keys())
    
    # For each pair of sessions, analyze the relationship
    scaling_analysis = {}
    
    base_session = sessions[0]
    base_data = session_data[base_session]['raw_data']
    base_features = np.mean(base_data, axis=0)  # [features]
    
    for session in sessions[1:6]:  # Analyze first 5 comparisons
        session_data_curr = session_data[session]['raw_data']
        session_features = np.mean(session_data_curr, axis=0)  # [features]
        
        # Compute correlation
        correlation = np.corrcoef(base_features, session_features)[0, 1]
        
        # Compute optimal linear scaling: session = a * base + b
        # Using least squares: session_features = a * base_features + b
        X = np.column_stack([base_features, np.ones(len(base_features))])
        coeffs = np.linalg.lstsq(X, session_features, rcond=None)[0]
        scale_factor, offset = coeffs
        
        # Compute residuals after optimal scaling
        predicted = scale_factor * base_features + offset
        residual_std = np.std(session_features - predicted)
        
        scaling_analysis[session] = {
            'correlation': correlation,
            'scale_factor': scale_factor,
            'offset': offset,
            'residual_std': residual_std,
            'explained_variance': correlation**2
        }
        
        print(f"{base_session} -> {session}:")
        print(f"  Correlation: {correlation:.4f}")
        print(f"  Scale factor: {scale_factor:.4f}")
        print(f"  Offset: {offset:.4f}")
        print(f"  Residual std: {residual_std:.4f}")
        print(f"  Explained variance: {correlation**2:.4f}")
    
    return scaling_analysis

def analyze_electrode_group_patterns(session_data):
    """Detailed analysis of electrode group patterns"""
    print("\n=== DETAILED ELECTRODE GROUP ANALYSIS ===")
    
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
    
    sessions = list(session_data.keys())
    group_analysis = {}
    
    for group_name, (start, end) in electrode_groups.items():
        session_group_means = []
        session_group_stds = []
        
        for session in sessions:
            data = session_data[session]['raw_data']
            group_data = data[:, start:end]  # [time, group_features]
            
            group_mean = np.mean(group_data)
            group_std = np.std(group_data)
            
            session_group_means.append(group_mean)
            session_group_stds.append(group_std)
        
        session_group_means = np.array(session_group_means)
        session_group_stds = np.array(session_group_stds)
        
        # Compute cross-session variability
        mean_cv = np.std(session_group_means) / (np.abs(np.mean(session_group_means)) + 1e-8)
        std_cv = np.std(session_group_stds) / (np.abs(np.mean(session_group_stds)) + 1e-8)
        
        group_analysis[group_name] = {
            'session_means': session_group_means,
            'session_stds': session_group_stds,
            'mean_cv': mean_cv,
            'std_cv': std_cv,
            'mean_range': np.max(session_group_means) - np.min(session_group_means),
            'overall_mean': np.mean(session_group_means)
        }
        
        print(f"{group_name}:")
        print(f"  Mean CV: {mean_cv:.4f}")
        print(f"  Std CV: {std_cv:.4f}")
        print(f"  Mean range: {group_analysis[group_name]['mean_range']:.4f}")
        print(f"  Overall mean: {group_analysis[group_name]['overall_mean']:.4f}")
    
    return group_analysis

def create_detailed_visualizations(distribution_results, scaling_results, electrode_results):
    """Create detailed visualizations"""
    
    plt.figure(figsize=(20, 12))
    
    # 1. Feature CV distribution
    plt.subplot(2, 4, 1)
    cv_values = distribution_results['feature_mean_cv']
    plt.hist(cv_values, bins=50, alpha=0.7, edgecolor='black')
    plt.xlabel('Coefficient of Variation')
    plt.ylabel('Number of Features')
    plt.title('Cross-Session Feature Mean CV')
    plt.axvline(0.1, color='orange', linestyle='--', label='Moderate (0.1)')
    plt.axvline(0.5, color='red', linestyle='--', label='High (0.5)')
    plt.legend()
    
    # 2. Session mean comparison heatmap
    plt.subplot(2, 4, 2)
    session_means = distribution_results['session_means']
    # Show subset of features for visualization
    subset_means = session_means[:, ::8]  # Every 8th feature
    im = plt.imshow(subset_means.T, aspect='auto', cmap='viridis')
    plt.colorbar(im)
    plt.xlabel('Session Index')
    plt.ylabel('Feature Index (subsampled)')
    plt.title('Feature Means Across Sessions')
    
    # 3. Scaling factor analysis
    plt.subplot(2, 4, 3)
    if scaling_results:
        scale_factors = [scaling_results[s]['scale_factor'] for s in scaling_results.keys()]
        correlations = [scaling_results[s]['correlation'] for s in scaling_results.keys()]
        
        plt.scatter(scale_factors, correlations, alpha=0.7)
        plt.xlabel('Scale Factor')
        plt.ylabel('Correlation')
        plt.title('Scaling vs Correlation')
        plt.axhline(0.9, color='red', linestyle='--', label='High correlation')
        plt.axvline(1.0, color='red', linestyle='--', label='No scaling')
        plt.legend()
    
    # 4. Electrode group variability
    plt.subplot(2, 4, 4)
    if electrode_results:
        group_names = list(electrode_results.keys())
        group_cvs = [electrode_results[g]['mean_cv'] for g in group_names]
        
        plt.bar(range(len(group_names)), group_cvs)
        plt.xlabel('Electrode Group')
        plt.ylabel('CV of Group Means')
        plt.title('Cross-Session Variability by Electrode Group')
        plt.xticks(range(len(group_names)), [g.replace('_', '\n') for g in group_names], rotation=45, ha='right')
    
    # 5. High variability features location
    plt.subplot(2, 4, 5)
    high_var_features = np.where(cv_values > 0.5)[0]
    plt.hist(high_var_features, bins=20, alpha=0.7, edgecolor='black')
    plt.xlabel('Feature Index')
    plt.ylabel('Count')
    plt.title('Location of High Variability Features')
    
    # Add electrode group boundaries
    boundaries = [64, 128, 192, 256, 320, 384, 448, 512]
    for b in boundaries:
        plt.axvline(b, color='red', linestyle='--', alpha=0.5)
    
    # 6. Session-to-session changes
    plt.subplot(2, 4, 6)
    sessions = distribution_results['sessions']
    overall_means = np.mean(distribution_results['session_means'], axis=1)
    session_changes = np.diff(overall_means)
    
    plt.plot(session_changes, 'o-', alpha=0.7)
    plt.xlabel('Session Transition')
    plt.ylabel('Change in Overall Mean')
    plt.title('Session-to-Session Changes')
    plt.axhline(0, color='red', linestyle='--', alpha=0.5)
    
    # 7. Feature mean vs CV scatter
    plt.subplot(2, 4, 7)
    overall_feature_means = np.mean(distribution_results['session_means'], axis=0)
    plt.scatter(overall_feature_means, cv_values, alpha=0.5)
    plt.xlabel('Overall Feature Mean')
    plt.ylabel('CV')
    plt.title('Feature Mean vs Variability')
    plt.axhline(0.5, color='red', linestyle='--', alpha=0.5)
    
    # 8. Cumulative CV distribution
    plt.subplot(2, 4, 8)
    sorted_cvs = np.sort(cv_values)
    cumulative = np.arange(1, len(sorted_cvs) + 1) / len(sorted_cvs)
    plt.plot(sorted_cvs, cumulative)
    plt.xlabel('CV Threshold')
    plt.ylabel('Fraction of Features Below Threshold')
    plt.title('Cumulative CV Distribution')
    plt.axvline(0.1, color='orange', linestyle='--', label='0.1')
    plt.axvline(0.5, color='red', linestyle='--', label='0.5')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('detailed_recording_artifacts.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Main detailed analysis"""
    print("Starting detailed recording artifacts analysis...")
    
    # Load data
    data_dir = '../data/t15_copyTask_neuralData/hdf5_data_final'
    
    # Get sessions
    sessions = []
    for item in os.listdir(data_dir):
        if item.startswith('t15.') and os.path.isdir(os.path.join(data_dir, item)):
            sessions.append(item)
    
    sessions.sort()
    
    # Load detailed data
    session_data = load_detailed_session_data(sessions, data_dir)
    
    if len(session_data) < 2:
        print("Not enough sessions!")
        return
    
    # Analyze feature distributions
    distribution_results = analyze_feature_distributions(session_data)
    
    # Analyze scaling patterns
    scaling_results = analyze_signal_scaling_patterns(session_data)
    
    # Analyze electrode groups
    electrode_results = analyze_electrode_group_patterns(session_data)
    
    # Create visualizations
    create_detailed_visualizations(distribution_results, scaling_results, electrode_results)
    
    # Final conclusions
    print(f"\n{'='*60}")
    print("DETAILED ANALYSIS CONCLUSIONS")
    print(f"{'='*60}")
    
    high_var_features = np.sum(distribution_results['feature_mean_cv'] > 0.5)
    moderate_var_features = np.sum(distribution_results['feature_mean_cv'] > 0.1)
    
    print(f"Features with high variability (CV > 0.5): {high_var_features} / 512 ({high_var_features/512*100:.1f}%)")
    print(f"Features with moderate variability (CV > 0.1): {moderate_var_features} / 512 ({moderate_var_features/512*100:.1f}%)")
    
    # Check if scaling explains the differences
    if scaling_results:
        avg_correlation = np.mean([scaling_results[s]['correlation'] for s in scaling_results.keys()])
        avg_explained_var = np.mean([scaling_results[s]['explained_variance'] for s in scaling_results.keys()])
        
        print(f"\nAverage correlation between sessions: {avg_correlation:.4f}")
        print(f"Average explained variance by linear scaling: {avg_explained_var:.4f}")
        
        if avg_correlation > 0.9:
            print("✅ Sessions are highly correlated - differences are mainly scaling/offset")
        elif avg_correlation > 0.7:
            print("⚠️  Sessions are moderately correlated - some structural differences")
        else:
            print("🚨 Sessions have low correlation - major structural differences")
    
    print("\nAnalysis complete! Check 'detailed_recording_artifacts.png' for visualizations.")

if __name__ == "__main__":
    main()
