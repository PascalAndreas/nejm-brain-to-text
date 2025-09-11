#!/usr/bin/env python3
"""
Analysis of recording artifacts across days to understand why day-specific
transformations are so critical for the RNN model.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import h5py
from scipy import stats
# import seaborn as sns  # Not needed for this analysis
from tqdm import tqdm

def load_neural_data_sample(sessions, data_dir, max_trials_per_session=10):
    """Load a sample of neural data from multiple sessions"""
    print("Loading neural data samples from multiple sessions...")
    
    session_data = {}
    
    for session in tqdm(sessions[:20], desc="Loading sessions"):  # Limit to first 20 for speed
        session_path = os.path.join(data_dir, session)
        
        # Try to load from any available split (train, val, test)
        for split in ['train', 'val', 'test']:
            file_path = os.path.join(session_path, f'data_{split}.hdf5')
            
            if os.path.exists(file_path):
                try:
                    with h5py.File(file_path, 'r') as f:
                        trials = list(f.keys())[:max_trials_per_session]
                        
                        neural_features = []
                        for trial_key in trials:
                            trial_data = f[trial_key]['input_features'][:]
                            neural_features.append(trial_data)
                        
                        if neural_features:
                            session_data[session] = {
                                'neural_features': neural_features,
                                'split': split
                            }
                            print(f"  {session}: {len(neural_features)} trials from {split}")
                            break  # Use first available split
                            
                except Exception as e:
                    print(f"  Error loading {session}: {e}")
                    continue
    
    print(f"Successfully loaded data from {len(session_data)} sessions")
    return session_data

def analyze_amplitude_statistics(session_data):
    """Analyze amplitude statistics across sessions"""
    print("\n=== AMPLITUDE ANALYSIS ===")
    
    session_stats = {}
    
    for session, data in session_data.items():
        # Concatenate all trials for this session
        all_features = np.concatenate(data['neural_features'], axis=0)  # [time, features]
        
        # Compute statistics
        stats_dict = {
            'mean': np.mean(all_features, axis=0),  # [features]
            'std': np.std(all_features, axis=0),    # [features]
            'median': np.median(all_features, axis=0),  # [features]
            'rms': np.sqrt(np.mean(all_features**2, axis=0)),  # [features]
            'percentile_95': np.percentile(all_features, 95, axis=0),
            'percentile_5': np.percentile(all_features, 5, axis=0),
            'range': np.ptp(all_features, axis=0),  # peak-to-peak
            'session': session,
            'n_timepoints': all_features.shape[0]
        }
        
        session_stats[session] = stats_dict
    
    return session_stats

def analyze_cross_session_variability(session_stats):
    """Analyze how much neural features vary across sessions"""
    print("\n=== CROSS-SESSION VARIABILITY ANALYSIS ===")
    
    sessions = list(session_stats.keys())
    n_features = len(session_stats[sessions[0]]['mean'])
    
    # Collect statistics across sessions
    session_means = np.array([session_stats[s]['mean'] for s in sessions])  # [sessions, features]
    session_stds = np.array([session_stats[s]['std'] for s in sessions])
    session_rms = np.array([session_stats[s]['rms'] for s in sessions])
    
    # Analyze variability
    results = {
        'mean_across_sessions': np.mean(session_means, axis=0),  # [features]
        'std_of_means': np.std(session_means, axis=0),  # How much session means vary
        'mean_of_stds': np.mean(session_stds, axis=0),  # Average within-session variability
        'coefficient_of_variation': np.std(session_means, axis=0) / (np.mean(session_means, axis=0) + 1e-8),
        'session_means': session_means,
        'session_stds': session_stds,
        'session_rms': session_rms,
        'sessions': sessions
    }
    
    # Print summary statistics
    print(f"Number of features: {n_features}")
    print(f"Number of sessions: {len(sessions)}")
    
    # Overall amplitude changes
    overall_session_amplitude = np.mean(session_rms, axis=1)  # [sessions]
    print(f"\nOverall RMS amplitude across sessions:")
    print(f"  Mean: {np.mean(overall_session_amplitude):.4f}")
    print(f"  Std: {np.std(overall_session_amplitude):.4f}")
    print(f"  Range: {np.min(overall_session_amplitude):.4f} - {np.max(overall_session_amplitude):.4f}")
    print(f"  Coefficient of variation: {np.std(overall_session_amplitude) / np.mean(overall_session_amplitude):.4f}")
    
    # Feature-wise variability
    high_variability_features = np.where(results['coefficient_of_variation'] > 0.5)[0]
    print(f"\nFeatures with high cross-session variability (CV > 0.5): {len(high_variability_features)} / {n_features}")
    
    return results

def analyze_temporal_drift(session_data):
    """Analyze if there's systematic drift over time"""
    print("\n=== TEMPORAL DRIFT ANALYSIS ===")
    
    # Sort sessions by date
    sessions_with_dates = []
    for session in session_data.keys():
        try:
            # Extract date from session name (e.g., t15.2023.08.13)
            parts = session.split('.')
            if len(parts) >= 4:
                year, month, day = int(parts[1]), int(parts[2]), int(parts[3])
                date = pd.Timestamp(year=year, month=month, day=day)
                sessions_with_dates.append((date, session))
        except:
            continue
    
    sessions_with_dates.sort()  # Sort by date
    sorted_sessions = [s[1] for s in sessions_with_dates]
    dates = [s[0] for s in sessions_with_dates]
    
    print(f"Analyzing temporal drift across {len(sorted_sessions)} sessions")
    print(f"Date range: {dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}")
    
    # Compute overall amplitude for each session
    session_amplitudes = []
    for session in sorted_sessions:
        all_features = np.concatenate(session_data[session]['neural_features'], axis=0)
        overall_rms = np.sqrt(np.mean(all_features**2))
        session_amplitudes.append(overall_rms)
    
    # Analyze trend
    days_since_start = [(d - dates[0]).days for d in dates]
    correlation, p_value = stats.pearsonr(days_since_start, session_amplitudes)
    
    print(f"Correlation between time and amplitude: r={correlation:.4f}, p={p_value:.4f}")
    
    return {
        'dates': dates,
        'sessions': sorted_sessions,
        'amplitudes': session_amplitudes,
        'days_since_start': days_since_start,
        'correlation': correlation,
        'p_value': p_value
    }

def create_visualizations(session_stats, variability_results, drift_results):
    """Create visualizations of the recording artifacts"""
    
    # 1. Session amplitude comparison
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 3, 1)
    sessions = variability_results['sessions']
    overall_amplitudes = np.mean(variability_results['session_rms'], axis=1)
    plt.bar(range(len(sessions)), overall_amplitudes)
    plt.xlabel('Session Index')
    plt.ylabel('Overall RMS Amplitude')
    plt.title('RMS Amplitude Across Sessions')
    plt.xticks(range(0, len(sessions), max(1, len(sessions)//10)), rotation=45)
    
    # 2. Coefficient of variation distribution
    plt.subplot(2, 3, 2)
    cv_values = variability_results['coefficient_of_variation']
    plt.hist(cv_values, bins=50, alpha=0.7, edgecolor='black')
    plt.xlabel('Coefficient of Variation')
    plt.ylabel('Number of Features')
    plt.title('Distribution of Cross-Session Variability')
    plt.axvline(0.5, color='red', linestyle='--', label='High variability threshold')
    plt.legend()
    
    # 3. Feature-wise variability heatmap
    plt.subplot(2, 3, 3)
    # Show first 64 features for visualization
    session_means_subset = variability_results['session_means'][:, :64]
    im = plt.imshow(session_means_subset.T, aspect='auto', cmap='viridis')
    plt.colorbar(im)
    plt.xlabel('Session Index')
    plt.ylabel('Feature Index (first 64)')
    plt.title('Feature Means Across Sessions')
    
    # 4. Temporal drift
    plt.subplot(2, 3, 4)
    if drift_results:
        plt.scatter(drift_results['days_since_start'], drift_results['amplitudes'], alpha=0.7)
        # Fit trend line
        z = np.polyfit(drift_results['days_since_start'], drift_results['amplitudes'], 1)
        p = np.poly1d(z)
        plt.plot(drift_results['days_since_start'], p(drift_results['days_since_start']), 
                "r--", alpha=0.8, label=f'r={drift_results["correlation"]:.3f}')
        plt.xlabel('Days Since Start')
        plt.ylabel('Overall RMS Amplitude')
        plt.title('Temporal Drift in Signal Amplitude')
        plt.legend()
    
    # 5. Session-to-session amplitude ratios
    plt.subplot(2, 3, 5)
    amplitude_ratios = []
    for i in range(len(overall_amplitudes)-1):
        ratio = overall_amplitudes[i+1] / overall_amplitudes[i]
        amplitude_ratios.append(ratio)
    
    plt.plot(amplitude_ratios, 'o-', alpha=0.7)
    plt.axhline(1.0, color='red', linestyle='--', label='No change')
    plt.xlabel('Session Transition')
    plt.ylabel('Amplitude Ratio (Next/Current)')
    plt.title('Session-to-Session Amplitude Changes')
    plt.legend()
    
    # 6. Distribution of feature means across sessions
    plt.subplot(2, 3, 6)
    feature_mean_ranges = []
    for feature_idx in range(min(512, variability_results['session_means'].shape[1])):
        feature_means = variability_results['session_means'][:, feature_idx]
        range_val = np.max(feature_means) - np.min(feature_means)
        feature_mean_ranges.append(range_val)
    
    plt.hist(feature_mean_ranges, bins=50, alpha=0.7, edgecolor='black')
    plt.xlabel('Range of Feature Means Across Sessions')
    plt.ylabel('Number of Features')
    plt.title('Distribution of Feature Mean Ranges')
    
    plt.tight_layout()
    plt.savefig('recording_artifacts_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def analyze_electrode_groups(session_stats):
    """Analyze if different electrode groups show different patterns"""
    print("\n=== ELECTRODE GROUP ANALYSIS ===")
    
    # Based on the README, features are organized as:
    # 0-64: ventral 6v threshold crossings
    # 65-128: area 4 threshold crossings  
    # 129-192: 55b threshold crossings
    # 193-256: dorsal 6v threshold crossings
    # 257-320: ventral 6v spike band power
    # 321-384: area 4 spike band power
    # 385-448: 55b spike band power
    # 449-512: dorsal 6v spike band power
    
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
    
    sessions = list(session_stats.keys())
    group_variability = {}
    
    for group_name, (start, end) in electrode_groups.items():
        # Get means for this group across sessions
        group_means = []
        for session in sessions:
            group_mean = np.mean(session_stats[session]['mean'][start:end])
            group_means.append(group_mean)
        
        group_means = np.array(group_means)
        cv = np.std(group_means) / (np.abs(np.mean(group_means)) + 1e-8)
        
        group_variability[group_name] = {
            'means': group_means,
            'cv': cv,
            'range': np.max(group_means) - np.min(group_means)
        }
        
        print(f"{group_name}: CV={cv:.4f}, Range={group_variability[group_name]['range']:.4f}")
    
    return group_variability

def main():
    """Main analysis function"""
    print("Starting recording artifacts analysis...")
    
    # Load data
    data_dir = '../data/t15_copyTask_neuralData/hdf5_data_final'
    
    # Get list of sessions
    sessions = []
    for item in os.listdir(data_dir):
        if item.startswith('t15.') and os.path.isdir(os.path.join(data_dir, item)):
            sessions.append(item)
    
    sessions.sort()
    print(f"Found {len(sessions)} sessions")
    
    # Load sample data
    session_data = load_neural_data_sample(sessions, data_dir)
    
    if len(session_data) < 2:
        print("Not enough sessions loaded for analysis!")
        return
    
    # Analyze amplitude statistics
    session_stats = analyze_amplitude_statistics(session_data)
    
    # Analyze cross-session variability
    variability_results = analyze_cross_session_variability(session_stats)
    
    # Analyze temporal drift
    drift_results = analyze_temporal_drift(session_data)
    
    # Analyze electrode groups
    electrode_group_results = analyze_electrode_groups(session_stats)
    
    # Create visualizations
    create_visualizations(session_stats, variability_results, drift_results)
    
    # Summary and conclusions
    print(f"\n{'='*60}")
    print("RECORDING ARTIFACTS ANALYSIS SUMMARY")
    print(f"{'='*60}")
    
    overall_cv = np.std(np.mean(variability_results['session_rms'], axis=1)) / np.mean(np.mean(variability_results['session_rms'], axis=1))
    
    print(f"Overall amplitude coefficient of variation: {overall_cv:.4f}")
    
    if overall_cv > 0.3:
        print("🚨 HIGH AMPLITUDE VARIABILITY DETECTED!")
        print("   Significant recording artifacts present across sessions.")
        print("   This explains why day-specific transformations are so critical.")
    elif overall_cv > 0.15:
        print("⚠️  MODERATE AMPLITUDE VARIABILITY DETECTED")
        print("   Some recording artifacts present.")
    else:
        print("✅ LOW AMPLITUDE VARIABILITY")
        print("   Recording artifacts are minimal.")
    
    # Check for systematic drift
    if drift_results and abs(drift_results['correlation']) > 0.3 and drift_results['p_value'] < 0.05:
        print(f"\n📈 SYSTEMATIC TEMPORAL DRIFT DETECTED!")
        print(f"   Correlation with time: r={drift_results['correlation']:.4f}, p={drift_results['p_value']:.4f}")
        print("   Signal amplitude changes systematically over time.")
    
    # Most variable electrode groups
    most_variable_group = max(electrode_group_results.keys(), 
                             key=lambda x: electrode_group_results[x]['cv'])
    least_variable_group = min(electrode_group_results.keys(), 
                              key=lambda x: electrode_group_results[x]['cv'])
    
    print(f"\nMost variable electrode group: {most_variable_group} (CV={electrode_group_results[most_variable_group]['cv']:.4f})")
    print(f"Least variable electrode group: {least_variable_group} (CV={electrode_group_results[least_variable_group]['cv']:.4f})")
    
    print(f"\nAnalysis complete! Check 'recording_artifacts_analysis.png' for visualizations.")
    
    return {
        'session_stats': session_stats,
        'variability_results': variability_results,
        'drift_results': drift_results,
        'electrode_group_results': electrode_group_results
    }

if __name__ == "__main__":
    results = main()
