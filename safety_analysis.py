import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from parameters import *

class SafetyAnalyzer:
    def __init__(self):
        self.results = []
        
    def calculate_original_rss_constraint(self, ego_velocity, perception_distance, perception_velocity):
        """
        Calculate original RSS constraint:
        d_perception ≥ u·ρ + (1/2)·α_max·ρ² + (u + ρ·α_max)²/(2·β_min) - v_perception²/(2·β_max)
        """
        if np.isnan(ego_velocity) or np.isnan(perception_distance):
            return np.nan, np.nan
        
        # Handle NaN perception velocity
        v_perception = 0 if np.isnan(perception_velocity) else perception_velocity
        
        u = ego_velocity
        rho = RSS_REACTION_TIME
        alpha_max = RSS_MAX_ACCELERATION
        beta_min = RSS_EGO_BRAKING
        beta_max = RSS_NPC_BRAKING
        
        # Calculate required safe distance
        term1 = u * rho
        term2 = 0.5 * alpha_max * (rho ** 2)
        term3 = ((u + rho * alpha_max) ** 2) / (2 * beta_min)
        term4 = (v_perception ** 2) / (2 * beta_max)
        
        required_distance = term1 + term2 + term3 - term4
        
        # Check if constraint is satisfied
        constraint_satisfied = perception_distance >= required_distance
        
        return required_distance, constraint_satisfied
    
    def calculate_new_safety_constraint(self, ego_velocity, perception_distance, perception_velocity, timestamp_lag=0):
        """
        Calculate new enhanced safety constraint:
        (p_t - ε_max) ≥ u_t·ρ + (1/2)·α_max·ρ² + (u_t + ρ·α_max)²/(2·β_min) - v_eff²/(2·β_max) + Δd
        """
        if np.isnan(ego_velocity) or np.isnan(perception_distance):
            return np.nan, np.nan
        
        # Handle NaN perception velocity
        v_perception = 0 if np.isnan(perception_velocity) else perception_velocity
        
        u_t = ego_velocity
        p_t = perception_distance
        rho = NEW_REACTION_TIME
        alpha_max = NEW_MAX_ACCELERATION
        beta_min = NEW_EGO_BRAKING
        beta_max = NEW_NPC_BRAKING
        epsilon_max = MAX_DISTANCE_NOISE
        gamma_max = MAX_VELOCITY_ERROR
        delta = MAX_PERCEPTION_LAG
        
        # Calculate effective velocity with uncertainty
        v_eff = max(0, v_perception - gamma_max - beta_max * delta)
        
        # Calculate lag compensation term (simplified)
        delta_d = timestamp_lag * u_t if timestamp_lag > 0 else 0
        
        # Calculate required safe distance
        term1 = u_t * rho
        term2 = 0.5 * alpha_max * (rho ** 2)
        term3 = ((u_t + rho * alpha_max) ** 2) / (2 * beta_min)
        term4 = (v_eff ** 2) / (2 * beta_max)
        
        required_distance = term1 + term2 + term3 - term4 + delta_d
        
        # Apply noise consideration
        adjusted_perception_distance = p_t - epsilon_max
        
        # Check if constraint is satisfied
        constraint_satisfied = adjusted_perception_distance >= required_distance
        
        return required_distance, constraint_satisfied
    
    def detect_crashes(self, df):
        """Detect crashes based on consecutive frames with distance < threshold"""
        crashes = []
        consecutive_count = 0
        crash_start_idx = None
        
        for idx, row in df.iterrows():
            ego_npc_distance = row['ego_npc_distance']
            
            if not np.isnan(ego_npc_distance) and ego_npc_distance < CRASH_DISTANCE_THRESHOLD:
                if consecutive_count == 0:
                    crash_start_idx = idx
                consecutive_count += 1
                
                if consecutive_count >= CRASH_CONSECUTIVE_FRAMES:
                    # Found a crash
                    crash_timestamp = df.iloc[crash_start_idx]['timestamp']
                    crashes.append({
                        'crash_start_idx': crash_start_idx,
                        'crash_timestamp': crash_timestamp,
                        'crash_distance': ego_npc_distance,
                        'detection_frame': idx
                    })
                    # Reset to avoid duplicate detections
                    consecutive_count = 0
                    crash_start_idx = None
            else:
                consecutive_count = 0
                crash_start_idx = None
        
        return crashes
    
    def analyze_safety_violations(self, df, crashes):
        """Analyze when safety constraints are violated relative to crashes"""
        violations = {
            'original_rss': [],
            'new_constraint': []
        }
        
        for idx, row in df.iterrows():
            timestamp = row['timestamp']
            ego_velocity = row['ego_velocity_magnitude']
            perception_distance = row['ego_perception_distance']
            perception_velocity = row['closest_perception_velocity']
            
            # Calculate original RSS constraint
            rss_required, rss_satisfied = self.calculate_original_rss_constraint(
                ego_velocity, perception_distance, perception_velocity
            )
            
            if not np.isnan(rss_required) and not rss_satisfied:
                violations['original_rss'].append({
                    'timestamp': timestamp,
                    'frame_idx': idx,
                    'required_distance': rss_required,
                    'actual_distance': perception_distance,
                    'violation_magnitude': rss_required - perception_distance
                })
            
            # Calculate new safety constraint
            new_required, new_satisfied = self.calculate_new_safety_constraint(
                ego_velocity, perception_distance, perception_velocity
            )
            
            if not np.isnan(new_required) and not new_satisfied:
                violations['new_constraint'].append({
                    'timestamp': timestamp,
                    'frame_idx': idx,
                    'required_distance': new_required,
                    'actual_distance': perception_distance,
                    'violation_magnitude': new_required - perception_distance
                })
        
        return violations
    
    def analyze_predictive_capability(self, crashes, violations, df):
        """Analyze if safety violations occur before crashes (predictive capability)"""
        predictive_analysis = {
            'original_rss': {
                'predicted_crashes': 0,
                'total_crashes': len(crashes),
                'early_warnings': [],
                'concurrent_detections': 0
            },
            'new_constraint': {
                'predicted_crashes': 0,
                'total_crashes': len(crashes),
                'early_warnings': [],
                'concurrent_detections': 0
            }
        }
        
        for crash in crashes:
            crash_timestamp = crash['crash_timestamp']
            
            # Check RSS violations before crash
            rss_early_warning = False
            for violation in violations['original_rss']:
                if violation['timestamp'] < crash_timestamp:
                    # Found violation before crash
                    warning_time = crash_timestamp - violation['timestamp']
                    predictive_analysis['original_rss']['early_warnings'].append({
                        'crash_timestamp': crash_timestamp,
                        'violation_timestamp': violation['timestamp'],
                        'warning_time': warning_time
                    })
                    rss_early_warning = True
                    break
                elif abs(violation['timestamp'] - crash_timestamp) < 0.1:  # Concurrent
                    predictive_analysis['original_rss']['concurrent_detections'] += 1
            
            if rss_early_warning:
                predictive_analysis['original_rss']['predicted_crashes'] += 1
            
            # Check new constraint violations before crash
            new_early_warning = False
            for violation in violations['new_constraint']:
                if violation['timestamp'] < crash_timestamp:
                    # Found violation before crash
                    warning_time = crash_timestamp - violation['timestamp']
                    predictive_analysis['new_constraint']['early_warnings'].append({
                        'crash_timestamp': crash_timestamp,
                        'violation_timestamp': violation['timestamp'],
                        'warning_time': warning_time
                    })
                    new_early_warning = True
                    break
                elif abs(violation['timestamp'] - crash_timestamp) < 0.1:  # Concurrent
                    predictive_analysis['new_constraint']['concurrent_detections'] += 1
            
            if new_early_warning:
                predictive_analysis['new_constraint']['predicted_crashes'] += 1
        
        return predictive_analysis
    
    def process_csv_file(self, csv_path):
        """Process a single CSV file for safety analysis"""
        try:
            df = pd.read_csv(csv_path)
            
            # Basic validation
            if len(df) == 0:
                return None
            
            # Detect crashes
            crashes = self.detect_crashes(df)
            
            # Analyze safety violations
            violations = self.analyze_safety_violations(df, crashes)
            
            # Analyze predictive capability
            predictive_analysis = self.analyze_predictive_capability(crashes, violations, df)
            
            # Calculate statistics
            total_frames = len(df)
            frames_with_perception = len(df[~df['ego_perception_distance'].isna()])
            
            result = {
                'file_name': os.path.basename(csv_path),
                'total_frames': total_frames,
                'frames_with_perception': frames_with_perception,
                'crash_count': len(crashes),
                'crashed': len(crashes) > 0,
                'first_crash_time': crashes[0]['crash_timestamp'] if crashes else None,
                'rss_violations': len(violations['original_rss']),
                'new_violations': len(violations['new_constraint']),
                'rss_predicted_crashes': predictive_analysis['original_rss']['predicted_crashes'],
                'new_predicted_crashes': predictive_analysis['new_constraint']['predicted_crashes'],
                'rss_concurrent_detections': predictive_analysis['original_rss']['concurrent_detections'],
                'new_concurrent_detections': predictive_analysis['new_constraint']['concurrent_detections'],
                'rss_early_warnings': len(predictive_analysis['original_rss']['early_warnings']),
                'new_early_warnings': len(predictive_analysis['new_constraint']['early_warnings']),
                'crashes': crashes,
                'violations': violations,
                'predictive_analysis': predictive_analysis
            }
            
            return result
            
        except Exception as e:
            print(f"Error processing {csv_path}: {str(e)}")
            return None
    
    def analyze_all_files(self):
        """Analyze all CSV files in the CSV_OUTPUT_DIR"""
        if not os.path.exists(CSV_OUTPUT_DIR):
            print(f"CSV directory not found: {CSV_OUTPUT_DIR}")
            return
        
        csv_files = [f for f in os.listdir(CSV_OUTPUT_DIR) if f.endswith('_distances.csv')]
        csv_paths = [os.path.join(CSV_OUTPUT_DIR, f) for f in csv_files]
        
        print(f"Found {len(csv_paths)} CSV files to analyze")
        
        self.results = []
        for csv_path in tqdm(csv_paths, desc="Analyzing safety constraints"):
            result = self.process_csv_file(csv_path)
            if result is not None:
                self.results.append(result)
        
        print(f"Successfully analyzed {len(self.results)} files")
    
    def generate_summary_report(self):
        """Generate comprehensive summary report"""
        if not self.results:
            print("No results to analyze")
            return
        
        # Create results directory
        os.makedirs(RESULTS_OUTPUT_DIR, exist_ok=True)
        
        # Convert results to DataFrame for analysis
        summary_data = []
        for result in self.results:
            summary_data.append({
                'file_name': result['file_name'],
                'total_frames': result['total_frames'],
                'frames_with_perception': result['frames_with_perception'],
                'crashed': result['crashed'],
                'crash_count': result['crash_count'],
                'first_crash_time': result['first_crash_time'],
                'rss_violations': result['rss_violations'],
                'new_violations': result['new_violations'],
                'rss_predicted_crashes': result['rss_predicted_crashes'],
                'new_predicted_crashes': result['new_predicted_crashes'],
                'rss_early_warnings': result['rss_early_warnings'],
                'new_early_warnings': result['new_early_warnings']
            })
        
        summary_df = pd.DataFrame(summary_data)
        
        # Save detailed results
        summary_df.to_csv(os.path.join(RESULTS_OUTPUT_DIR, 'safety_analysis_summary.csv'), index=False)
        
        # Calculate overall statistics
        total_experiments = len(summary_df)
        total_crashes = summary_df['crash_count'].sum()
        crashed_experiments = summary_df['crashed'].sum()
        crash_rate = crashed_experiments / total_experiments * 100
        
        # RSS performance
        rss_total_violations = summary_df['rss_violations'].sum()
        rss_predicted_crashes = summary_df['rss_predicted_crashes'].sum()
        rss_recall = rss_predicted_crashes / crashed_experiments * 100 if crashed_experiments > 0 else 0
        
        # New constraint performance
        new_total_violations = summary_df['new_violations'].sum()
        new_predicted_crashes = summary_df['new_predicted_crashes'].sum()
        new_recall = new_predicted_crashes / crashed_experiments * 100 if crashed_experiments > 0 else 0
        
        # Generate report
        report = f"""
# Autonomous Vehicle Safety Analysis Report

## Dataset Overview
- Total Experiments: {total_experiments}
- Total Crashes Detected: {total_crashes}
- Experiments with Crashes: {crashed_experiments}
- Overall Crash Rate: {crash_rate:.1f}%

## Original RSS Constraint Performance
- Total Violations Detected: {rss_total_violations}
- Crashes Predicted (Early Warning): {rss_predicted_crashes}
- Predictive Recall: {rss_recall:.1f}%
- Early Warning Rate: {rss_predicted_crashes}/{crashed_experiments} crashes

## New Enhanced Safety Constraint Performance
- Total Violations Detected: {new_total_violations}
- Crashes Predicted (Early Warning): {new_predicted_crashes}
- Predictive Recall: {new_recall:.1f}%
- Early Warning Rate: {new_predicted_crashes}/{crashed_experiments} crashes

## Comparative Analysis
- RSS vs New Constraint Violations: {rss_total_violations} vs {new_total_violations}
- RSS vs New Predictive Performance: {rss_recall:.1f}% vs {new_recall:.1f}%

## Key Findings
"""
        
        if rss_recall == 0 and new_recall == 0:
            report += "- CRITICAL: Both safety constraints failed to provide early warning for any crashes\n"
            report += "- Safety violations appear to be reactive (concurrent with crashes) rather than predictive\n"
        elif new_recall > rss_recall:
            report += f"- New constraint shows {new_recall - rss_recall:.1f}% better predictive performance\n"
        elif rss_recall > new_recall:
            report += f"- Original RSS shows {rss_recall - new_recall:.1f}% better predictive performance\n"
        else:
            report += "- Both constraints show similar predictive performance\n"
        
        # Save report
        with open(os.path.join(RESULTS_OUTPUT_DIR, 'safety_analysis_report.md'), 'w') as f:
            f.write(report)
        
        print(report)
        print(f"\nDetailed results saved to: {RESULTS_OUTPUT_DIR}")
        
        return summary_df

def main():
    """Main function to run complete safety analysis"""
    analyzer = SafetyAnalyzer()
    analyzer.analyze_all_files()
    summary_df = analyzer.generate_summary_report()
    
    return analyzer, summary_df

if __name__ == "__main__":
    analyzer, summary = main() 