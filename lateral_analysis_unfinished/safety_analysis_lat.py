import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from parameters import *

class SafetyAnalyzerLat:
    def __init__(self):
        self.results = []

    def calculate_original_lateral_constraint(self, ego_v_lat, perception_d, npc_v_lat, ego_break_lat, npc_break_lat):
        """
        Calculate original lateral RSS constraint:
        d_perception ≥ mew + (u·ρ) + (u² / 2·β1) - (v_lat·(σ + ρ) + v_lat² / 2·β2)
        """
        if np.isnan(ego_v_lat) or np.isnan(perception_d):
            return np.nan, np.nan

        v_npc = 0 if np.isnan(npc_v_lat) else npc_v_lat

        rho = RSS_REACTION_TIME
        sigma = MAX_PERCEPTION_LAG
        mew = RSS_CLEARANCE_BUFFER_LAT
        beta1 = ego_break_lat
        beta2 = npc_break_lat

        term1 = mew + ego_v_lat * rho + (ego_v_lat**2) / (2 * beta1)
        term2 = v_npc * (sigma + rho) + (v_npc**2) / (2 * beta2)

        required = term1 - term2
        satisfied = perception_d >= required
        
        return required, satisfied

    def calculate_new_lateral_constraint(self, ego_v_lat, perception_d, npc_v_lat, ego_break_lat, npc_break_lat, timestamp_lag=0):
        """
        Calculate new enhanced safety constraint:
        (p_t - ε_max) ≥ mew + u·ρ + (u² / 2·β1) - (v_lat * (σ + ρ) + v_lat² / 2·β2) + Δd
        """
        if np.isnan(ego_v_lat) or np.isnan(perception_d):
            return np.nan, np.nan

        v_npc = 0 if np.isnan(npc_v_lat) else npc_v_lat

        rho = NEW_REACTION_TIME
        sigma = MAX_PERCEPTION_LAG
        mew = RSS_CLEARANCE_BUFFER_LAT
        beta1 = ego_break_lat
        beta2 = npc_break_lat

        epsilon = MAX_DISTANCE_NOISE

        v_eff = max(0, v_npc - MAX_VELOCITY_ERROR - beta2 * sigma)
        delta_d = timestamp_lag * ego_v_lat if timestamp_lag > 0 else 0

        term1 = mew + ego_v_lat * rho + (ego_v_lat**2) / (2 * beta1)
        term2 = v_eff * (sigma + rho) + (v_eff**2) / (2 * beta2)

        required = term1 - term2 + delta_d
        adjusted = perception_d - epsilon
        satisfied = adjusted >= required

        return required, satisfied

    def detect_crashes(self, df):
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
                    crash_timestamp = df.iloc[crash_start_idx]['timestamp']
                    crashes.append({
                        'crash_start_idx': crash_start_idx,
                        'crash_timestamp': crash_timestamp,
                        'crash_distance': ego_npc_distance,
                        'detection_frame': idx
                    })
                    consecutive_count = 0
                    crash_start_idx = None
            else:
                consecutive_count = 0
                crash_start_idx = None

        return crashes

    def analyze_safety_violations(self, df, crashes):
        violations = {
            'original_lateral': [],
            'new_lateral': []
        }

        for idx, row in df.iterrows():
            timestamp = row['timestamp']

            v_ego_lat = row.get('ego_velocity_lateral', np.nan)
            v_npc_lat = row.get('npc1_velocity_latral', np.nan) 

            ego_velocity = v_ego_lat if v_ego_lat > 0 else np.nan  # when positive
            ego_breaking = -v_ego_lat if v_ego_lat < 0 else np.nan  # when negative, convert to positive magnitude
            npc_breaking = -v_npc_lat if v_npc_lat < 0 else np.nan  # when negative, convert to positive magnitude

            perception_distance = row.get('ego_perception_distance', np.nan)
            perception_velocity = row.get('closest_perception_velocity_lateral', np.nan)

            rss_required, rss_satisfied = self.calculate_original_lateral_constraint(
                ego_velocity, perception_distance, perception_velocity, ego_breaking, npc_breaking
            )

            if not np.isnan(rss_required) and not rss_satisfied:
                violations['original_lateral'].append({
                    'timestamp': timestamp,
                    'frame_idx': idx,
                    'required_distance': rss_required,
                    'actual_distance': perception_distance,
                    'violation_magnitude': rss_required - perception_distance
                })

            new_required, new_satisfied = self.calculate_new_lateral_constraint(
                ego_velocity, perception_distance, perception_velocity, ego_breaking, npc_breaking
            )

            if not np.isnan(new_required) and not new_satisfied:
                violations['new_lateral'].append({
                    'timestamp': timestamp,
                    'frame_idx': idx,
                    'required_distance': new_required,
                    'actual_distance': perception_distance,
                    'violation_magnitude': new_required - perception_distance
                })

        return violations

    def analyze_predictive_capability(self, crashes, violations, df):
        predictive_analysis = {
            'original_lateral': {
                'predicted_crashes': 0,
                'total_crashes': len(crashes),
                'early_warnings': [],
                'concurrent_detections': 0
            },
            'new_lateral': {
                'predicted_crashes': 0,
                'total_crashes': len(crashes),
                'early_warnings': [],
                'concurrent_detections': 0
            }
        }

        for crash in crashes:
            crash_timestamp = crash['crash_timestamp']

            for key in ['original_lateral', 'new_lateral']:
                early_warning = False
                for violation in violations[key]:
                    if violation['timestamp'] < crash_timestamp:
                        warning_time = crash_timestamp - violation['timestamp']
                        predictive_analysis[key]['early_warnings'].append({
                            'crash_timestamp': crash_timestamp,
                            'violation_timestamp': violation['timestamp'],
                            'warning_time': warning_time
                        })
                        early_warning = True
                        break
                    elif abs(violation['timestamp'] - crash_timestamp) < 0.1:
                        predictive_analysis[key]['concurrent_detections'] += 1

                if early_warning:
                    predictive_analysis[key]['predicted_crashes'] += 1

        return predictive_analysis

    def process_csv_file(self, csv_path):
        try:
            df = pd.read_csv(csv_path)

            if len(df) == 0:
                return None

            crashes = self.detect_crashes(df)
            violations = self.analyze_safety_violations(df, crashes)
            predictive_analysis = self.analyze_predictive_capability(crashes, violations, df)

            total_frames = len(df)
            frames_with_perception = len(df[~df['ego_perception_distance'].isna()])

            result = {
                'file_name': os.path.basename(csv_path),
                'total_frames': total_frames,
                'frames_with_perception': frames_with_perception,
                'crash_count': len(crashes),
                'crashed': len(crashes) > 0,
                'first_crash_time': crashes[0]['crash_timestamp'] if crashes else None,
                'rss_violations': len(violations['original_lateral']),
                'new_violations': len(violations['new_lateral']),
                'rss_predicted_crashes': predictive_analysis['original_lateral']['predicted_crashes'],
                'new_predicted_crashes': predictive_analysis['new_lateral']['predicted_crashes'],
                'rss_concurrent_detections': predictive_analysis['original_lateral']['concurrent_detections'],
                'new_concurrent_detections': predictive_analysis['new_lateral']['concurrent_detections'],
                'rss_early_warnings': len(predictive_analysis['original_lateral']['early_warnings']),
                'new_early_warnings': len(predictive_analysis['new_lateral']['early_warnings']),
                'crashes': crashes,
                'violations': violations,
                'predictive_analysis': predictive_analysis
            }

            return result

        except Exception as e:
            print(f"Error processing {csv_path}: {str(e)}")
            return None

    def analyze_all_files(self):
        if not os.path.exists(CSV_OUTPUT_DIR):
            print(f"CSV directory not found: {CSV_OUTPUT_DIR}")
            return

        csv_files = [f for f in os.listdir(CSV_OUTPUT_DIR) if f.endswith('_distances.csv')]
        csv_paths = [os.path.join(CSV_OUTPUT_DIR, f) for f in csv_files]

        print(f"Found {len(csv_paths)} CSV files to analyze")

        self.results = []
        for csv_path in tqdm(csv_paths, desc="Analyzing lateral safety constraints"):
            result = self.process_csv_file(csv_path)
            if result is not None:
                self.results.append(result)

        print(f"Successfully analyzed {len(self.results)} files")

    def generate_summary_report(self):
        if not self.results:
            print("No results to analyze")
            return

        os.makedirs(RESULTS_OUTPUT_DIR, exist_ok=True)

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
        summary_path = os.path.join(RESULTS_OUTPUT_DIR, 'lateral_safety_summary.csv')
        summary_df.to_csv(summary_path, index=False)

        total_experiments = len(summary_df)
        total_crashes = summary_df['crash_count'].sum()
        crashed_experiments = summary_df['crashed'].sum()
        crash_rate = crashed_experiments / total_experiments * 100 if total_experiments > 0 else 0

        rss_total_violations = summary_df['rss_violations'].sum()
        rss_predicted_crashes = summary_df['rss_predicted_crashes'].sum()
        rss_recall = rss_predicted_crashes / crashed_experiments * 100 if crashed_experiments > 0 else 0

        new_total_violations = summary_df['new_violations'].sum()
        new_predicted_crashes = summary_df['new_predicted_crashes'].sum()
        new_recall = new_predicted_crashes / crashed_experiments * 100 if crashed_experiments > 0 else 0

        report = f"""
# Autonomous Vehicle Safety Analysis Report (Lateral Cases)

## Dataset Overview
- Total Experiments: {total_experiments}
- Total Crashes Detected: {total_crashes}
- Experiments with Crashes: {crashed_experiments}
- Overall Crash Rate: {crash_rate:.1f}%

## Original Lateral RSS Constraint Performance
- Total Violations Detected: {rss_total_violations}
- Crashes Predicted (Early Warning): {rss_predicted_crashes}
- Predictive Recall: {rss_recall:.1f}%
- Early Warning Rate: {rss_predicted_crashes}/{crashed_experiments} crashes

## New Lateral RSS Safety Constraint Performance
- Total Violations Detected: {new_total_violations}
- Crashes Predicted (Early Warning): {new_predicted_crashes}
- Predictive Recall: {new_recall:.1f}%
- Early Warning Rate: {new_predicted_crashes}/{crashed_experiments} crashes

## Comparison
- Violations: Original = {rss_total_violations}, New = {new_total_violations}
- Predictive Power: Original = {rss_recall:.1f}%, New = {new_recall:.1f}%
"""
        if rss_recall == 0 and new_recall == 0:
            report += "- CRITICAL: Neither constraint predicted any crashes.\n"
        elif new_recall > rss_recall:
            report += f"- New constraint outperformed original by {new_recall - rss_recall:.1f}%\n"
        elif rss_recall > new_recall:
            report += f"- Original constraint outperformed new by {rss_recall - new_recall:.1f}%\n"
        else:
            report += "- Both constraints performed equally in predictive recall.\n"

        report_path = os.path.join(RESULTS_OUTPUT_DIR, 'lateral_safety_report.md')
        with open(report_path, 'w') as f:
            f.write(report)

        print(report)
        print(f"\nReport saved to: {report_path}")
        return summary_df

def main():
    analyzer = SafetyAnalyzerLat()
    analyzer.analyze_all_files()
    summary_df = analyzer.generate_summary_report()

    return analyzer, summary_df

if __name__ == "__main__":
    analyzer, summary = main()
