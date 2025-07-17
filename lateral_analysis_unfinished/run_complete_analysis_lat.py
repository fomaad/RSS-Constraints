#!/usr/bin/env python3
"""
Complete Autonomous Vehicle Safety Analysis Pipeline

This script runs the complete pipeline:
1. Convert YAML files to JSON with bounding box data
2. Convert JSON to CSV with distance calculations
3. Perform crash detection and safety constraint analysis
4. Generate comparative analysis report

Usage: python run_complete_analysis.py [--step STEP_NUMBER]
"""

import argparse
import sys
import os
from datetime import datetime

# Import our modules
from yaml_to_json_optimized import main as yaml_to_json_main
from json_to_csv_improved import main as json_to_csv_main
from safety_analysis_lat import main as safety_analysis_main
from parameters import *

def print_banner():
    """Print analysis banner"""
    print("=" * 80)
    print("    AUTONOMOUS VEHICLE SAFETY EVALUATION FRAMEWORK")
    print("    AWSIM Simulation Data Analysis Pipeline")
    print("=" * 80)
    print()

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = [
        'yaml', 'numpy', 'pandas', 'shapely', 'tqdm', 'matplotlib', 'seaborn'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"ERROR: Missing required packages: {', '.join(missing_packages)}")
        print("Please install them using: pip install " + ' '.join(missing_packages))
        return False
    
    return True

def check_input_data():
    """Check if input data exists"""
    if not os.path.exists(INPUT_YAML_DIR):
        print(f"ERROR: Input YAML directory not found: {INPUT_YAML_DIR}")
        return False
    
    yaml_files = [f for f in os.listdir(INPUT_YAML_DIR) if f.endswith('.yaml')]
    if not yaml_files:
        print(f"ERROR: No YAML files found in {INPUT_YAML_DIR}")
        return False
    
    print(f"Found {len(yaml_files)} YAML files to process")
    return True

def step1_yaml_to_json():
    """Step 1: Convert YAML files to JSON"""
    print("\n" + "="*50)
    print("STEP 1: Converting YAML files to JSON")
    print("="*50)
    
    try:
        yaml_to_json_main()
        return True
    except Exception as e:
        print(f"ERROR in Step 1: {str(e)}")
        return False

def step2_json_to_csv():
    """Step 2: Convert JSON files to CSV with distance calculations"""
    print("\n" + "="*50)
    print("STEP 2: Converting JSON to CSV with distance calculations")
    print("="*50)
    
    if not os.path.exists(JSON_OUTPUT_DIR):
        print(f"ERROR: JSON directory not found: {JSON_OUTPUT_DIR}")
        print("Please run Step 1 first")
        return False
    
    try:
        json_to_csv_main()
        return True
    except Exception as e:
        print(f"ERROR in Step 2: {str(e)}")
        return False

def step3_safety_analysis():
    """Step 3: Perform crash detection and safety constraint analysis"""
    print("\n" + "="*50)
    print("STEP 3: Crash Detection and Lateral Safety Constraint Analysis")
    print("="*50)
    
    if not os.path.exists(CSV_OUTPUT_DIR):
        print(f"ERROR: CSV directory not found: {CSV_OUTPUT_DIR}")
        print("Please run Steps 1 and 2 first")
        return False
    
    try:
        analyzer, summary = safety_analysis_main()
        return True
    except Exception as e:
        print(f"ERROR in Step 3: {str(e)}")
        return False

def print_final_summary():
    """Print final summary of the analysis"""
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    
    # Check what was generated
    directories_to_check = [
        (JSON_OUTPUT_DIR, "JSON files"),
        (CSV_OUTPUT_DIR, "CSV files"),
        (RESULTS_OUTPUT_DIR, "Analysis results")
    ]
    
    for directory, description in directories_to_check:
        if os.path.exists(directory):
            file_count = len(os.listdir(directory))
            print(f"✓ {description}: {file_count} files in {directory}")
        else:
            print(f"✗ {description}: Directory not found")
    
    print("\nKey Output Files:")
    
    # Check for key files
    key_files = [
        (os.path.join(CSV_OUTPUT_DIR, "processing_summary.csv"), "Processing Summary"),
        (os.path.join(RESULTS_OUTPUT_DIR, "lateral_safety_summary.csv"), "Lateral Safety Summary"),
        (os.path.join(RESULTS_OUTPUT_DIR, "lateral_safety_report.md"), "Final Lateral Analysis Report")
    ]
    
    for file_path, description in key_files:
        if os.path.exists(file_path):
            print(f"✓ {description}: {file_path}")
        else:
            print(f"✗ {description}: Not found")
    
    print(f"\nAnalysis completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nTo view the final report, check:")
    print(f"  {os.path.join(RESULTS_OUTPUT_DIR, 'lateral_safety_report.md')}")

def run_complete_pipeline():
    """Run the complete analysis pipeline"""
    print_banner()
    
    # Check dependencies and input data
    if not check_dependencies():
        return False
    
    if not check_input_data():
        return False
    
    print(f"Starting complete analysis pipeline at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Run all steps
    steps = [
        ("Step 1: YAML → JSON", step1_yaml_to_json),
        ("Step 2: JSON → CSV", step2_json_to_csv),
        ("Step 3: Safety Analysis", step3_safety_analysis)
    ]
    
    for step_name, step_function in steps:
        print(f"\nExecuting {step_name}...")
        if not step_function():
            print(f"FAILED: {step_name}")
            return False
        print(f"COMPLETED: {step_name}")
    
    print_final_summary()
    return True

def run_single_step(step_number):
    """Run a single step of the pipeline"""
    print_banner()
    
    if not check_dependencies():
        return False
    
    steps = {
        1: ("Step 1: YAML → JSON", step1_yaml_to_json),
        2: ("Step 2: JSON → CSV", step2_json_to_csv),
        3: ("Step 3: Safety Analysis", step3_safety_analysis)
    }
    
    if step_number not in steps:
        print(f"ERROR: Invalid step number {step_number}. Valid steps: 1, 2, 3")
        return False
    
    step_name, step_function = steps[step_number]
    print(f"Executing {step_name}...")
    
    if step_function():
        print(f"COMPLETED: {step_name}")
        return True
    else:
        print(f"FAILED: {step_name}")
        return False

def main():
    """Main function with command line argument parsing"""
    parser = argparse.ArgumentParser(
        description="Autonomous Vehicle Safety Analysis Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_complete_analysis.py              # Run complete pipeline
  python run_complete_analysis.py --step 1     # Run only YAML to JSON conversion
  python run_complete_analysis.py --step 2     # Run only JSON to CSV conversion
  python run_complete_analysis.py --step 3     # Run only safety analysis

Steps:
  1: Convert YAML simulation files to JSON with bounding box data
  2: Convert JSON to CSV with distance calculations and validation
  3: Perform crash detection and safety constraint analysis
        """
    )
    
    parser.add_argument(
        '--step', 
        type=int, 
        choices=[1, 2, 3],
        help='Run only a specific step (1, 2, or 3)'
    )
    
    args = parser.parse_args()
    
    try:
        if args.step:
            success = run_single_step(args.step)
        else:
            success = run_complete_pipeline()
        
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 