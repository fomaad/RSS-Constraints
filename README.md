# Autonomous Vehicle Safety Evaluation Framework

A comprehensive framework for analyzing autonomous vehicle safety using AWSIM simulation data. This system evaluates the predictive capability of safety constraints (original RSS and enhanced formulations) in crash scenarios.

## Overview

This framework processes 714+ YAML files containing AWSIM autonomous driving simulation data to:

1. **Extract 2D bounding boxes** and velocity data from 3D vehicle simulations
2. **Calculate distances** between ego vehicle, NPCs, and perceived objects
3. **Detect crashes** using consecutive proximity measurements
4. **Evaluate safety constraints** to determine their predictive capability
5. **Compare** original RSS vs. enhanced safety constraint performance

## Key Research Question

**Can safety constraints predict crashes before they occur, or are they merely reactive?**

Based on previous analysis, both safety constraints failed to provide early warning (0% predictive recall), only detecting violations concurrent with crashes.

## Installation

### Prerequisites

```bash
pip install numpy pandas shapely pyyaml tqdm matplotlib seaborn
```

### Required Python Packages
- `numpy` - Numerical computations
- `pandas` - Data manipulation and analysis
- `shapely` - Geometric operations for polygon distances
- `pyyaml` - YAML file parsing
- `tqdm` - Progress bars
- `matplotlib` & `seaborn` - Visualization (optional)

## Project Structure

```
CREST2/
├── allYaml/                    # Input: 714+ YAML simulation files
├── JSON_Data/                  # Output: Converted JSON files with bounding boxes
├── CSV_Data/                   # Output: Distance calculations and metrics
├── Results/                    # Output: Final analysis results
├── parameters.py               # Configuration parameters
├── yaml_to_json_improved.py    # Step 1: YAML → JSON conversion
├── json_to_csv_improved.py     # Step 2: JSON → CSV with distances
├── safety_analysis.py          # Step 3: Crash detection & constraint analysis
├── run_complete_analysis.py    # Master script to run entire pipeline
└── README.md                   # This file
```

## Usage

### Quick Start (Complete Pipeline)

```bash
python run_complete_analysis.py
```

This runs all three steps automatically:
1. YAML → JSON conversion
2. JSON → CSV distance calculation
3. Safety constraint analysis and comparison

### Step-by-Step Execution

```bash
# Step 1: Convert YAML files to JSON with bounding box data
python run_complete_analysis.py --step 1

# Step 2: Calculate distances and create CSV files
python run_complete_analysis.py --step 2

# Step 3: Analyze crashes and safety constraints
python run_complete_analysis.py --step 3
```

### Individual Script Execution

```bash
# Alternative: Run scripts individually
python yaml_to_json_improved.py
python json_to_csv_improved.py
python safety_analysis.py
```

## Configuration Parameters

Edit `parameters.py` to adjust:

### Safety Constraint Parameters
```python
# Original RSS Constraint
RSS_REACTION_TIME = 1.0          # seconds
RSS_MAX_ACCELERATION = 3.0       # m/s²
RSS_EGO_BRAKING = 4.0           # m/s²
RSS_NPC_BRAKING = 8.0           # m/s²

# Enhanced Safety Constraint
NEW_REACTION_TIME = 0.5          # seconds
NEW_MAX_ACCELERATION = 5.0       # m/s²
MAX_DISTANCE_NOISE = 2.0         # meters
MAX_VELOCITY_ERROR = 1.5         # m/s
MAX_PERCEPTION_LAG = 0.4         # seconds
```

### Data Validation Parameters
```python
MIN_PERCEPTION_TIMESTAMPS = 10   # Minimum valid perception data points
MAX_TIME_DIFF_THRESHOLD = 0.2    # Max time gap between timestamps (seconds)
MAX_DISTANCE_CHANGE_THRESHOLD = 10.0  # Max distance change between frames (meters)
```

### Crash Detection Parameters
```python
CRASH_DISTANCE_THRESHOLD = 0.5   # Distance threshold for crash detection (meters)
CRASH_CONSECUTIVE_FRAMES = 3     # Required consecutive frames for crash confirmation
```

## Data Processing Pipeline

### Step 1: YAML → JSON Conversion
- Extracts 2D bounding boxes from 3D vehicle data (X-Z plane projection)
- Calculates velocity vectors from twist data or position differences
- Handles ego vehicle, NPC, and perception object data
- Outputs structured JSON with frame-by-frame data

### Step 2: JSON → CSV Distance Calculation
- Creates Shapely polygons from bounding box corners
- Calculates ego-NPC distance (ground truth)
- Calculates ego-perception distance (closest perceived object)
- Validates data consistency and filters invalid experiments
- **Only processes experiments with 10+ valid perception timestamps**

### Step 3: Safety Analysis
- **Crash Detection**: 3+ consecutive frames with distance < 0.5m
- **RSS Constraint Evaluation**: Original responsibility-sensitive safety formula
- **Enhanced Constraint Evaluation**: Improved formulation with noise consideration
- **Predictive Analysis**: Determines if violations occur before crashes
- **Comparative Analysis**: RSS vs. Enhanced constraint performance

## Safety Constraint Formulations

### Original RSS Constraint
```
d_perception ≥ u·ρ + (1/2)·α_max·ρ² + (u + ρ·α_max)²/(2·β_min) - v_perception²/(2·β_max)
```

Where:
- `d_perception`: Perceived distance to object
- `u`: Ego vehicle velocity
- `v_perception`: Perceived object velocity
- `ρ`: Reaction time (1.0s)
- `α_max`: Maximum acceleration (3.0 m/s²)
- `β_min`: Ego braking capability (4.0 m/s²)
- `β_max`: Object braking capability (8.0 m/s²)

### Enhanced Safety Constraint
```
(p_t - ε_max) ≥ u_t·ρ + (1/2)·α_max·ρ² + (u_t + ρ·α_max)²/(2·β_min) - v_eff²/(2·β_max) + Δd
```

Where:
- `p_t`: Perceived distance with noise consideration
- `ε_max`: Maximum distance perception noise (2.0m)
- `v_eff`: Effective velocity accounting for uncertainty
- `Δd`: Lag compensation term
- Updated parameters for more conservative safety margins

## Output Files

### Key Results
- `Results/safety_analysis_summary.csv` - Detailed per-experiment results
- `Results/safety_analysis_report.md` - Comprehensive analysis report
- `CSV_Data/processing_summary.csv` - Data processing statistics

### Analysis Metrics
- **Crash Rate**: Percentage of experiments resulting in crashes
- **Predictive Recall**: Percentage of crashes predicted before occurrence
- **Violation Detection**: Total safety constraint violations detected
- **Early Warning Capability**: Time between violation and crash

## Expected Findings

Based on previous analysis:

- **Crash Rate**: ~15.9% (114/715 experiments)
- **Critical Pattern**: `deceleration=0` scenarios have 100% crash rate
- **RSS Predictive Recall**: 0% (reactive, not predictive)
- **Enhanced Constraint Recall**: 0% (also reactive)
- **Key Insight**: Both constraints detect violations concurrent with crashes, not before

## Data Quality Validation

The framework includes robust validation:

1. **Minimum Data Requirements**: Experiments need 10+ valid perception timestamps
2. **Temporal Consistency**: Consecutive timestamps must be within reasonable time gaps
3. **Distance Consistency**: Perception distances cannot change dramatically between frames
4. **Geometric Validation**: Bounding boxes must form valid polygons

## Troubleshooting

### Common Issues

1. **Missing Dependencies**: Install required packages using pip
2. **No YAML Files**: Ensure `allYaml/` directory contains .yaml files
3. **Insufficient Perception Data**: Many experiments may be filtered out due to sparse perception data
4. **Memory Issues**: Large datasets may require processing in batches

### Debug Mode
Add print statements or modify `tqdm` parameters for more detailed progress tracking.

## Research Applications

This framework enables:

- **Safety Standard Evaluation**: Testing effectiveness of current AV safety constraints
- **Constraint Development**: Developing new predictive safety formulations  
- **Simulation Validation**: Validating AWSIM simulation fidelity
- **Risk Assessment**: Quantifying crash prediction capabilities

## Future Enhancements

- **Multi-NPC Analysis**: Extend beyond single NPC scenarios
- **3D Distance Calculations**: Include Y-axis for full 3D analysis
- **Machine Learning Integration**: Develop ML-based predictive models
- **Real-time Processing**: Optimize for real-time safety monitoring

## Contributing

When modifying the framework:

1. Update `parameters.py` for new configuration options
2. Maintain data validation consistency across all steps
3. Document new safety constraint formulations
4. Update this README with significant changes

## License

This framework is designed for autonomous vehicle safety research and evaluation. 