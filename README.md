## Overview

This framework processes 715 YAML files containing AWSIM autonomous driving simulation data to:

1. **Extract 2D bounding boxes** and velocity data from 3D vehicle simulations
2. **Calculate distances** between ego vehicle, NPCs, and perceived objects
3. **Detect crashes**
4. **Evaluate safety constraints** to determine their capability
5. **Compare** original RSS vs. enhanced safety constraint performance

### Required Python Packages
- `numpy` - Numerical computations
- `pandas` - Data manipulation and analysis
- `shapely` - Geometric operations for polygon distances
- `pyyaml` - YAML file parsing
- `tqdm` - Progress bars
- `matplotlib` & `seaborn` - Visualization (optional)


## Usage


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
RSS_REACTION_TIME          # seconds
RSS_MAX_ACCELERATION       # m/s²
RSS_EGO_BRAKING           # m/s²
RSS_NPC_BRAKING           # m/s²

# Enhanced Safety Constraint
NEW_REACTION_TIME          # seconds
NEW_MAX_ACCELERATION       # m/s²
MAX_DISTANCE_NOISE         # meters
MAX_VELOCITY_ERROR         # m/s
MAX_PERCEPTION_LAG         # seconds
```

### Data Validation Parameters
```python
MIN_PERCEPTION_TIMESTAMPS   # Minimum valid perception data points
MAX_TIME_DIFF_THRESHOLD    # Max time gap between timestamps (seconds)
MAX_DISTANCE_CHANGE_THRESHOLD  # Max distance change between frames (meters)
```

### Crash Detection Parameters
```python
CRASH_DISTANCE_THRESHOLD   # Distance threshold for crash detection (meters)
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
- Keeps values for velocity
- 
### Step 3: Safety Analysis
- **Crash Detection**: frame in the CSV with a crash
- **RSS Constraint Evaluation**: Original rss formula
- **Enhanced Constraint Evaluation**: Improved formula
- **Comparative Analysis**: RSS vs Improved vs Crash Analysis


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
