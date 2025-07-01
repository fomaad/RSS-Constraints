# Parameters for Autonomous Vehicle Safety Evaluation Framework

# =============================================
# ORIGINAL RSS CONSTRAINT PARAMETERS
# =============================================
RSS_REACTION_TIME = 1.0  # seconds (ρ)
RSS_MAX_ACCELERATION = 3.0  # m/s² (α_max)
RSS_EGO_BRAKING = 4.0  # m/s² (β_min)
RSS_NPC_BRAKING = 8.0  # m/s² (β_max)

# =============================================
# NEW ENHANCED SAFETY CONSTRAINT PARAMETERS
# =============================================
NEW_REACTION_TIME = 0.5  # seconds (ρ)
NEW_MAX_ACCELERATION = 5.0  # m/s² (α_max)
NEW_EGO_BRAKING = 5.0  # m/s² (β_min)
NEW_NPC_BRAKING = 5.0  # m/s² (β_max)
MAX_DISTANCE_NOISE = 2.0  # meters (ε_max)
MAX_VELOCITY_ERROR = 1.5  # m/s (γ_max)
MAX_PERCEPTION_LAG = 0.4  # seconds (δ)

# =============================================
# CRASH DETECTION PARAMETERS
# =============================================
CRASH_DISTANCE_THRESHOLD = 0.5  # meters
CRASH_CONSECUTIVE_FRAMES = 3  # number of consecutive frames

# =============================================
# DATA VALIDATION PARAMETERS
# =============================================
MIN_PERCEPTION_TIMESTAMPS = 10  # minimum timestamps with perception data
MAX_TIME_DIFF_THRESHOLD = 0.2  # seconds - max time difference between consecutive timestamps
MAX_DISTANCE_CHANGE_THRESHOLD = 10.0  # meters - max distance change between consecutive timestamps
MAX_VELOCITY_CHANGE_THRESHOLD = 20.0  # m/s - max velocity change between consecutive timestamps

# =============================================
# FILE PROCESSING PARAMETERS
# =============================================
INPUT_YAML_DIR = "allYaml/"
JSON_OUTPUT_DIR = "JSON_Data/"
CSV_OUTPUT_DIR = "CSV_Data/"
RESULTS_OUTPUT_DIR = "Results/"

# Frame selection for processing (to match video output)
MAX_FRAMES = 60 * 20  # 20 seconds at 60 fps 