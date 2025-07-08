import yaml
import os
import numpy as np
import json
from tqdm import tqdm
import gc
from parameters import *

class NpEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle NaN values and numpy types"""
    def default(self, obj):
        if isinstance(obj, np.floating):
            if np.isnan(obj) or np.isinf(obj):
                return None
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

def load_yaml_data(file_path):
    """Load YAML data with error handling"""
    try:
        with open(file_path, 'r') as file:
            return yaml.safe_load(file)
    except Exception as e:
        print(f"Error loading {file_path}: {str(e)}")
        return None

def get_velocity_from_twist(twist):
    """Extract velocity from twist data, preserving actual zeros"""
    try:
        vx = twist['linear']['x']
        vy = twist['linear']['y'] 
        vz = twist['linear']['z']
        return np.array([vx, vy, vz])
    except (KeyError, TypeError):
        return np.array([np.nan, np.nan, np.nan])

def calculate_velocity_vector(pos_current, pos_previous, time_current, time_previous):
    """Calculate velocity from position changes"""
    if time_previous == time_current:
        return np.array([np.nan, np.nan, np.nan])
    
    dt = time_current - time_previous
    dx = pos_current['x'] - pos_previous['x']
    dy = pos_current['y'] - pos_previous['y'] 
    dz = pos_current['z'] - pos_previous['z']
    
    return np.array([dx/dt, dy/dt, dz/dt])

def create_2d_oriented_box(position, rotation, extents):
    """Create 2D bounding box from 3D vehicle data"""
    half_length = extents['x']
    half_width = extents['z']
    
    # Convert rotation to radians
    yaw_rad = np.radians(-rotation['y'])
    cos_yaw = np.cos(yaw_rad)
    sin_yaw = np.sin(yaw_rad)
    
    # Define corners in local coordinate system
    corners_local = np.array([
        [half_length, half_width],
        [half_length, -half_width], 
        [-half_length, -half_width],
        [-half_length, half_width]
    ])
    
    # Apply rotation
    rotation_matrix = np.array([
        [cos_yaw, -sin_yaw],
        [sin_yaw, cos_yaw]
    ])
    
    corners_rotated = np.dot(corners_local, rotation_matrix.T)
    corners_world = corners_rotated + np.array([position['x'], position['z']])
    
    return corners_world

def extract_data_from_yaml_optimized(yaml_data, output_path):
    """Extract bounding box and velocity data from YAML - optimized version"""
    try:
        total_frames = len(yaml_data['states'])
        frames_needed = min(MAX_FRAMES, total_frames)
        step = max(1, total_frames // frames_needed) if frames_needed < total_frames else 1
        frame_indices = list(range(0, total_frames, step))
        
        previous_state = None
        previous_timestamp = 0
        
        bbox_data = {
            "metadata": {
                "total_frames": total_frames,
                "processed_frames": len(frame_indices),
                "source_file": os.path.basename(output_path).replace('_data.json', '.yaml')
            },
            "frames": []
        }
        
        for frame_idx in frame_indices:
            state = yaml_data['states'][frame_idx]
            current_timestamp = state["timeStamp"]
            
            frame_data = {
                "timestamp": current_timestamp,
                "frame_index": frame_idx,
                "ego": {"box": None, "velocity": None, "position": None},
                "npc1": {"box": None, "velocity": None, "position": None},
                "perception_objects": []
            }
            
            # ===== EGO VEHICLE DATA =====
            ego_data = state['groundtruth_ego']
            ego_pos = ego_data['pose']['position']
            ego_rot = ego_data['pose']['rotation']
            
            frame_data["ego"]["position"] = {
                "x": float(ego_pos['x']),
                "y": float(ego_pos['y']),
                "z": float(ego_pos['z'])
            }
            
            # Get ego velocity
            ego_velocity = np.array([np.nan, np.nan, np.nan])
            velocity_source = "none"
            
            if 'twist' in ego_data:
                ego_velocity = get_velocity_from_twist(ego_data['twist'])
                velocity_source = "twist"
            elif previous_state is not None:
                prev_ego_pos = previous_state['groundtruth_ego']['pose']['position']
                ego_velocity = calculate_velocity_vector(
                    ego_pos, prev_ego_pos, current_timestamp, previous_timestamp)
                velocity_source = "calculated"
            
            magnitude = np.linalg.norm(ego_velocity) if not np.isnan(ego_velocity).any() else np.nan
            
            frame_data["ego"]["velocity"] = {
                "x": ego_velocity[0],
                "y": ego_velocity[1], 
                "z": ego_velocity[2],
                "magnitude": magnitude,
                "source": velocity_source,
            }
            
            # Get ego bounding box
            ego_extent = None
            if 'ego_detail' in yaml_data:
                ego_extent = yaml_data['ego_detail']['extents']
            
            if ego_extent is not None:
                ego_box_2d = create_2d_oriented_box(ego_pos, ego_rot, ego_extent)
                frame_data["ego"]["box"] = ego_box_2d.tolist()
            
            # ===== NPC DATA =====
            if 'groundtruth_NPCs' in state and state['groundtruth_NPCs']:
                npc = state['groundtruth_NPCs'][0]  # Focus on first NPC
                npc_pos = npc['pose']['position']
                npc_rot = npc['pose']['rotation']
                
                frame_data["npc1"]["position"] = {
                    "x": float(npc_pos['x']),
                    "y": float(npc_pos['y']),
                    "z": float(npc_pos['z'])
                }
                
                # Get NPC velocity
                npc_velocity = np.array([np.nan, np.nan, np.nan])
                velocity_source = "none"
                
                if 'twist' in npc:
                    npc_velocity = get_velocity_from_twist(npc['twist'])
                    velocity_source = "twist"
                elif previous_state is not None and 'groundtruth_NPCs' in previous_state:
                    if len(previous_state['groundtruth_NPCs']) > 0:
                        prev_npc_pos = previous_state['groundtruth_NPCs'][0]['pose']['position']
                        npc_velocity = calculate_velocity_vector(
                            npc_pos, prev_npc_pos, current_timestamp, previous_timestamp)
                        velocity_source = "calculated"
                
                magnitude = np.linalg.norm(npc_velocity) if not np.isnan(npc_velocity).any() else np.nan
                
                frame_data["npc1"]["velocity"] = {
                    "x": npc_velocity[0],
                    "y": npc_velocity[1],
                    "z": npc_velocity[2], 
                    "magnitude": magnitude,
                    "source": velocity_source,
                }
                
                # Get NPC bounding box
                npc_extent = None
                if 'npcs_detail' in yaml_data and len(yaml_data['npcs_detail']) > 0:
                    npc_extent = yaml_data['npcs_detail'][0]['extents']
                
                if npc_extent is not None:
                    npc_box_2d = create_2d_oriented_box(npc_pos, npc_rot, npc_extent)
                    frame_data["npc1"]["box"] = npc_box_2d.tolist()
            
            # ===== PERCEPTION OBJECTS =====
            perception_objects = []
            if 'perception_objects' in state and state['perception_objects'] is not None:
                perception_objects = state['perception_objects']
            elif 'perception' in state and 'detected_objects' in state['perception'] and state['perception']['detected_objects'] is not None:
                perception_objects = state['perception']['detected_objects']
            
            if perception_objects:  # Only iterate if we have valid perception objects
                for perc_obj in perception_objects:
                    try:
                        perc_pos = perc_obj['pose']['position']
                        perc_rot = perc_obj['pose']['rotation']
                        
                        # Get perception velocity
                        perc_velocity = np.array([np.nan, np.nan, np.nan])
                        velocity_source = "none"
                        
                        if 'twist' in perc_obj:
                            perc_velocity = get_velocity_from_twist(perc_obj['twist'])
                            velocity_source = "twist"
                        elif 'velocity' in perc_obj:
                            vel = perc_obj['velocity']
                            if isinstance(vel, dict) and all(k in vel for k in ['x', 'y', 'z']):
                                perc_velocity = np.array([vel['x'], vel['y'], vel['z']])
                                velocity_source = "direct"
                        
                        # Get perception bounding box
                        perc_extent = None
                        if 'shape' in perc_obj and perc_obj['shape'] is not None and 'size' in perc_obj['shape']:
                            perc_extent = {
                                'x': perc_obj['shape']['size']['x'] / 2,
                                'y': perc_obj['shape']['size']['y'] / 2,
                                'z': perc_obj['shape']['size']['z'] / 2
                            }
                        elif 'dimensions' in perc_obj:
                            dims = perc_obj['dimensions']
                            if 'width' in dims and 'length' in dims and 'height' in dims:
                                perc_extent = {
                                    'x': dims['length'] / 2,
                                    'y': dims['height'] / 2,
                                    'z': dims['width'] / 2
                                }
                        
                        if perc_extent is None:
                            continue
                        
                        perc_box_2d = create_2d_oriented_box(perc_pos, perc_rot, perc_extent)
                        magnitude = np.linalg.norm(perc_velocity) if not np.isnan(perc_velocity).any() else np.nan
                        
                        obj_data = {
                            "box": perc_box_2d.tolist(),
                            "velocity": {
                                "x": perc_velocity[0],
                                "y": perc_velocity[1],
                                "z": perc_velocity[2],
                                "magnitude": magnitude,
                                "source": velocity_source
                            },
                            "position": {
                                "x": float(perc_pos['x']),
                                "y": float(perc_pos['y']),
                                "z": float(perc_pos['z'])
                            }
                        }
                        frame_data["perception_objects"].append(obj_data)
                    except Exception as e:
                        # Skip problematic perception objects
                        continue
            
            bbox_data["frames"].append(frame_data)
            previous_state = state
            previous_timestamp = current_timestamp
        
        # Save JSON data
        with open(output_path, 'w') as f:
            json.dump(bbox_data, f, indent=2, cls=NpEncoder)
        
        return output_path
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error processing data for {output_path}: {str(e)}")
        return None

def process_yaml_file_optimized(yaml_path):
    """Process a single YAML file to JSON - optimized version"""
    try:
        basename = os.path.basename(yaml_path)
        output_path = os.path.join(JSON_OUTPUT_DIR, f"{os.path.splitext(basename)[0]}_data.json")
        
        # Skip if file already exists
        if os.path.exists(output_path):
            return {
                'yaml_path': yaml_path,
                'json_path': output_path,
                'success': True,
                'skipped': True
            }
        
        yaml_data = load_yaml_data(yaml_path)
        if yaml_data is None:
            return None
        
        json_path = extract_data_from_yaml_optimized(yaml_data, output_path)
        
        # Clear memory
        del yaml_data
        gc.collect()
        
        return {
            'yaml_path': yaml_path,
            'json_path': json_path,
            'success': json_path is not None,
            'skipped': False
        }
        
    except Exception as e:
        print(f"Error processing {yaml_path}: {str(e)}")
        return None

def process_batch(yaml_paths_batch, batch_num, total_batches):
    """Process a batch of YAML files"""
    print(f"\n--- Processing Batch {batch_num}/{total_batches} ({len(yaml_paths_batch)} files) ---")
    
    successful_conversions = 0
    failed_conversions = 0
    skipped_conversions = 0
    
    for yaml_path in tqdm(yaml_paths_batch, desc=f"Batch {batch_num}", leave=False):
        result = process_yaml_file_optimized(yaml_path)
        if result is not None:
            if result.get('skipped', False):
                skipped_conversions += 1
            elif result['success']:
                successful_conversions += 1
            else:
                failed_conversions += 1
        else:
            failed_conversions += 1
    
    print(f"Batch {batch_num} complete: {successful_conversions} success, {skipped_conversions} skipped, {failed_conversions} failed")
    
    # Force garbage collection after batch
    gc.collect()
    
    return successful_conversions, skipped_conversions, failed_conversions

def main():
    """Main function to process all YAML files in batches"""
    if not os.path.exists(INPUT_YAML_DIR):
        print(f"Input directory not found: {INPUT_YAML_DIR}")
        return
    
    # Create output directory
    os.makedirs(JSON_OUTPUT_DIR, exist_ok=True)
    
    # Find all YAML files
    yaml_files = [f for f in os.listdir(INPUT_YAML_DIR) if f.endswith('.yaml')]
    yaml_paths = [os.path.join(INPUT_YAML_DIR, f) for f in yaml_files]
    
    print(f"Found {len(yaml_paths)} YAML files total")
    
    # Check existing files
    existing_json = set()
    if os.path.exists(JSON_OUTPUT_DIR):
        existing_json = set(os.listdir(JSON_OUTPUT_DIR))
    
    # Filter files that need processing
    files_to_process = []
    for yaml_path in yaml_paths:
        basename = os.path.basename(yaml_path)
        expected_json = f"{os.path.splitext(basename)[0]}_data.json"
        if expected_json not in existing_json:
            files_to_process.append(yaml_path)
    
    print(f"Found {len(existing_json)} existing JSON files")
    print(f"Need to process {len(files_to_process)} YAML files")
    
    if not files_to_process:
        print("All files already processed!")
        return
    
    # Process in batches of 40
    BATCH_SIZE = 40
    total_successful = 0
    total_skipped = 0
    total_failed = 0
    
    total_batches = (len(files_to_process) + BATCH_SIZE - 1) // BATCH_SIZE
    
    for i in range(0, len(files_to_process), BATCH_SIZE):
        batch = files_to_process[i:i + BATCH_SIZE]
        batch_num = (i // BATCH_SIZE) + 1
        
        success, skipped, failed = process_batch(batch, batch_num, total_batches)
        total_successful += success
        total_skipped += skipped
        total_failed += failed
        
        print(f"Overall progress: {total_successful + total_skipped + total_failed}/{len(files_to_process)} files processed")
    
    print(f"\n=== FINAL RESULTS ===")
    print(f"Total files processed: {len(files_to_process)}")
    print(f"Successful conversions: {total_successful}")
    print(f"Skipped (already existed): {total_skipped}")
    print(f"Failed conversions: {total_failed}")
    print(f"JSON files saved to: {JSON_OUTPUT_DIR}")

if __name__ == "__main__":
    main() 