import json
import os
import pandas as pd
import numpy as np
from shapely.geometry import Polygon
from shapely.errors import ShapelyError
from tqdm import tqdm
from parameters import *

def load_json_data(file_path):
    """Load JSON data with error handling"""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {file_path}: {str(e)}")
        return None

def create_polygon_from_box(box_corners):
    """Create Shapely polygon from bounding box corners"""
    if box_corners is None or len(box_corners) != 4:
        return None
    
    try:
        # Ensure we have valid coordinates
        coords = [(float(corner[0]), float(corner[1])) for corner in box_corners]
        polygon = Polygon(coords)
        
        # Check if polygon is valid
        if not polygon.is_valid:
            # Try to fix invalid polygon
            polygon = polygon.buffer(0)
            if not polygon.is_valid:
                return None
        
        return polygon
    except (ShapelyError, ValueError, TypeError):
        return None

def calculate_polygon_distance(poly1, poly2):
    """Calculate distance between two polygons"""
    if poly1 is None or poly2 is None:
        return np.nan
    
    try:
        return poly1.distance(poly2)
    except ShapelyError:
        return np.nan

def find_closest_perception_object(ego_polygon, perception_objects):
    """Find closest perception object to ego vehicle"""
    if ego_polygon is None or not perception_objects:
        return np.nan, None
    
    min_distance = float('inf')
    closest_obj = None
    
    for perc_obj in perception_objects:
        if perc_obj.get('box') is None:
            continue
            
        perc_polygon = create_polygon_from_box(perc_obj['box'])
        if perc_polygon is None:
            continue
            
        distance = calculate_polygon_distance(ego_polygon, perc_polygon)
        if not np.isnan(distance) and distance < min_distance:
            min_distance = distance
            closest_obj = perc_obj
    
    return min_distance if min_distance != float('inf') else np.nan, closest_obj

def validate_perception_data(frames):
    """Validate perception data according to specified criteria"""
    # Count frames with valid perception data
    valid_perception_count = 0
    perception_distances = []
    timestamps = []
    
    for frame in frames:
        if frame.get('perception_objects') and len(frame['perception_objects']) > 0:
            # Check if at least one perception object has valid data
            has_valid_perception = False
            for obj in frame['perception_objects']:
                if obj.get('box') is not None:
                    has_valid_perception = True
                    break
            
            if has_valid_perception:
                valid_perception_count += 1
                # We'll calculate actual distance later, just mark as valid for now
                perception_distances.append(frame['timestamp'])
                timestamps.append(frame['timestamp'])
    
    # Check minimum perception timestamps requirement
    if valid_perception_count < MIN_PERCEPTION_TIMESTAMPS:
        return False, f"Only {valid_perception_count} valid perception timestamps (minimum: {MIN_PERCEPTION_TIMESTAMPS})"
    
    # Check time differences between consecutive timestamps
    if len(timestamps) > 1:
        for i in range(1, len(timestamps)):
            time_diff = timestamps[i] - timestamps[i-1]
            if time_diff > MAX_TIME_DIFF_THRESHOLD:
                return False, f"Time difference too large: {time_diff:.3f}s > {MAX_TIME_DIFF_THRESHOLD}s"
    
    return True, "Valid"

def validate_distance_consistency(distances, timestamps):
    """Validate that perception distances don't change too dramatically between consecutive timestamps"""
    if len(distances) < 2:
        return True, "Insufficient data for validation"
    
    for i in range(1, len(distances)):
        if not np.isnan(distances[i]) and not np.isnan(distances[i-1]):
            distance_change = abs(distances[i] - distances[i-1])
            time_diff = timestamps[i] - timestamps[i-1]
            
            if distance_change > MAX_DISTANCE_CHANGE_THRESHOLD and time_diff <= MAX_TIME_DIFF_THRESHOLD:
                return False, f"Distance change too large: {distance_change:.2f}m in {time_diff:.3f}s"
    
    return True, "Valid"

def process_json_to_csv(json_path):
    """Convert JSON bounding box data to CSV with distance calculations"""
    try:
        json_data = load_json_data(json_path)
        if json_data is None:
            return None
        
        frames = json_data.get('frames', [])
        if not frames:
            print(f"No frames found in {json_path}")
            return None
        
        # Validate perception data
        is_valid, validation_message = validate_perception_data(frames)
        if not is_valid:
            print(f"Skipping {os.path.basename(json_path)}: {validation_message}")
            return None
        
        # Prepare data for CSV
        csv_data = []
        perception_distances = []
        timestamps = []
        
        for frame in frames:
            timestamp = frame['timestamp']
            
            # Initialize row data
            row = {
                'timestamp': timestamp,
                'ego_velocity_magnitude': np.nan,
                'npc1_velocity_magnitude': np.nan,
                
                'ego_npc_distance': np.nan,
                'ego_perception_distance': np.nan,
                'closest_perception_velocity': np.nan,
                'ego_position_x': np.nan,
                'ego_position_z': np.nan,
                'npc1_position_x': np.nan,
                'npc1_position_z': np.nan,
                'perception_object_count': 0,

                'ego_velocity_lateral': np.nan,
                'npc1_velocity_lateral': np.nan,
                'closest_perception_velocity_lateral': np.nan,
            }
            
            # Extract ego data
            ego_data = frame.get('ego', {})
            if ego_data.get('velocity') and ego_data['velocity'].get('magnitude') is not None:
                row['ego_velocity_magnitude'] = ego_data['velocity']['magnitude']
                row['ego_velocity_lateral'] = ego_data['velocity'].get('z', np.nan)
            
            if ego_data.get('position'):
                row['ego_position_x'] = ego_data['position']['x']
                row['ego_position_z'] = ego_data['position']['z']
            
            # Extract NPC1 data
            npc1_data = frame.get('npc1', {})
            if npc1_data.get('velocity') and npc1_data['velocity'].get('magnitude') is not None:
                row['npc1_velocity_magnitude'] = npc1_data['velocity']['magnitude']
                row['npc1_velocity_lateral'] = npc1_data['velocity'].get('z', np.nan)
            
            if npc1_data.get('position'):
                row['npc1_position_x'] = npc1_data['position']['x']
                row['npc1_position_z'] = npc1_data['position']['z']
            
            # Calculate ego-NPC distance
            ego_polygon = create_polygon_from_box(ego_data.get('box'))
            npc1_polygon = create_polygon_from_box(npc1_data.get('box'))
            
            if ego_polygon is not None and npc1_polygon is not None:
                row['ego_npc_distance'] = calculate_polygon_distance(ego_polygon, npc1_polygon)
            
            # Calculate ego-perception distance and find closest object
            perception_objects = frame.get('perception_objects', [])
            row['perception_object_count'] = len(perception_objects)
            
            if ego_polygon is not None and perception_objects:
                closest_distance, closest_obj = find_closest_perception_object(ego_polygon, perception_objects)
                row['ego_perception_distance'] = closest_distance
                
                if closest_obj and closest_obj.get('velocity') and closest_obj['velocity'].get('magnitude') is not None:
                    row['closest_perception_velocity'] = closest_obj['velocity']['magnitude']
                    row['closest_perception_velocity_lateral'] = closest_obj['velocity'].get('y', np.nan)

            
            csv_data.append(row)
            
            # Collect data for validation
            if not np.isnan(row['ego_perception_distance']):
                perception_distances.append(row['ego_perception_distance'])
                timestamps.append(timestamp)
        
        # Validate distance consistency
        is_consistent, consistency_message = validate_distance_consistency(perception_distances, timestamps)
        if not is_consistent:
            print(f"Warning for {os.path.basename(json_path)}: {consistency_message}")
            # Continue processing but log the warning
        
        # Create DataFrame and save to CSV
        df = pd.DataFrame(csv_data)
        
        # Generate output path
        basename = os.path.basename(json_path).replace('_data.json', '')
        csv_path = os.path.join(CSV_OUTPUT_DIR, f"{basename}_distances.csv")
        
        # Save CSV
        df.to_csv(csv_path, index=False)
        
        return {
            'json_path': json_path,
            'csv_path': csv_path,
            'frame_count': len(frames),
            'valid_perception_frames': len([d for d in perception_distances if not np.isnan(d)]),
            'validation_status': validation_message,
            'consistency_status': consistency_message
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error processing {json_path}: {str(e)}")
        return None

def main():
    """Main function to process all JSON files to CSV"""
    if not os.path.exists(JSON_OUTPUT_DIR):
        print(f"JSON directory not found: {JSON_OUTPUT_DIR}")
        return
    
    # Create output directory
    os.makedirs(CSV_OUTPUT_DIR, exist_ok=True)
    
    # Find all JSON files
    json_files = [f for f in os.listdir(JSON_OUTPUT_DIR) if f.endswith('_data.json')]
    json_paths = [os.path.join(JSON_OUTPUT_DIR, f) for f in json_files]
    
    # Filter out already processed files
    if os.path.exists(CSV_OUTPUT_DIR):
        existing_csv = set(os.listdir(CSV_OUTPUT_DIR))
        filter_json_paths = []
        for json_path in json_paths:
            basename = os.path.basename(json_path).replace('_data.json', '')
            expected_csv = f"{basename}_distances.csv"
            if expected_csv not in existing_csv:
                filter_json_paths.append(json_path)
        json_paths = filter_json_paths
    
    print(f"Found {len(json_paths)} JSON files to process")
    
    successful_conversions = 0
    failed_conversions = 0
    skipped_conversions = 0
    
    results = []
    
    for json_path in tqdm(json_paths, desc="Converting JSON to CSV"):
        result = process_json_to_csv(json_path)
        if result is not None:
            if "Skipping" in str(result.get('validation_status', '')):
                skipped_conversions += 1
            else:
                successful_conversions += 1
                results.append(result)
        else:
            failed_conversions += 1
    
    # Save processing summary
    if results:
        summary_df = pd.DataFrame(results)
        summary_path = os.path.join(CSV_OUTPUT_DIR, "processing_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        print(f"Processing summary saved to: {summary_path}")
    
    print(f"\nJSON to CSV conversion complete:")
    print(f"  Successful: {successful_conversions}")
    print(f"  Skipped (insufficient perception data): {skipped_conversions}")
    print(f"  Failed: {failed_conversions}")
    print(f"  CSV files saved to: {CSV_OUTPUT_DIR}")

if __name__ == "__main__":
    main() 