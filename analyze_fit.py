"""Cycling Workout Analysis Tool.

This module provides functionality for analyzing cycling workout data from .fit files.
It processes cycling metrics including power, heart rate, speed, and elevation data
to provide comprehensive workout analysis and visualization.

Key Features:
    - Power zone analysis and normalized power calculations
    - Heart rate zone analysis
    - Climb detection and analysis
    - Workout comparison tools
    - Power-to-weight ratio calculations
    - Data visualization and export capabilities

Typical usage example:
    python analyze_fit.py

Dependencies:
    - fitparse: For reading .fit files
    - pandas: For data manipulation
    - matplotlib: For data visualization
    - numpy: For numerical computations
"""

import sys
if sys.platform.startswith("win"):
    import os
    os.system("chcp 65001 > nul")

from fitparse import FitFile
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import matplotlib.patches as mpatches
import json
from datetime import datetime, timedelta

# === CONFIGURATION ===
# Training zones and physiological parameters
FTP = 230  # Functional Threshold Power in watts
HR_MAX = 179  # Maximum heart rate in bpm
HR_REST = 58  # Resting heart rate in bpm
RIDER_WEIGHT = 75  # Default rider weight in kg

# File system paths
FIT_FOLDER = r"D:\Projects\220\fit-files"  # Source directory for .fit files
CSV_FOLDER = r"D:\Projects\220\csv-files"  # Export directory for CSV files
PNG_FOLDER = r"D:\Projects\220\png-files"  # Export directory for plot images

# Create required directories
os.makedirs(CSV_FOLDER, exist_ok=True)
os.makedirs(PNG_FOLDER, exist_ok=True)

# === ZONE DEFINITIONS ===
def get_power_zone(power: float) -> str:
    """Determines the power training zone based on FTP percentage.

    Categorizes power output into one of seven training zones based on
    percentage of FTP (Functional Threshold Power).

    Zone definitions:
        Zone 1: < 55% FTP - Active Recovery
        Zone 2: 55-75% FTP - Endurance
        Zone 3: 75-90% FTP - Tempo
        Zone 4: 90-105% FTP - Threshold
        Zone 5: 105-120% FTP - VO2 Max
        Zone 6: 120-150% FTP - Anaerobic
        Zone 7: > 150% FTP - Neuromuscular

    Args:
        power: Power output in watts.

    Returns:
        A string indicating the power training zone.
    """
    if power is None:
        return "No Data"
    pct = power / FTP
    if pct < 0.55:
        return "Zone 1 (Active Recovery)"
    elif pct < 0.75:
        return "Zone 2 (Endurance)"
    elif pct < 0.90:
        return "Zone 3 (Tempo)"
    elif pct < 1.05:
        return "Zone 4 (Threshold)"
    elif pct < 1.20:
        return "Zone 5 (VO2 Max)"
    elif pct < 1.50:
        return "Zone 6 (Anaerobic)"
    else:
        return "Zone 7 (Neuromuscular)"

def get_hr_zone(hr: float) -> str:
    """Determines the heart rate training zone based on heart rate reserve.

    Categorizes heart rate into one of five training zones using the
    heart rate reserve (HRR) method, calculated as: HRR = HRmax - HRrest.
    Zone percentages are based on common training zone definitions.

    Zone definitions:
        Below Zone 1: < 50% HRR
        Zone 1: 50-60% HRR - Recovery
        Zone 2: 60-70% HRR - Endurance
        Zone 3: 70-80% HRR - Tempo
        Zone 4: 80-90% HRR - Threshold
        Zone 5: > 90% HRR - VO2 Max

    Args:
        hr: Heart rate in beats per minute.

    Returns:
        A string indicating the heart rate training zone.
    """
    if hr is None:
        return "No Data"
    hrr = HR_MAX - HR_REST
    hr_pct = (hr - HR_REST) / hrr
    if hr_pct < 0.50:
        return "Below Zone 1"
    elif hr_pct < 0.60:
        return "Zone 1 (Recovery)"
    elif hr_pct < 0.70:
        return "Zone 2 (Endurance)"
    elif hr_pct < 0.80:
        return "Zone 3 (Tempo)"
    elif hr_pct < 0.90:
        return "Zone 4 (Threshold)"
    else:
        return "Zone 5 (VO2 Max)"

def analyze_climbs(df: pd.DataFrame) -> list:
    """Analyzes and identifies significant climbs in a cycling activity.

    Uses Garmin-like climb detection criteria to identify and analyze
    significant climbs in the ride data. A climb is considered significant
    if it meets all the following criteria:
        - Minimum horizontal distance: 300 meters
        - Minimum elevation gain: 10 meters
        - Average gradient: ≥3%

    The algorithm handles minor descents within climbs and uses smoothed
    gradient calculations to reduce noise in the data.

    Args:
        df: Pandas DataFrame containing ride data with at least 'altitude'
            and 'distance' columns. Optional columns 'power' and 'heart_rate'
            will be used for additional metrics if present.

    Returns:
        list: List of dictionaries containing climb data. Each dictionary includes:
            - start_idx: Index where climb begins
            - end_idx: Index where climb ends
            - distance: Climb distance in meters
            - elevation_gain: Total elevation gain in meters
            - avg_gradient: Average gradient as percentage
            - max_gradient: Maximum gradient as percentage
            - start_distance: Distance from ride start in km
            - duration: Climb duration in minutes
            - avg_power: Average power during climb (if available)
            - max_power: Maximum power during climb (if available)
            - avg_hr: Average heart rate during climb (if available)
            - max_hr: Maximum heart rate during climb (if available)
        Returns None if required columns are missing.

    Example:
        >>> climbs = analyze_climbs(ride_data)
        >>> if climbs:
        ...     print(f"Found {len(climbs)} significant climbs")
    """
    if 'altitude' not in df.columns or 'distance' not in df.columns:
        return None
        
    # Calculate gradient
    distance_diff = df['distance'].diff()  # in meters
    altitude_diff = df['altitude'].diff()  # in meters
    df['gradient'] = (altitude_diff / distance_diff * 100).fillna(0)
    
    # Smooth gradient to reduce noise (using 10-point moving average)
    df['gradient_smooth'] = df['gradient'].rolling(window=10, center=True).mean().fillna(0)
    
    # Initialize variables for climb detection
    climbs = []
    in_climb = False
    climb_start_idx = 0
    potential_end_idx = 0
    cumulative_descent = 0
    
    # Garmin-based parameters
    min_climb_distance = 300  # meters
    min_elevation_gain = 10   # meters
    min_gradient = 3.0       # percent
    max_descent_allowance = 5  # meters (maximum cumulative descent allowed within a climb)
    flat_section_threshold = 1.0  # percent (gradient below this is considered flat)
    
    for i in range(1, len(df)):
        current_gradient = df['gradient_smooth'].iloc[i]
        
        if not in_climb:
            if current_gradient >= min_gradient:
                # Start of potential climb
                in_climb = True
                climb_start_idx = i
                cumulative_descent = 0
                potential_end_idx = i
        else:
            # We're in a climb, check conditions
            if current_gradient >= min_gradient:
                # Strong climbing section, update potential end
                potential_end_idx = i
                cumulative_descent = 0
            elif current_gradient >= flat_section_threshold:
                # Flat-ish section, continue climb but don't update end
                pass
            elif current_gradient < 0:
                # Descending section
                cumulative_descent -= altitude_diff.iloc[i]
                
                # If too much descent, end the climb
                if cumulative_descent > max_descent_allowance:
                    in_climb = False
            else:
                # Flat or very shallow section, use as potential end
                potential_end_idx = i
            
            # Check if we should end the climb
            if not in_climb or i == len(df) - 1:
                # Calculate climb metrics
                climb_distance = df['distance'].iloc[potential_end_idx] - df['distance'].iloc[climb_start_idx]
                elevation_gain = df['altitude'].iloc[potential_end_idx] - df['altitude'].iloc[climb_start_idx]
                
                # Verify if this is a valid climb
                if (climb_distance >= min_climb_distance and 
                    elevation_gain >= min_elevation_gain):
                    
                    # Calculate climb metrics
                    avg_gradient = elevation_gain / climb_distance * 100
                    if avg_gradient >= min_gradient:
                        # Get climb segments for gradient analysis
                        climb_segment = df.iloc[climb_start_idx:potential_end_idx+1]
                        
                        climb_data = {
                            'start_idx': climb_start_idx,
                            'end_idx': potential_end_idx,
                            'distance': climb_distance,
                            'elevation_gain': elevation_gain,
                            'avg_gradient': avg_gradient,
                            'max_gradient': climb_segment['gradient_smooth'].max(),
                            'start_distance': df['distance'].iloc[climb_start_idx] / 1000,  # km
                            'duration': (df['timestamp'].iloc[potential_end_idx] - 
                                       df['timestamp'].iloc[climb_start_idx]).total_seconds() / 60,  # minutes
                        }
                        
                        # Add power and heart rate metrics if available
                        if 'power' in df.columns:
                            climb_data['avg_power'] = climb_segment['power'].mean()
                            climb_data['max_power'] = climb_segment['power'].max()
                            
                        if 'heart_rate' in df.columns:
                            climb_data['avg_hr'] = climb_segment['heart_rate'].mean()
                            climb_data['max_hr'] = climb_segment['heart_rate'].max()
                        
                        climbs.append(climb_data)
                
                in_climb = False
    
    return climbs

def analyze_climbs_offmeta(df: pd.DataFrame, garmin_climbs: list = None) -> dict:
    """Analyzes climbs using relaxed "off-meta" criteria to capture climbs missed by strict Garmin rules.

    This function implements alternative climb detection criteria that relax the standard
    Garmin rules to capture climbs that are meaningful to riders but don't meet the
    official Garmin climb detection standards.

    Off-Meta Criteria Sets:
        1. Short but Steep Climbs: 150m distance, 8m gain, ≥4% gradient
        2. Long but Gentle Climbs: 500m distance, 15m gain, ≥2% gradient
        3. Undulating Climbs: 400m distance, 12m gain, ≥2.5% gradient, 10m descent allowance
        4. Urban/Stop-Start Climbs: 200m distance, 6m gain, ≥3.5% gradient, traffic-aware
        5. Very Long and Easy Climbs: 2000m distance, 12m gain, ≥0.7% gradient, max 15m descent, 0.3% flat threshold

    Args:
        df: Pandas DataFrame containing ride data with at least 'altitude' and 'distance' columns
        garmin_climbs: Optional list of already-detected Garmin climbs to avoid overlap

    Returns:
        dict: Dictionary containing:
            - garmin_climbs: Standard Garmin-detected climbs
            - offmeta_climbs: Off-meta detected climbs with categories
            - all_climbs: Combined list with confidence indicators
            - climb_summary: Summary statistics for both types
    """
    if 'altitude' not in df.columns or 'distance' not in df.columns:
        return None
    
    # Get Garmin climbs if not provided
    if garmin_climbs is None:
        garmin_climbs = analyze_climbs(df) or []
    
    # Calculate gradient (reuse from Garmin analysis)
    distance_diff = df['distance'].diff()
    altitude_diff = df['altitude'].diff()
    df['gradient'] = (altitude_diff / distance_diff * 100).fillna(0)
    df['gradient_smooth'] = df['gradient'].rolling(window=10, center=True).mean().fillna(0)
    
    # Create mask for sections already covered by Garmin climbs
    garmin_mask = pd.Series([False] * len(df))
    for climb in garmin_climbs:
        garmin_mask.iloc[climb['start_idx']:climb['end_idx']+1] = True
    
    # Off-meta criteria sets
    offmeta_criteria = [
        {
            'name': 'Short but Steep',
            'min_distance': 150,
            'min_elevation': 8,
            'min_gradient': 4.0,
            'max_descent': 5,
            'flat_threshold': 1.0,
            'confidence': 85
        },
        {
            'name': 'Long but Gentle',
            'min_distance': 500,
            'min_elevation': 15,
            'min_gradient': 2.0,
            'max_descent': 8,
            'flat_threshold': 0.5,
            'confidence': 75
        },
        {
            'name': 'Undulating',
            'min_distance': 400,
            'min_elevation': 12,
            'min_gradient': 2.5,
            'max_descent': 10,
            'flat_threshold': 0.5,
            'confidence': 80
        },
        {
            'name': 'Urban/Stop-Start',
            'min_distance': 200,
            'min_elevation': 6,
            'min_gradient': 3.5,
            'max_descent': 6,
            'flat_threshold': 1.0,
            'confidence': 70
        },
        {
            'name': 'Very Long and Easy',
            'min_distance': 2000,  # 2 km
            'min_elevation': 8,    # reduced from 12m
            'min_gradient': 0.5,   # reduced from 0.7%
            'max_descent': 25,     # increased from 15m
            'flat_threshold': 0.1, # reduced from 0.3%
            'max_flat_length': 500, # new parameter
            'interruption_tolerance': 200, # new parameter
            'confidence': 55       # reduced confidence for ultra-relaxed rule
        },
    ]
    
    offmeta_climbs = []
    
    # Apply each criteria set to non-Garmin sections
    for criteria in offmeta_criteria:
        # Find sections not covered by Garmin climbs
        available_sections = []
        start_idx = None
        
        for i in range(len(df)):
            if not garmin_mask.iloc[i]:
                if start_idx is None:
                    start_idx = i
            elif start_idx is not None:
                available_sections.append((start_idx, i-1))
                start_idx = None
        
        # Add final section if needed
        if start_idx is not None:
            available_sections.append((start_idx, len(df)-1))
        
        # Analyze each available section
        for section_start, section_end in available_sections:
            section_df = df.iloc[section_start:section_end+1].copy()
            
            # Detect climbs in this section using current criteria
            section_climbs = _detect_climbs_in_section(
                section_df, 
                criteria, 
                section_start,
                garmin_mask
            )
            
            for climb in section_climbs:
                climb['category'] = criteria['name']
                climb['confidence'] = criteria['confidence']
                climb['type'] = 'offmeta'
                offmeta_climbs.append(climb)
    
    # Remove overlapping climbs (keep the one with higher confidence)
    offmeta_climbs = _remove_overlapping_climbs(offmeta_climbs)
    
    # Add type and confidence to Garmin climbs
    for climb in garmin_climbs:
        climb['category'] = 'Garmin Standard'
        climb['confidence'] = 100
        climb['type'] = 'garmin'
    
    # Combine all climbs
    all_climbs = garmin_climbs + offmeta_climbs
    all_climbs.sort(key=lambda x: x['start_idx'])
    
    # Create summary
    climb_summary = {
        'total_climbs': len(all_climbs),
        'garmin_climbs': len(garmin_climbs),
        'offmeta_climbs': len(offmeta_climbs),
        'categories': {}
    }
    
    # Count by category
    for climb in all_climbs:
        category = climb['category']
        if category not in climb_summary['categories']:
            climb_summary['categories'][category] = 0
        climb_summary['categories'][category] += 1
    
    return {
        'garmin_climbs': garmin_climbs,
        'offmeta_climbs': offmeta_climbs,
        'all_climbs': all_climbs,
        'climb_summary': climb_summary
    }

def _detect_climbs_in_section(section_df: pd.DataFrame, criteria: dict, section_start: int, garmin_mask: pd.Series) -> list:
    """Helper function to detect climbs in a specific section using given criteria."""
    climbs = []
    in_climb = False
    climb_start_idx = 0
    potential_end_idx = 0
    cumulative_descent = 0
    interruption_start = None
    interruption_length = 0
    
    for i in range(1, len(section_df)):
        current_gradient = section_df['gradient_smooth'].iloc[i]
        actual_idx = section_start + i
        
        # Skip if this point is already covered by a Garmin climb
        if garmin_mask.iloc[actual_idx]:
            if in_climb:
                in_climb = False
            continue
        
        if not in_climb:
            if current_gradient >= criteria['min_gradient']:
                in_climb = True
                climb_start_idx = i
                cumulative_descent = 0
                potential_end_idx = i
                interruption_start = None
                interruption_length = 0
        else:
            if current_gradient >= criteria['min_gradient']:
                # Strong climbing section, update potential end and reset interruption
                potential_end_idx = i
                cumulative_descent = 0
                interruption_start = None
                interruption_length = 0
            elif current_gradient >= criteria['flat_threshold']:
                # Flat-ish section, continue climb but don't update end
                if interruption_start is None:
                    interruption_start = i
                interruption_length = i - interruption_start
                
                # Check if flat section is too long
                if 'max_flat_length' in criteria and interruption_length > criteria['max_flat_length']:
                    in_climb = False
            elif current_gradient < 0:
                # Descending section
                cumulative_descent -= section_df['altitude'].diff().iloc[i]
                
                # Start tracking interruption
                if interruption_start is None:
                    interruption_start = i
                interruption_length = i - interruption_start
                
                # Check interruption tolerance
                if 'interruption_tolerance' in criteria and interruption_length > criteria['interruption_tolerance']:
                    in_climb = False
                elif cumulative_descent > criteria['max_descent']:
                    in_climb = False
            else:
                # Very shallow section, use as potential end
                potential_end_idx = i
            
            if not in_climb or i == len(section_df) - 1:
                # Calculate climb metrics
                climb_distance = (section_df['distance'].iloc[potential_end_idx] - 
                                section_df['distance'].iloc[climb_start_idx])
                elevation_gain = (section_df['altitude'].iloc[potential_end_idx] - 
                                section_df['altitude'].iloc[climb_start_idx])
                
                # Verify criteria
                if (climb_distance >= criteria['min_distance'] and 
                    elevation_gain >= criteria['min_elevation']):
                    
                    avg_gradient = elevation_gain / climb_distance * 100
                    if avg_gradient >= criteria['min_gradient']:
                        climb_segment = section_df.iloc[climb_start_idx:potential_end_idx+1]
                        
                        climb_data = {
                            'start_idx': section_start + climb_start_idx,
                            'end_idx': section_start + potential_end_idx,
                            'distance': climb_distance,
                            'elevation_gain': elevation_gain,
                            'avg_gradient': avg_gradient,
                            'max_gradient': climb_segment['gradient_smooth'].max(),
                            'start_distance': section_df['distance'].iloc[climb_start_idx] / 1000,
                            'duration': (section_df['timestamp'].iloc[potential_end_idx] - 
                                       section_df['timestamp'].iloc[climb_start_idx]).total_seconds() / 60,
                        }
                        
                        # Add power and heart rate metrics if available
                        if 'power' in section_df.columns:
                            climb_data['avg_power'] = climb_segment['power'].mean()
                            climb_data['max_power'] = climb_segment['power'].max()
                            
                        if 'heart_rate' in section_df.columns:
                            climb_data['avg_hr'] = climb_segment['heart_rate'].mean()
                            climb_data['max_hr'] = climb_segment['heart_rate'].max()
                        
                        climbs.append(climb_data)
                
                in_climb = False
    
    return climbs

def _remove_overlapping_climbs(climbs: list) -> list:
    """Remove overlapping climbs, keeping the one with higher confidence."""
    if not climbs:
        return climbs
    
    # Sort by confidence (descending) and start index
    climbs.sort(key=lambda x: (-x['confidence'], x['start_idx']))
    
    non_overlapping = []
    for climb in climbs:
        overlaps = False
        for existing in non_overlapping:
            # Check if climbs overlap
            if (climb['start_idx'] <= existing['end_idx'] and 
                climb['end_idx'] >= existing['start_idx']):
                overlaps = True
                break
        
        if not overlaps:
            non_overlapping.append(climb)
    
    # Sort by start index for final order
    non_overlapping.sort(key=lambda x: x['start_idx'])
    return non_overlapping

def calculate_normalized_speed(speed_series: pd.Series) -> float:
    """Calculates normalized speed by excluding stopped time.

    Normalizes speed data by:
    1. Converting speed from m/s to km/h
    2. Removing zero values (stopped time)
    3. Calculating the mean of non-zero values

    This provides a more accurate representation of moving speed by
    excluding time spent stopped at traffic lights, rest stops, etc.

    Args:
        speed_series: Pandas Series containing speed values in meters/second.

    Returns:
        float: Normalized speed in kilometers per hour.
    """
    # Convert to km/h and filter out zeros and None values
    speed_kmh = speed_series * 3.6  # Convert m/s to km/h
    moving_speed = speed_kmh[speed_kmh > 0]
    
    if len(moving_speed) == 0:
        return 0.0
        
    return moving_speed.mean()

def calculate_power_to_weight(power: float, weight: float) -> float:
    """Calculates power-to-weight ratio.

    Power-to-weight ratio is a key metric in cycling performance,
    particularly important for climbing ability. Higher values indicate
    better performance potential on climbs.

    Typical values for cyclists:
        2.0 W/kg: Beginner
        3.0 W/kg: Intermediate
        4.0 W/kg: Advanced
        5.0+ W/kg: Elite/Professional

    Args:
        power: Power output in watts.
        weight: Rider weight in kilograms.

    Returns:
        float: Power-to-weight ratio in watts/kg.
        Returns None if power is None or weight is 0/None.
    """
    if power is None or weight is None or weight == 0:
        return None
    return power / weight

def calculate_normalized_power(power_series: pd.Series) -> float:
    """Calculates Normalized Power (NP) using the standard algorithm.

    Normalized Power is a metric that better represents the physiological
    demands of a workout compared to average power. It accounts for the
    non-linear relationship between power output and physiological stress.

    The calculation follows these steps:
    1. Apply 30-second rolling average to smooth power data
    2. Raise each value to the fourth power
    3. Take the average of all values
    4. Take the fourth root of the result

    This method gives more weight to power variations and surges,
    providing a better estimate of the metabolic cost of the workout.

    Args:
        power_series: Pandas Series containing power values in watts.
            Expected to contain 1-second interval data.

    Returns:
        float: Normalized Power value in watts.
        Returns 0.0 if input is None or empty.

    Example:
        >>> power_data = pd.Series([200, 250, 300, 200, 180])
        >>> np = calculate_normalized_power(power_data)
    """
    if power_series is None or len(power_series) == 0:
        return 0.0
        
    # Calculate 30-second rolling average (assuming 1-second data points)
    rolling_avg = power_series.rolling(window=30, min_periods=1).mean()
    
    # Calculate fourth power average
    fourth_power_avg = (rolling_avg ** 4).mean()
    
    # Take the fourth root
    return fourth_power_avg ** 0.25

def process_fit_file(fit_filename: str, show_plots: bool = False) -> dict:
    """Processes a FIT file and extracts cycling metrics and statistics.

    This function reads a FIT file containing cycling activity data,
    processes it to extract various performance metrics, and optionally
    generates visualizations. The processed data is saved to CSV and PNG
    files in the configured output directories.

    The function analyzes the following metrics when available:
        - Power (watts)
        - Heart Rate (bpm)
        - Speed (km/h)
        - Cadence (rpm)
        - Altitude (meters)
        - Distance (km)
        - Calories burned

    For power data, it calculates:
        - Average and max power
        - Power-to-weight ratios
        - Normalized power
        - Training Stress Score (TSS)
        - Time in power zones

    Args:
        fit_filename: Name of the FIT file to process (must be in FIT_FOLDER)
        show_plots: Whether to display plots during processing (default: False)

    Returns:
        dict: Dictionary containing processed metrics and statistics:
            - filename: Name of the processed file
            - date: Timestamp of the activity
            - duration_min: Activity duration in minutes
            - avg_power: Average power in watts
            - max_power: Maximum power in watts
            - avg_power_per_kg: Average power-to-weight ratio
            - max_power_per_kg: Maximum power-to-weight ratio
            - np: Normalized power in watts
            - np_per_kg: Normalized power-to-weight ratio
            - tss: Training Stress Score
            - avg_hr: Average heart rate in bpm
            - max_hr: Maximum heart rate in bpm
            - avg_speed: Average speed in km/h
            - max_speed: Maximum speed in km/h
            - normalized_speed: Normalized speed in km/h
            - distance_km: Total distance in kilometers
            - elevation_gain: Total elevation gain in meters
            - calories: Estimated calories burned
            - rider_weight: Rider weight used for calculations

    Raises:
        FileNotFoundError: If the specified FIT file is not found
        ValueError: If no valid data records are found in the file
    """
    fit_path = os.path.join(FIT_FOLDER, fit_filename)
    if not os.path.exists(fit_path):
        print(f"[ERROR] File not found: {fit_path}")
        return

    base_name = os.path.splitext(fit_filename)[0]
    csv_path = os.path.join(CSV_FOLDER, f"{base_name}.csv")
    png_path = os.path.join(PNG_FOLDER, f"{base_name}.png")

    # Check for existing files and inform user
    if os.path.exists(csv_path):
        print(f"[INFO] Replacing existing CSV file: {csv_path}")
    if os.path.exists(png_path):
        print(f"[INFO] Replacing existing plot file: {png_path}")

    fitfile = FitFile(fit_path)

    records = []
    for record in fitfile.get_messages("record"):
        data = {d.name: d.value for d in record}
        records.append(data)

    if not records:
        print(f"[ERROR] No data records found in {fit_filename}")
        return

    calories = 0
    for msg in fitfile.get_messages("session"):
        for d in msg:
            if d.name == "total_calories":
                calories = d.value

    df = pd.DataFrame(records)
    
    # Initialize file_data dictionary for storing metrics
    file_data = {
        "filename": fit_filename,
        "date": None,
        "duration_min": None,
        "avg_power": None,
        "max_power": None,
        "avg_power_per_kg": None,
        "max_power_per_kg": None,
        "np": None,
        "np_per_kg": None,
        "tss": None,
        "avg_hr": None,
        "max_hr": None,
        "avg_speed": None,
        "max_speed": None,
        "normalized_speed": None,
        "distance_km": None,
        "elevation_gain": None,
        "calories": calories,
        "rider_weight": RIDER_WEIGHT
    }
    
    # Print available columns for debugging
    print("\nAvailable data fields in the FIT file:")
    print(df.columns.tolist())
    
    # Define all possible columns we might want to analyze
    possible_cols = ["timestamp", "power", "heart_rate", "cadence", "speed", "altitude", "distance"]
    
    # Check which columns are available
    available_cols = [col for col in possible_cols if col in df.columns]
    if not available_cols:
        print(f"[ERROR] No valid data fields found in {fit_filename}")
        return
        
    print("\nAnalyzing available metrics:")
    for col in available_cols:
        print(f"- {col}")
    
    # Ensure we have at least timestamp and one other metric
    if "timestamp" not in available_cols:
        print("[ERROR] No timestamp data found in the file")
        return
        
    # Create DataFrame with available columns
    df = df[available_cols].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Filter out rows where both speed and power are zero (full stop)
    # Do this before any processing or conversion
    if "speed" in df.columns and "power" in df.columns:
        before = len(df)
        # Keep rows where either speed or power is non-zero
        df = df[~((df["speed"].fillna(0) == 0) & (df["power"].fillna(0) == 0))].copy()
        df.reset_index(drop=True, inplace=True)
        after = len(df)
        removed = before - after
        print(f"[INFO] Removed {removed} full stop rows (speed=0 and power=0) - {(removed/before*100):.1f}% of data")
    
    # Convert speed to km/h if available
    if "speed" in df.columns:
        df["speed_kmh"] = df["speed"] * 3.6
    
    # Add zones for power and heart rate if available
    if "power" in df.columns:
        df["power_zone"] = df["power"].apply(get_power_zone)
    if "heart_rate" in df.columns:
        df["hr_zone"] = df["heart_rate"].apply(get_hr_zone)

    # Analyze climbs and add climb data to DataFrame
    climbs = analyze_climbs(df)
    if climbs:
        # Initialize climb columns with default values
        df['climb_number'] = 0
        df['in_climb'] = False
        df['climb_distance'] = 0.0
        df['climb_elevation'] = 0.0
        df['climb_gradient'] = 0.0
        df['climb_duration'] = 0.0
        
        # Fill in climb data
        for i, climb in enumerate(climbs, 1):
            start_idx = climb['start_idx']
            end_idx = climb['end_idx']
            
            # Mark the climb segment
            df.loc[start_idx:end_idx, 'climb_number'] = i
            df.loc[start_idx:end_idx, 'in_climb'] = True
            
            # Add climb metrics for the segment
            df.loc[start_idx:end_idx, 'climb_distance'] = climb['distance']
            df.loc[start_idx:end_idx, 'climb_elevation'] = climb['elevation_gain']
            df.loc[start_idx:end_idx, 'climb_gradient'] = climb['avg_gradient']
            df.loc[start_idx:end_idx, 'climb_duration'] = climb['duration']

    # Analyze climbs using both Garmin and off-meta criteria
    climb_analysis = analyze_climbs_offmeta(df)
    if climb_analysis:
        garmin_climbs = climb_analysis['garmin_climbs']
        offmeta_climbs = climb_analysis['offmeta_climbs']
        all_climbs = climb_analysis['all_climbs']
        climb_summary = climb_analysis['climb_summary']
        
        # Initialize climb columns with default values
        df['climb_number'] = 0
        df['in_climb'] = False
        df['climb_type'] = 'none'
        df['climb_category'] = 'none'
        df['climb_confidence'] = 0
        df['climb_distance'] = 0.0
        df['climb_elevation'] = 0.0
        df['climb_gradient'] = 0.0
        df['climb_duration'] = 0.0
        
        # Fill in climb data for all climbs
        for i, climb in enumerate(all_climbs, 1):
            start_idx = climb['start_idx']
            end_idx = climb['end_idx']
            
            # Mark the climb segment
            df.loc[start_idx:end_idx, 'climb_number'] = i
            df.loc[start_idx:end_idx, 'in_climb'] = True
            df.loc[start_idx:end_idx, 'climb_type'] = climb['type']
            df.loc[start_idx:end_idx, 'climb_category'] = climb['category']
            df.loc[start_idx:end_idx, 'climb_confidence'] = climb['confidence']
            
            # Add climb metrics for the segment
            df.loc[start_idx:end_idx, 'climb_distance'] = climb['distance']
            df.loc[start_idx:end_idx, 'climb_elevation'] = climb['elevation_gain']
            df.loc[start_idx:end_idx, 'climb_gradient'] = climb['avg_gradient']
            df.loc[start_idx:end_idx, 'climb_duration'] = climb['duration']

    # Save the processed data
    df.to_csv(csv_path, index=False)
    print(f"[OK] Saved CSV: {csv_path}")
    log_user_action('File created', {'type': 'csv', 'path': csv_path, 'source_fit': fit_filename})

    # Create plots based on available data
    plt.figure(figsize=(12, 5))
    
    # Plot power if available
    if "power" in df.columns:
        plt.plot(df["timestamp"], df["power"], label="Power (W)", color="blue")
    
    # Plot heart rate if available
    if "heart_rate" in df.columns:
        plt.plot(df["timestamp"], df["heart_rate"], label="Heart Rate (bpm)", color="red", alpha=0.7)
    
    # Plot speed if available
    if "speed_kmh" in df.columns:
        plt.plot(df["timestamp"], df["speed_kmh"], label="Speed (km/h)", color="green", alpha=0.7)
    
    # Get current handles and labels
    handles, labels = plt.gca().get_legend_handles_labels()
    
    # Highlight climb sections in the plot with different colors
    if climb_analysis and all_climbs:
        # Create custom patches for the legend
        garmin_patch = mpatches.Patch(color='yellow', alpha=0.3, label='Garmin Climbs')
        offmeta_patch = mpatches.Patch(color='orange', alpha=0.3, label='Off-Meta Climbs')
        
        for climb in all_climbs:
            start_time = df['timestamp'].iloc[climb['start_idx']]
            end_time = df['timestamp'].iloc[climb['end_idx']]
            
            # Use different colors for Garmin vs off-meta climbs
            if climb['type'] == 'garmin':
                plt.axvspan(start_time, end_time, color='yellow', alpha=0.3)
            else:  # offmeta
                plt.axvspan(start_time, end_time, color='orange', alpha=0.3)
        
        # Add patches to legend
        handles.append(garmin_patch)
        handles.append(offmeta_patch)
    elif climbs:  # Fallback to old method if only Garmin climbs
        yellow_patch = mpatches.Patch(color='yellow', alpha=0.2, label='Climbs')
        for climb in climbs:
            start_time = df['timestamp'].iloc[climb['start_idx']]
            end_time = df['timestamp'].iloc[climb['end_idx']]
            plt.axvspan(start_time, end_time, color='yellow', alpha=0.2)
        handles.append(yellow_patch)
            
    plt.title("Activity Metrics Over Time")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend(handles=handles)
    plt.tight_layout()
    plt.savefig(png_path)
    if show_plots:
        plt.show()
    else:
        plt.close()
    print(f"[OK] Saved Plot: {png_path}")
    log_user_action('File created', {'type': 'png', 'path': png_path, 'source_fit': fit_filename})

    # Print zone summaries if available
    if "power_zone" in df.columns:
        print("\nTime in Power Zones:")
        summary = df["power_zone"].value_counts().sort_index()
        total = len(df)
        for zone, count in summary.items():
            percent = (count / total) * 100
            print(f"  {zone}: {count} seconds ({percent:.1f}%)")
            
    if "hr_zone" in df.columns:
        print("\nTime in Heart Rate Zones:")
        summary = df["hr_zone"].value_counts().sort_index()
        total = len(df)
        for zone, count in summary.items():
            percent = (count / total) * 100
            print(f"  {zone}: {count} seconds ({percent:.1f}%)")

    try:
        # Calculate duration
        duration_sec = (df["timestamp"].iloc[-1] - df["timestamp"].iloc[0]).total_seconds()
        
        print("\nRide Summary:")
        print(f"  Duration: {duration_sec/60:.1f} min")
        
        # Print available metrics
        if "power" in df.columns:
            avg_power = df['power'].mean()
            max_power = df['power'].max()
            print(f"  Avg Power: {avg_power:.1f} W")
            print(f"  Max Power: {max_power:.1f} W")
            print(f"  Avg Power/kg: {calculate_power_to_weight(avg_power, RIDER_WEIGHT):.2f} W/kg")
            print(f"  Max Power/kg: {calculate_power_to_weight(max_power, RIDER_WEIGHT):.2f} W/kg")
            
            np_val = calculate_normalized_power(df["power"])
            print(f"  Normalized Power: {np_val:.1f} W")
            print(f"  Normalized Power/kg: {calculate_power_to_weight(np_val, RIDER_WEIGHT):.2f} W/kg")
            if_val = np_val / FTP
            print(f"  IF (Intensity Factor): {if_val:.2f}")
            tss = (duration_sec * np_val * if_val) / (FTP * 3600) * 100
            print(f"  TSS: {tss:.1f}")
            
        if "heart_rate" in df.columns:
            print(f"  Avg HR: {df['heart_rate'].mean():.1f} bpm")
            print(f"  Max HR: {df['heart_rate'].max():.1f} bpm")
            
        if "speed_kmh" in df.columns:
            print(f"  Avg Speed: {df['speed_kmh'].mean():.1f} km/h")
            print(f"  Max Speed: {df['speed_kmh'].max():.1f} km/h")
            
        if "distance" in df.columns:
            total_distance = df["distance"].iloc[-1] - df["distance"].iloc[0]
            print(f"  Total Distance: {total_distance/1000:.2f} km")
            
        if "altitude" in df.columns:
            elevation_gain = df["altitude"].diff().clip(lower=0).sum()
            print(f"  Elevation Gain: {elevation_gain:.1f} m")
            
            # Display comprehensive climb analysis
            if climb_analysis and all_climbs:
                print(f"\nClimb Analysis Summary:")
                print(f"  Total Climbs: {climb_summary['total_climbs']}")
                print(f"  Garmin Climbs: {climb_summary['garmin_climbs']}")
                print(f"  Off-Meta Climbs: {climb_summary['offmeta_climbs']}")
                
                print(f"\nClimb Categories:")
                for category, count in climb_summary['categories'].items():
                    print(f"  {category}: {count}")
                
                print(f"\nDetailed Climb Information:")
                for i, climb in enumerate(all_climbs, 1):
                    print(f"\nClimb {i} ({climb['type'].upper()} - {climb['category']}):")
                    print(f"  Confidence: {climb['confidence']}%")
                    print(f"  Start: {climb['start_distance']:.1f} km")
                    print(f"  Length: {climb['distance']:.0f} m")
                    print(f"  Duration: {climb['duration']:.1f} min")
                    print(f"  Elevation Gain: {climb['elevation_gain']:.0f} m")
                    print(f"  Average Gradient: {climb['avg_gradient']:.1f}%")
                    print(f"  Maximum Gradient: {climb['max_gradient']:.1f}%")
                    
                    if 'avg_power' in climb:
                        print(f"  Average Power: {climb['avg_power']:.0f} W")
                        print(f"  Maximum Power: {climb['max_power']:.0f} W")
                    
                    if 'avg_hr' in climb:
                        print(f"  Average HR: {climb['avg_hr']:.0f} bpm")
                        print(f"  Maximum HR: {climb['max_hr']:.0f} bpm")
            elif climbs:  # Fallback to old method
                print("\nSignificant Climbs (Garmin Criteria):")
                for i, climb in enumerate(climbs, 1):
                    print(f"\nClimb {i}:")
                    print(f"  Start: {climb['start_distance']:.1f} km")
                    print(f"  Length: {climb['distance']:.0f} m")
                    print(f"  Duration: {climb['duration']:.1f} min")
                    print(f"  Elevation Gain: {climb['elevation_gain']:.0f} m")
                    print(f"  Average Gradient: {climb['avg_gradient']:.1f}%")
                    print(f"  Maximum Gradient: {climb['max_gradient']:.1f}%")
                    
                    if 'avg_power' in climb:
                        print(f"  Average Power: {climb['avg_power']:.0f} W")
                        print(f"  Maximum Power: {climb['max_power']:.0f} W")
                    
                    if 'avg_hr' in climb:
                        print(f"  Average HR: {climb['avg_hr']:.0f} bpm")
                        print(f"  Maximum HR: {climb['max_hr']:.0f} bpm")
            
        if calories > 0:
            print(f"  Calories: {calories} kcal")
            
        if "speed" in df.columns:
            speed_kmh = df["speed"] * 3.6
            file_data["avg_speed"] = speed_kmh.mean()
            file_data["max_speed"] = speed_kmh.max()
            file_data["normalized_speed"] = calculate_normalized_speed(df["speed"])
            print(f"  Avg Speed (all time): {file_data['avg_speed']:.1f} km/h")
            print(f"  Normalized Speed (moving only): {file_data['normalized_speed']:.1f} km/h")
            print(f"  Max Speed: {file_data['max_speed']:.1f} km/h")
            
    except Exception as e:
        print(f"[ERROR] Failed to calculate ride metrics: {str(e)}")

    return file_data

def weekly_summary(file_list: list) -> None:
    """Generates a summary of cycling activities for the past week.

    Processes all FIT files from the past week and generates aggregate
    statistics including total distance, time, elevation gain, and
    training load. Also calculates averages for key metrics.

    Args:
        file_list: List of FIT filenames to include in the summary

    Returns:
        None: Results are printed to console
    """
    print("\nWeekly Summary:")
    for fit_file in file_list:
        print(f"\n--- {fit_file} ---")
        process_fit_file(os.path.basename(fit_file))

def create_overview_csv() -> None:
    """Creates a CSV file containing an overview of all cycling activities.

    Processes all FIT files in the configured directory and creates a
    comprehensive CSV file containing key metrics for each activity.
    This allows for long-term tracking and analysis of training progress.

    The CSV includes the following metrics for each activity:
        - Date and duration
        - Distance and elevation gain
        - Power metrics (average, normalized, max)
        - Heart rate metrics (average, max)
        - Speed metrics (average, max, normalized)
        - Training stress score
        - Calories burned
        - Power-to-weight ratios

    The file is saved in the configured CSV_FOLDER with the name
    'ride_overview.csv'.

    Returns:
        None: Results are saved to CSV file and success/failure is
        printed to console
    """
    print("\nCreating overview CSV file...")
    
    # Get all FIT files
    fit_files = sorted(glob.glob(os.path.join(FIT_FOLDER, "*.fit")))
    if not fit_files:
        print("[ERROR] No FIT files found in the source folder")
        return
        
    overview_data = []
    
    for fit_file in fit_files:
        try:
            print(f"\nProcessing {os.path.basename(fit_file)}...")
            fitfile = FitFile(fit_file)
            
            # Initialize data dictionary for this file
            file_data = {
                "filename": os.path.basename(fit_file),
                "date": None,
                "duration_min": None,
                "avg_power": None,
                "max_power": None,
                "avg_power_per_kg": None,
                "max_power_per_kg": None,
                "np": None,
                "np_per_kg": None,
                "tss": None,
                "avg_hr": None,
                "max_hr": None,
                "avg_speed": None,
                "max_speed": None,
                "distance_km": None,
                "elevation_gain": None,
                "calories": None,
                "rider_weight": RIDER_WEIGHT
            }
            
            # Get records
            records = []
            for record in fitfile.get_messages("record"):
                data = {d.name: d.value for d in record}
                records.append(data)
                
            if not records:
                print(f"[WARNING] No data records found in {os.path.basename(fit_file)}")
                continue
                
            df = pd.DataFrame(records)
            
            # Get session data (calories, etc.)
            for msg in fitfile.get_messages("session"):
                for d in msg:
                    if d.name == "total_calories":
                        file_data["calories"] = d.value
                    elif d.name == "start_time":
                        file_data["date"] = d.value
            
            # Process available metrics
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                duration_sec = (df["timestamp"].iloc[-1] - df["timestamp"].iloc[0]).total_seconds()
                file_data["duration_min"] = duration_sec / 60
                
            if "power" in df.columns:
                file_data["avg_power"] = df["power"].mean()
                file_data["max_power"] = df["power"].max()
                file_data["avg_power_per_kg"] = calculate_power_to_weight(file_data["avg_power"], RIDER_WEIGHT)
                file_data["max_power_per_kg"] = calculate_power_to_weight(file_data["max_power"], RIDER_WEIGHT)
                np_30s = df["power"].rolling(window=30, min_periods=1).mean() ** 4
                np_val = np_30s.mean() ** 0.25
                file_data["np"] = np_val
                file_data["np_per_kg"] = calculate_power_to_weight(np_val, RIDER_WEIGHT)
                if_val = np_val / FTP
                file_data["tss"] = (duration_sec * np_val * if_val) / (FTP * 3600) * 100
                
            if "heart_rate" in df.columns:
                file_data["avg_hr"] = df["heart_rate"].mean()
                file_data["max_hr"] = df["heart_rate"].max()
                
            if "speed" in df.columns:
                speed_kmh = df["speed"] * 3.6
                file_data["avg_speed"] = speed_kmh.mean()
                file_data["max_speed"] = speed_kmh.max()
                file_data["normalized_speed"] = calculate_normalized_speed(df["speed"])
                print(f"  Avg Speed (all time): {file_data['avg_speed']:.1f} km/h")
                print(f"  Normalized Speed (moving only): {file_data['normalized_speed']:.1f} km/h")
                print(f"  Max Speed: {file_data['max_speed']:.1f} km/h")
                
            if "distance" in df.columns:
                file_data["distance_km"] = (df["distance"].iloc[-1] - df["distance"].iloc[0]) / 1000
                
            if "altitude" in df.columns:
                file_data["elevation_gain"] = df["altitude"].diff().clip(lower=0).sum()
            
            overview_data.append(file_data)
            print(f"[OK] Processed {os.path.basename(fit_file)}")
            
        except Exception as e:
            print(f"[ERROR] Failed to process {os.path.basename(fit_file)}: {str(e)}")
            continue
    
    if overview_data:
        # Create overview DataFrame and save to CSV
        overview_df = pd.DataFrame(overview_data)
        overview_path = os.path.join(CSV_FOLDER, "ride_overview.csv")
        overview_df.to_csv(overview_path, index=False)
        print(f"\n[OK] Created overview file: {overview_path}")
        log_user_action('File created', {'type': 'csv', 'path': overview_path, 'source': 'overview'})
        
        # Print summary statistics
        print("\nOverview Summary:")
        print(f"Total rides analyzed: {len(overview_data)}")
        if "duration_min" in overview_df.columns:
            print(f"Total time: {overview_df['duration_min'].sum():.1f} minutes")
        if "distance_km" in overview_df.columns:
            print(f"Total distance: {overview_df['distance_km'].sum():.1f} km")
        if "elevation_gain" in overview_df.columns:
            print(f"Total elevation gain: {overview_df['elevation_gain'].sum():.1f} m")
    else:
        print("[ERROR] No data could be processed from any FIT files")

def compare_rides(fit_file1: str, fit_file2: str) -> None:
    """Compares two cycling activities and visualizes their differences.

    This function processes two FIT files and creates a detailed comparison
    of their metrics, including power, heart rate, speed, and elevation
    profiles. It generates comparative visualizations and calculates
    percentage differences between key metrics.

    The comparison includes:
        - Overall metrics (time, distance, elevation, etc.)
        - Power metrics (average, normalized, max)
        - Heart rate metrics (average, max)
        - Speed metrics (average, max, normalized)
        - Elevation profiles
        - Power and heart rate distributions
        - Time spent in different training zones

    The function also calculates the decoupling ratio (power:heart rate)
    for both rides to compare training stress and fatigue effects.

    Args:
        fit_file1: Name of the first FIT file to compare (must be in FIT_FOLDER)
        fit_file2: Name of the second FIT file to compare (must be in FIT_FOLDER)

    Returns:
        None: Results are printed to console and plots are displayed/saved

    Note:
        - Both files must contain compatible data fields for comparison
        - The function handles different ride durations by comparing
          relative metrics and aligning data where possible
        - Missing metrics in either file are noted but don't prevent
          comparison of available metrics
    """
    # Process both files
    print("\nProcessing files for comparison...")
    df1 = process_fit_file(fit_file1)
    df2 = process_fit_file(fit_file2)

    # Create DataFrames for both rides
    records1 = []
    records2 = []
    for record in fitfile1.get_messages("record"):
        records1.append({d.name: d.value for d in record})
    for record in fitfile2.get_messages("record"):
        records2.append({d.name: d.value for d in record})
        
    df1 = pd.DataFrame(records1)
    df2 = pd.DataFrame(records2)
    
    # Convert timestamps
    df1["timestamp"] = pd.to_datetime(df1["timestamp"])
    df2["timestamp"] = pd.to_datetime(df2["timestamp"])
    
    # Filter out rows where both speed and power are zero (full stop) for both rides
    if "speed" in df1.columns and "power" in df1.columns:
        before = len(df1)
        df1 = df1[~((df1["speed"].fillna(0) == 0) & (df1["power"].fillna(0) == 0))].copy()
        df1.reset_index(drop=True, inplace=True)
        after = len(df1)
        removed = before - after
        print(f"[INFO] Ride 1: Removed {removed} full stop rows (speed=0 and power=0) - {(removed/before*100):.1f}% of data")
    
    if "speed" in df2.columns and "power" in df2.columns:
        before = len(df2)
        df2 = df2[~((df2["speed"].fillna(0) == 0) & (df2["power"].fillna(0) == 0))].copy()
        df2.reset_index(drop=True, inplace=True)
        after = len(df2)
        removed = before - after
        print(f"[INFO] Ride 2: Removed {removed} full stop rows (speed=0 and power=0) - {(removed/before*100):.1f}% of data")
    
    # Calculate distance-based metrics
    if "distance" in df1.columns and "distance" in df2.columns:
        # Resample data to equal distance intervals
        distance_interval = 10  # meters
        max_dist = min(df1["distance"].max(), df2["distance"].max())
        distances = np.arange(0, max_dist, distance_interval)
        
        # Create comparison plots
        plt.figure(figsize=(15, 10))
        
        # Power comparison
        if "power" in df1.columns and "power" in df2.columns:
            plt.subplot(2, 1, 1)
            # Calculate rolling averages for smoother plots
            power1 = df1["power"].rolling(window=30, min_periods=1).mean()
            power2 = df2["power"].rolling(window=30, min_periods=1).mean()
            plt.plot(df1["distance"], power1, label=f"Power - {fit_file1}", color="blue")
            plt.plot(df2["distance"], power2, label=f"Power - {fit_file2}", color="red")
            plt.xlabel("Distance (m)")
            plt.ylabel("Power (W)")
            plt.title("Power Comparison")
            plt.legend()
            
            # Calculate and display key power metrics
            print("\nPower Metrics:")
            print(f"Ride 1 - Avg Power: {df1['power'].mean():.1f}W, NP: {calculate_normalized_power(df1['power']):.1f}W")
            print(f"Ride 2 - Avg Power: {df2['power'].mean():.1f}W, NP: {calculate_normalized_power(df2['power']):.1f}W")
            
            # Calculate time in power zones
            if "power_zone" in df1.columns and "power_zone" in df2.columns:
                print("\nTime in Power Zones:")
                zones1 = df1["power_zone"].value_counts().sort_index()
                zones2 = df2["power_zone"].value_counts().sort_index()
                for zone in sorted(set(zones1.index) | set(zones2.index)):
                    time1 = zones1.get(zone, 0)
                    time2 = zones2.get(zone, 0)
                    print(f"{zone}:")
                    print(f"  Ride 1: {time1} seconds ({time1/len(df1)*100:.1f}%)")
                    print(f"  Ride 2: {time2} seconds ({time2/len(df2)*100:.1f}%)")
        
        # Speed comparison
        if "speed" in df1.columns and "speed" in df2.columns:
            plt.subplot(2, 1, 2)
            speed1_kmh = df1["speed"] * 3.6
            speed2_kmh = df2["speed"] * 3.6
            plt.plot(df1["distance"], speed1_kmh, label=f"Speed - {fit_file1}", color="blue")
            plt.plot(df2["distance"], speed2_kmh, label=f"Speed - {fit_file2}", color="red")
            plt.xlabel("Distance (m)")
            plt.ylabel("Speed (km/h)")
            plt.title("Speed Comparison")
            plt.legend()
            
            # Calculate and display key speed metrics
            print("\nSpeed Metrics:")
            print(f"Ride 1 - Avg Speed: {speed1_kmh.mean():.1f}km/h, Max: {speed1_kmh.max():.1f}km/h")
            print(f"Ride 2 - Avg Speed: {speed2_kmh.mean():.1f}km/h, Max: {speed2_kmh.max():.1f}km/h")
            
            # Calculate normalized speeds (excluding stops)
            norm_speed1 = calculate_normalized_speed(df1["speed"])
            norm_speed2 = calculate_normalized_speed(df2["speed"])
            print(f"Ride 1 - Normalized Speed: {norm_speed1:.1f}km/h")
            print(f"Ride 2 - Normalized Speed: {norm_speed2:.1f}km/h")
        
        # Advanced Training Metrics
        if "heart_rate" in df1.columns and "heart_rate" in df2.columns and "power" in df1.columns and "power" in df2.columns:
            print("\nTraining Metrics:")
            
            # Calculate Efficiency Factor (EF = NP/Avg HR)
            ef1 = calculate_normalized_power(df1["power"]) / df1["heart_rate"].mean()
            ef2 = calculate_normalized_power(df2["power"]) / df2["heart_rate"].mean()
            print(f"Efficiency Factor:")
            print(f"  Ride 1: {ef1:.2f}")
            print(f"  Ride 2: {ef2:.2f}")
            
            # Calculate Power-to-Heart Rate Decoupling
            def calculate_decoupling(df):
                midpoint = len(df) // 2
                first_half_ef = calculate_normalized_power(df["power"][:midpoint]) / df["heart_rate"][:midpoint].mean()
                second_half_ef = calculate_normalized_power(df["power"][midpoint:]) / df["heart_rate"][midpoint:].mean()
                return ((second_half_ef - first_half_ef) / first_half_ef) * 100
            
            decoupling1 = calculate_decoupling(df1)
            decoupling2 = calculate_decoupling(df2)
            print(f"Power:HR Decoupling:")
            print(f"  Ride 1: {decoupling1:.1f}%")
            print(f"  Ride 2: {decoupling2:.1f}%")
            
            # Calculate Training Stress Score (TSS)
            duration1_hours = (df1["timestamp"].max() - df1["timestamp"].min()).total_seconds() / 3600
            duration2_hours = (df2["timestamp"].max() - df2["timestamp"].min()).total_seconds() / 3600
            
            if1 = calculate_normalized_power(df1["power"]) / FTP
            if2 = calculate_normalized_power(df2["power"]) / FTP
            
            tss1 = (duration1_hours * calculate_normalized_power(df1["power"]) * if1) / (FTP * 3600) * 100
            tss2 = (duration2_hours * calculate_normalized_power(df2["power"]) * if2) / (FTP * 3600) * 100
            
            print(f"Training Stress Score (TSS):")
            print(f"  Ride 1: {tss1:.1f}")
            print(f"  Ride 2: {tss2:.1f}")
        
        # Save the comparison plot
        plt.tight_layout()
        comparison_path = os.path.join(PNG_FOLDER, f"comparison_{os.path.splitext(fit_file1)[0]}_{os.path.splitext(fit_file2)[0]}.png")
        plt.savefig(comparison_path)
        plt.close()
        print(f"\n[OK] Saved comparison plot: {comparison_path}")
    else:
        print("[ERROR] Distance data not available in one or both files")

def update_rider_stats() -> None:
    """Updates rider statistics and configuration values.

    Provides an interactive interface for updating rider-specific values
    used in calculations throughout the program. These values are saved
    to a configuration file for persistence between sessions.

    The following values can be updated:
        - FTP (Functional Threshold Power)
        - Maximum heart rate
        - Resting heart rate
        - Rider weight

    The function validates all inputs and provides current values as
    reference. Changes are saved to 'rider_stats.json' in the program
    directory.

    Returns:
        None: Results are saved to file and confirmation is printed
    """
    global FTP, HR_MAX, HR_REST, RIDER_WEIGHT
    
    print("\nUpdate Rider Statistics")
    print("Current values:")
    print(f"1. FTP: {FTP} watts")
    print(f"2. Max Heart Rate: {HR_MAX} bpm")
    print(f"3. Resting Heart Rate: {HR_REST} bpm")
    print(f"4. Weight: {RIDER_WEIGHT} kg")
    
    print("\nEnter new values (or press Enter to keep current value)")
    
    # Update Weight with validation
    while True:
        weight_input = input(f"New Weight (40-150 kg) [{RIDER_WEIGHT}]: ").strip()
        if not weight_input:  # Keep current value
            break
        try:
            new_weight = float(weight_input)
            if 40 <= new_weight <= 150:
                RIDER_WEIGHT = new_weight
                break
            else:
                print("Weight must be between 40 and 150 kg")
        except ValueError:
            print("Please enter a valid number")
    
    # Update FTP with validation
    while True:
        ftp_input = input(f"New FTP (100-500 watts) [{FTP}]: ").strip()
        if not ftp_input:  # Keep current value
            break
        try:
            new_ftp = int(ftp_input)
            if 100 <= new_ftp <= 500:
                FTP = new_ftp
                break
            else:
                print("FTP must be between 100 and 500 watts")
        except ValueError:
            print("Please enter a valid number")
    
    # Update Max HR with validation
    while True:
        hr_max_input = input(f"New Max Heart Rate (120-220 bpm) [{HR_MAX}]: ").strip()
        if not hr_max_input:  # Keep current value
            break
        try:
            new_hr_max = int(hr_max_input)
            if 120 <= new_hr_max <= 220:
                HR_MAX = new_hr_max
                break
            else:
                print("Max Heart Rate must be between 120 and 220 bpm")
        except ValueError:
            print("Please enter a valid number")
    
    # Update Resting HR with validation
    while True:
        hr_rest_input = input(f"New Resting Heart Rate (30-100 bpm) [{HR_REST}]: ").strip()
        if not hr_rest_input:  # Keep current value
            break
        try:
            new_hr_rest = int(hr_rest_input)
            if 30 <= new_hr_rest <= 100:
                if new_hr_rest >= HR_MAX:
                    print("Resting Heart Rate must be lower than Max Heart Rate")
                    continue
                HR_REST = new_hr_rest
                break
            else:
                print("Resting Heart Rate must be between 30 and 100 bpm")
        except ValueError:
            print("Please enter a valid number")
    
    # Save the new values to a configuration file
    config_path = os.path.join(os.path.dirname(__file__), "rider_config.txt")
    try:
        with open(config_path, "w") as f:
            f.write(f"FTP={FTP}\n")
            f.write(f"HR_MAX={HR_MAX}\n")
            f.write(f"HR_REST={HR_REST}\n")
            f.write(f"WEIGHT={RIDER_WEIGHT}\n")
        print(f"\n[OK] Saved configuration to {config_path}")
        
        print("\nUpdated Rider Statistics:")
        print(f"FTP: {FTP} watts")
        print(f"Max Heart Rate: {HR_MAX} bpm")
        print(f"Resting Heart Rate: {HR_REST} bpm")
        print(f"Weight: {RIDER_WEIGHT} kg")
        print(f"FTP/kg: {calculate_power_to_weight(FTP, RIDER_WEIGHT):.2f} W/kg")
        
    except Exception as e:
        print(f"[ERROR] Failed to save configuration: {str(e)}")

def load_rider_stats() -> dict:
    """Loads rider statistics from configuration file.

    Attempts to load rider-specific values from 'rider_config.txt'.
    If the file doesn't exist or is invalid, default values are used.

    Returns:
        dict: Dictionary containing rider statistics:
            - ftp: Functional Threshold Power in watts
            - hr_max: Maximum heart rate in bpm
            - hr_rest: Resting heart rate in bpm
            - weight: Rider weight in kg
            Returns default values if file cannot be loaded
    """
    try:
        config_path = os.path.join(os.path.dirname(__file__), "rider_config.txt")
        if os.path.exists(config_path):
            stats = {}
            with open(config_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if '=' in line:
                        key, value = line.split('=', 1)
                        key = key.strip()
                        value = value.strip()
                        
                        # Convert value to appropriate type
                        if key == 'FTP':
                            stats['ftp'] = int(value)
                        elif key == 'HR_MAX':
                            stats['hr_max'] = int(value)
                        elif key == 'HR_REST':
                            stats['hr_rest'] = int(value)
                        elif key == 'WEIGHT':
                            stats['weight'] = float(value)
            
            # Validate that we have all required values
            if all(key in stats for key in ['ftp', 'hr_max', 'hr_rest', 'weight']):
                return stats
            else:
                print("[WARNING] Incomplete configuration file, using default values")
        else:
            print("[WARNING] Configuration file not found, using default values")
    except Exception as e:
        print(f"[WARNING] Could not load rider stats: {str(e)}")
        print("Using default values")
    
    return {
        'ftp': FTP,
        'hr_max': HR_MAX,
        'hr_rest': HR_REST,
        'weight': RIDER_WEIGHT
    }

def analyze_gym_session(fit_filename: str, show_plots: bool = False) -> dict:
    """Analyzes a gym/strength training session from a FIT file.

    This function processes FIT files from gym/strength training sessions,
    focusing on heart rate data to analyze workout intensity, recovery periods,
    and overall cardiovascular load.

    Key metrics analyzed:
        - Overall session duration
        - Time in different heart rate zones
        - Recovery periods (periods of lower heart rate)
        - Maximum and average heart rate
        - Heart rate variability and trends
        - Estimated calorie burn
        - Work-to-rest ratio

    Args:
        fit_filename: Name of the FIT file to process (must be in FIT_FOLDER)
        show_plots: Whether to display plots during processing (default: False)

    Returns:
        dict: Dictionary containing processed metrics and statistics:
            - filename: Name of the processed file
            - date: Timestamp of the activity
            - duration_min: Activity duration in minutes
            - avg_hr: Average heart rate in bpm
            - max_hr: Maximum heart rate in bpm
            - recovery_periods: Number of detected recovery periods
            - work_rest_ratio: Ratio of high-intensity to recovery periods
            - calories: Estimated calories burned
            - time_in_zones: Dictionary of time spent in each HR zone
    """
    fit_path = os.path.join(FIT_FOLDER, fit_filename)
    if not os.path.exists(fit_path):
        print(f"[ERROR] File not found: {fit_path}")
        return

    base_name = os.path.splitext(fit_filename)[0]
    csv_path = os.path.join(CSV_FOLDER, f"{base_name}_gym.csv")
    png_path = os.path.join(PNG_FOLDER, f"{base_name}_gym.png")

    fitfile = FitFile(fit_path)

    records = []
    for record in fitfile.get_messages("record"):
        data = {d.name: d.value for d in record}
        records.append(data)

    if not records:
        print(f"[ERROR] No data records found in {fit_filename}")
        return

    df = pd.DataFrame(records)
    
    # Initialize metrics dictionary
    metrics = {
        "filename": fit_filename,
        "date": None,
        "duration_min": None,
        "avg_hr": None,
        "max_hr": None,
        "recovery_periods": 0,
        "work_rest_ratio": None,
        "calories": 0,
        "time_in_zones": {}
    }

    # Basic validation
    if "timestamp" not in df.columns or "heart_rate" not in df.columns:
        print("[ERROR] Required heart rate data not found in file")
        return metrics

    # Convert timestamp
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    
    # Calculate session duration
    duration_sec = (df["timestamp"].max() - df["timestamp"].min()).total_seconds()
    metrics["duration_min"] = duration_sec / 60
    metrics["date"] = df["timestamp"].min()

    # Heart rate analysis
    if "heart_rate" in df.columns:
        # Basic HR stats
        metrics["avg_hr"] = df["heart_rate"].mean()
        metrics["max_hr"] = df["heart_rate"].max()
        
        # Add HR zones
        df["hr_zone"] = df["heart_rate"].apply(get_hr_zone)
        
        # Calculate time in zones
        zone_counts = df["hr_zone"].value_counts()
        total_records = len(df)
        metrics["time_in_zones"] = {
            zone: {"seconds": count, "percentage": (count/total_records)*100}
            for zone, count in zone_counts.items()
        }

        # Detect recovery periods
        # A recovery period is defined as HR dropping below 65% of max HR for at least 30 seconds
        recovery_threshold = HR_MAX * 0.65
        df["is_recovery"] = df["heart_rate"] < recovery_threshold
        
        # Use rolling window to find sustained recovery periods (30 seconds)
        window_size = 30  # 30 seconds assuming 1-second recording intervals
        df["sustained_recovery"] = df["is_recovery"].rolling(window=window_size, min_periods=window_size).mean() == 1
        
        # Count recovery periods (transitions from work to recovery)
        recovery_transitions = df["sustained_recovery"].diff() == 1
        metrics["recovery_periods"] = recovery_transitions.sum()

        # Calculate work-to-rest ratio
        total_recovery_time = df["sustained_recovery"].sum()
        total_work_time = len(df) - total_recovery_time
        metrics["work_rest_ratio"] = total_work_time / total_recovery_time if total_recovery_time > 0 else float('inf')

    # Get calories if available
    for msg in fitfile.get_messages("session"):
        for d in msg:
            if d.name == "total_calories":
                metrics["calories"] = d.value

    # Create visualization
    plt.figure(figsize=(15, 10))

    # Heart Rate Plot
    plt.subplot(2, 1, 1)
    plt.plot(df["timestamp"], df["heart_rate"], label="Heart Rate", color="red")
    plt.title("Gym Session Heart Rate Analysis")
    plt.xlabel("Time")
    plt.ylabel("Heart Rate (bpm)")
    plt.grid(True)
    plt.legend()

    # Heart Rate Zone Distribution
    plt.subplot(2, 1, 2)
    zone_data = pd.Series(metrics["time_in_zones"]).apply(lambda x: x["percentage"])
    zone_data.plot(kind="bar")
    plt.title("Time in Heart Rate Zones")
    plt.xlabel("Zone")
    plt.ylabel("Percentage of Time")
    plt.tight_layout()

    # Save plots
    plt.savefig(png_path)
    if show_plots:
        plt.show()
    else:
        plt.close()
    df.to_csv(csv_path, index=False)
    log_user_action('File created', {'type': 'csv', 'path': csv_path, 'source_fit': fit_filename, 'session': 'gym'})
    log_user_action('File created', {'type': 'png', 'path': png_path, 'source_fit': fit_filename, 'session': 'gym'})

    # Save processed data
    df.to_csv(csv_path, index=False)

    # Print analysis summary
    print(f"\nGym Session Analysis for {fit_filename}")
    print(f"Duration: {metrics['duration_min']:.1f} minutes")
    print(f"Average Heart Rate: {metrics['avg_hr']:.1f} bpm")
    print(f"Maximum Heart Rate: {metrics['max_hr']:.1f} bpm")
    print(f"Recovery Periods: {metrics['recovery_periods']}")
    print(f"Work-to-Rest Ratio: {metrics['work_rest_ratio']:.2f}")
    print(f"Calories Burned: {metrics['calories']}")
    
    print("\nTime in Heart Rate Zones:")
    for zone, data in metrics["time_in_zones"].items():
        print(f"{zone}: {data['percentage']:.1f}% ({data['seconds']} seconds)")

    return metrics

def merge_rides(fit_files: list) -> None:
    """Merges multiple FIT files from the same day into a single analysis.

    This function combines data from multiple FIT files, typically used when
    a single ride was recorded as multiple segments. It processes the files
    in chronological order and combines their metrics while handling:
        - Total distance calculation
        - Cumulative elevation gain
        - Power and heart rate continuity
        - Overall time and duration
        - Combined workout stress calculation

    Args:
        fit_files: List of FIT filenames to merge (must be in FIT_FOLDER)

    Returns:
        None: Results are saved to files and summary is printed to console
    """
    if not fit_files:
        print("[ERROR] No files provided for merging")
        return

    print("\nMerging rides...")
    
    # Initialize accumulators for all metrics
    all_records = []
    total_calories = 0
    total_distance = 0
    total_elevation_gain = 0
    total_moving_time = 0  # Time when actually moving (speed > 0)
    weighted_power_sum = 0  # For calculating true average power
    weighted_hr_sum = 0     # For calculating true average heart rate
    moving_power_sum = 0    # Power sum only when moving
    moving_hr_sum = 0       # HR sum only when moving
    moving_records = 0      # Count of records when moving
    max_power = 0
    max_hr = 0
    total_records = 0
    earliest_timestamp = None
    latest_timestamp = None
    
    # Process each file in chronological order
    for fit_file in fit_files:
        fit_path = os.path.join(FIT_FOLDER, fit_file)
        try:
            fitfile = FitFile(fit_path)
            
            # Get records from this file
            file_records = []
            start_distance = None
            end_distance = None
            segment_records = 0
            
            for record in fitfile.get_messages("record"):
                data = {d.name: d.value for d in record}
                if "distance" in data:
                    if start_distance is None:
                        start_distance = data["distance"]
                    end_distance = data["distance"]
                if "timestamp" in data:
                    timestamp = pd.to_datetime(data["timestamp"])
                    if earliest_timestamp is None or timestamp < earliest_timestamp:
                        earliest_timestamp = timestamp
                    if latest_timestamp is None or timestamp > latest_timestamp:
                        latest_timestamp = timestamp
                
                # Only count power and HR when actually moving
                is_moving = data.get("speed", 0) > 0.5  # More than 0.5 m/s (1.8 km/h)
                if is_moving:
                    total_moving_time += 1  # Assuming 1-second recording intervals
                    moving_records += 1
                    
                    if "power" in data and data["power"] is not None:
                        power = min(data["power"], 1500)  # Cap power at 1500W to filter spikes
                        max_power = max(max_power, power)
                        moving_power_sum += power
                        
                    if "heart_rate" in data and data["heart_rate"] is not None:
                        max_hr = max(max_hr, data["heart_rate"])
                        moving_hr_sum += data["heart_rate"]
                
                segment_records += 1
                file_records.append(data)
            
            # Calculate segment metrics
            if start_distance is not None and end_distance is not None:
                segment_distance = end_distance - start_distance
                total_distance += segment_distance
            
            # Calculate segment elevation gain
            if file_records and "altitude" in file_records[0]:
                segment_elevation = sum(max(0, file_records[i+1]["altitude"] - file_records[i]["altitude"]) 
                                     for i in range(len(file_records)-1))
                total_elevation_gain += segment_elevation
            
            if file_records:
                # Adjust distance values to be continuous
                if all_records and "distance" in file_records[0]:
                    distance_offset = total_distance - file_records[0]["distance"]
                    for record in file_records:
                        if "distance" in record:
                            record["distance"] += distance_offset
                all_records.extend(file_records)
                total_records += segment_records
                
            # Get calories if available
            for msg in fitfile.get_messages("session"):
                for d in msg:
                    if d.name == "total_calories":
                        total_calories += d.value or 0
                        
        except Exception as e:
            print(f"[ERROR] Failed to process {fit_file}: {str(e)}")
            continue
    
    if not all_records:
        print("[ERROR] No valid records found in any of the files")
        return
        
    # Convert to DataFrame and sort by timestamp
    df = pd.DataFrame(all_records)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    
    # Filter out rows where both speed and power are zero (full stop)
    if "speed" in df.columns and "power" in df.columns:
        before = len(df)
        # Keep rows where either speed or power is non-zero
        df = df[~((df["speed"].fillna(0) == 0) & (df["power"].fillna(0) == 0))].copy()
        df.reset_index(drop=True, inplace=True)
        after = len(df)
        removed = before - after
        print(f"[INFO] Removed {removed} full stop rows (speed=0 and power=0) - {(removed/before*100):.1f}% of data")
    
    # Calculate total duration from earliest to latest timestamp
    total_duration = (latest_timestamp - earliest_timestamp).total_seconds() / 60  # Convert to minutes
    
    # Generate base name for output files using date
    date_str = df["timestamp"].min().strftime("%Y%m%d")
    base_name = f"merged_ride_{date_str}"
    csv_path = os.path.join(CSV_FOLDER, f"{base_name}.csv")
    png_path = os.path.join(PNG_FOLDER, f"{base_name}.png")
    
    # Calculate final averages (using moving time only)
    avg_power = moving_power_sum / moving_records if moving_records > 0 else 0
    avg_hr = moving_hr_sum / moving_records if moving_records > 0 else 0
    
    # Calculate overall normalized power and TSS using moving time
    if "power" in df.columns:
        # Filter for moving periods
        moving_mask = df["speed"] > 0.5
        moving_power = df.loc[moving_mask, "power"]
        overall_np = calculate_normalized_power(moving_power)
        
        if FTP > 0:
            overall_if = overall_np / FTP
            overall_tss = (total_moving_time * overall_np * overall_if) / (FTP * 3600) * 100
        else:
            overall_tss = 0
    else:
        overall_np = 0
        overall_tss = 0
    
    print("\nMerged Ride Summary:")
    print(f"Date: {df['timestamp'].min().strftime('%Y-%m-%d')}")
    print(f"Total Duration: {total_duration:.1f} minutes")
    print(f"Moving Time: {total_moving_time/60:.1f} minutes")
    print(f"Total Distance: {total_distance/1000:.2f} km")
    print(f"Total Elevation Gain: {total_elevation_gain:.1f} m")
    
    if "power" in df.columns:
        print(f"Average Power (while moving): {avg_power:.1f} W")
        print(f"Normalized Power: {overall_np:.1f} W")
        print(f"Maximum Power: {max_power:.1f} W")
        print(f"TSS: {overall_tss:.1f}")
    
    if "heart_rate" in df.columns:
        print(f"Average Heart Rate (while moving): {avg_hr:.1f} bpm")
        print(f"Maximum Heart Rate: {max_hr:.1f} bpm")
    
    if total_calories > 0:
        print(f"Total Calories: {total_calories} kcal")
    
    # Create visualization
    plt.figure(figsize=(12, 8))
    
    # Plot available metrics
    if "power" in df.columns:
        plt.plot(df["timestamp"], df["power"], label="Power (W)", color="blue")
    
    if "heart_rate" in df.columns:
        plt.plot(df["timestamp"], df["heart_rate"], label="Heart Rate (bpm)", 
                color="red", alpha=0.7)
    
    if "speed" in df.columns:
        speed_kmh = df["speed"] * 3.6
        plt.plot(df["timestamp"], speed_kmh, label="Speed (km/h)", 
                color="green", alpha=0.7)
    
    plt.title(f"Merged Ride Analysis - {date_str}")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    
    # Add vertical lines between original ride segments
    ride_starts = [df["timestamp"].iloc[0]]
    for i in range(1, len(df)):
        if (df["timestamp"].iloc[i] - df["timestamp"].iloc[i-1]).total_seconds() > 60:
            ride_starts.append(df["timestamp"].iloc[i])
            plt.axvline(x=df["timestamp"].iloc[i], color='gray', 
                       linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(png_path)
    plt.close()
    
    # Save processed data
    df.to_csv(csv_path, index=False)
    
    print(f"\n[OK] Saved merged data to {csv_path}")
    log_user_action('File created', {'type': 'csv', 'path': csv_path, 'source': 'merge'})
    print(f"[OK] Saved merged plot to {png_path}")
    log_user_action('File created', {'type': 'png', 'path': png_path, 'source': 'merge'})

def log_user_action(action: str, params: dict = None):
    """Logs user actions with timestamp to logs/logs.json, rotating every 10 days."""
    logs_dir = os.path.join(os.path.dirname(__file__), 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    now = datetime.now()
    log_file = os.path.join(logs_dir, 'logs.json')

    # Determine if log rotation is needed
    rotate = False
    if os.path.exists(log_file):
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                first_line = f.readline()
                if first_line:
                    first_entry = json.loads(first_line)
                    first_time = datetime.fromisoformat(first_entry['timestamp'])
                    if (now - first_time).days >= 10:
                        rotate = True
        except Exception:
            rotate = True  # If log is corrupt, rotate
    if rotate:
        # Archive old log file
        archive_name = os.path.join(logs_dir, f"logs_{now.strftime('%Y%m%d_%H%M%S')}.json")
        os.rename(log_file, archive_name)

    # Prepare log entry
    entry = {
        'timestamp': now.isoformat(),
        'action': action,
        'params': params or {}
    }
    try:
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(entry) + '\n')
    except Exception as e:
        print(f"[WARNING] Failed to write log: {e}")

def main() -> None:
    """Main program loop for the cycling workout analysis tool.

    Provides an interactive command-line interface for analyzing cycling
    workout data from FIT files. The program runs in a continuous loop
    until the user chooses to exit.

    Menu Options:
        0. Exit Program
        1. Process individual FIT files
        2. Get a weekly summary
        3. Process all FIT files
        4. Create overview CSV
        5. Compare two rides
        6. Update rider statistics
        7. Analyze gym session
        8. Merge rides from same day
    """
    # Load saved rider stats at startup
    stats = load_rider_stats()
    if stats:
        global FTP, HR_MAX, HR_REST, RIDER_WEIGHT
        FTP = stats.get('ftp', FTP)
        HR_MAX = stats.get('hr_max', HR_MAX)
        HR_REST = stats.get('hr_rest', HR_REST)
        RIDER_WEIGHT = stats.get('weight', RIDER_WEIGHT)

    while True:
        print("\nTraining Data Menu:")
        print("0. Exit Program")
        print("1. Analyze a single ride")
        print("2. Get a weekly summary (select 7 .fit files)")
        print("3. Process all .fit files in your folder")
        print("4. Create overview CSV of all rides")
        print("5. Compare two rides")
        print("6. Update rider statistics")
        print("7. Analyze gym session")
        print("8. Merge rides from same day")

        choice = input("\nEnter your choice (0-8): ").strip()
        log_user_action('Menu selection', {'choice': choice})

        if choice == "0":
            log_user_action('Exit program')
            print("\nExiting program. Goodbye!")
            break
        elif choice == "1":
            print("\nAvailable .fit files:")
            fit_files = sorted(glob.glob(os.path.join(FIT_FOLDER, "*.fit")))
            if not fit_files:
                print("[ERROR] No .fit files found in the fit-files folder")
                continue
                
            for i, file in enumerate(fit_files, 1):
                print(f"{i}. {os.path.basename(file)}")
            
            try:
                idx = int(input("\nSelect ride number: ")) - 1
                if 0 <= idx < len(fit_files):
                    process_fit_file(os.path.basename(fit_files[idx]), show_plots=True)
                    log_user_action('Analyze single ride', {'file': fit_files[idx]})
                else:
                    print("[ERROR] Invalid selection")
            except (ValueError, IndexError):
                print("[ERROR] Invalid selection")
        elif choice == "2":
            log_user_action('Weekly summary', {'files': sorted(glob.glob(os.path.join(FIT_FOLDER, "*.fit")), reverse=True)[:7]})
            print("\nProcessing last 7 .fit files...")
            files = sorted(glob.glob(os.path.join(FIT_FOLDER, "*.fit")), reverse=True)[:7]
            weekly_summary(files)
        elif choice == "3":
            log_user_action('Process all .fit files', {'files': sorted(glob.glob(os.path.join(FIT_FOLDER, "*.fit")))})
            print("\nProcessing all .fit files...")
            all_files = sorted(glob.glob(os.path.join(FIT_FOLDER, "*.fit")))
            weekly_summary(all_files)
        elif choice == "4":
            log_user_action('Create overview CSV')
            create_overview_csv()
        elif choice == "5":
            log_user_action('Compare two rides', {'file1': fit_files[idx1] if 'idx1' in locals() and 0 <= idx1 < len(fit_files) else None, 'file2': fit_files[idx2] if 'idx2' in locals() and 0 <= idx2 < len(fit_files) else None})
            print("\nAvailable .fit files:")
            fit_files = sorted(glob.glob(os.path.join(FIT_FOLDER, "*.fit")))
            if not fit_files:
                print("[ERROR] No .fit files found in the fit-files folder")
                continue
                
            for i, file in enumerate(fit_files, 1):
                print(f"{i}. {os.path.basename(file)}")
            
            try:
                idx1 = int(input("\nSelect first ride number: ")) - 1
                idx2 = int(input("Select second ride number: ")) - 1
                if 0 <= idx1 < len(fit_files) and 0 <= idx2 < len(fit_files):
                    compare_rides(os.path.basename(fit_files[idx1]), os.path.basename(fit_files[idx2]))
                else:
                    print("[ERROR] Invalid selection")
            except (ValueError, IndexError):
                print("[ERROR] Invalid selection")
        elif choice == "6":
            log_user_action('Update rider statistics')
            update_rider_stats()
        elif choice == "7":
            log_user_action('Analyze gym session', {'file': fit_files[idx] if 'idx' in locals() and 0 <= idx < len(fit_files) else None})
            print("\nAvailable .fit files:")
            fit_files = sorted(glob.glob(os.path.join(FIT_FOLDER, "*.fit")))
            if not fit_files:
                print("[ERROR] No .fit files found in the fit-files folder")
                continue
                
            for i, file in enumerate(fit_files, 1):
                print(f"{i}. {os.path.basename(file)}")
            
            try:
                idx = int(input("\nSelect gym session number: ")) - 1
                if 0 <= idx < len(fit_files):
                    analyze_gym_session(os.path.basename(fit_files[idx]), show_plots=True)
                else:
                    print("[ERROR] Invalid selection")
            except (ValueError, IndexError):
                print("[ERROR] Invalid selection")
        elif choice == "8":
            log_user_action('Merge rides', {'files': selected_files if 'selected_files' in locals() else None})
            print("\nAvailable .fit files:")
            fit_files = sorted(glob.glob(os.path.join(FIT_FOLDER, "*.fit")))
            if not fit_files:
                print("[ERROR] No .fit files found in the fit-files folder")
                continue
                
            for i, file in enumerate(fit_files, 1):
                print(f"{i}. {os.path.basename(file)}")
            
            try:
                print("\nEnter the numbers of the rides to merge (comma-separated):")
                print("Example: 1,2,3")
                selections = input("Ride numbers: ").strip()
                indices = [int(x.strip()) - 1 for x in selections.split(",")]
                
                # Validate all indices
                if all(0 <= idx < len(fit_files) for idx in indices):
                    selected_files = [os.path.basename(fit_files[idx]) for idx in indices]
                    merge_rides(selected_files)
                else:
                    print("[ERROR] Invalid selection")
            except (ValueError, IndexError):
                print("[ERROR] Invalid selection")
        else:
            log_user_action('Invalid menu selection', {'input': choice})
            print("\n[ERROR] Invalid choice. Please enter a number between 0 and 8.")

        input("\nPress Enter to continue...")

if __name__ == "__main__":
    main()
