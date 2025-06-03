# Function Documentation for analyze_fit.py

This document provides detailed descriptions of all functions in `analyze_fit.py`, including their purpose, implementation details, and calculations.

## Configuration Constants

- `FTP`: Functional Threshold Power set to 230 watts
- `HR_MAX`: Maximum heart rate set to 179 bpm
- `HR_REST`: Resting heart rate set to 58 bpm
- `RIDER_WEIGHT`: Rider's weight in kg (default 75 kg)
- Directory paths are configured for fit files, CSV outputs, and PNG plots

## Power Zones

### `get_power_zone(power)`

Calculates the power zone based on percentage of FTP (Functional Threshold Power).

**Zones:**
- Zone 1 (Active Recovery): < 55% of FTP
- Zone 2 (Endurance): 55-75% of FTP
- Zone 3 (Tempo): 75-90% of FTP
- Zone 4 (Threshold): 90-105% of FTP
- Zone 5 (VO2 Max): 105-120% of FTP
- Zone 6 (Anaerobic): 120-150% of FTP
- Zone 7 (Neuromuscular): > 150% of FTP

## Heart Rate Zones

### `get_hr_zone(hr)`

Calculates heart rate zones based on Heart Rate Reserve (HRR) method:
- HRR = Maximum HR - Resting HR
- Zone percentage = (Current HR - Resting HR) / HRR

**Zones:**
- Below Zone 1: < 50% of HRR
- Zone 1 (Recovery): 50-60% of HRR
- Zone 2 (Endurance): 60-70% of HRR
- Zone 3 (Tempo): 70-80% of HRR
- Zone 4 (Threshold): 80-90% of HRR
- Zone 5 (VO2 Max): > 90% of HRR

## Climb Analysis

### `analyze_climbs(df)`

Analyzes climbs in the ride using Garmin's climb detection criteria:

**Parameters:**
- Minimum horizontal distance: 300m
- Minimum elevation gain: 10m
- Minimum average gradient: 3%
- Maximum cumulative descent within climb: 5m
- Flat section threshold: 1% gradient

**Implementation:**
1. Calculates gradient from distance and altitude changes
2. Smooths gradient using 10-point moving average
3. Detects climbs by:
   - Identifying sections with gradient ≥ 3%
   - Allowing small descents (up to 5m cumulative)
   - Handling flat sections within climbs
4. For each climb, calculates:
   - Distance
   - Elevation gain
   - Average gradient
   - Maximum gradient
   - Start distance
   - Duration
   - Power metrics (if available)
   - Heart rate metrics (if available)

## Power Calculations

### `calculate_normalized_power(power_series)`

Calculates Normalized Power (NP) using the standard algorithm defined by TrainingPeaks.

**Implementation:**
```python
def calculate_normalized_power(power_series):
    rolling_avg = power_series.rolling(window=30, min_periods=1).mean()
    fourth_power_avg = (rolling_avg ** 4).mean()
    return fourth_power_avg ** 0.25
```

**Algorithm Steps:**
1. Calculate 30-second rolling average of power
2. Raise each value to the fourth power
3. Take the average of all values
4. Take the fourth root of the average

**Parameters:**
- `power_series`: pandas Series containing power values in watts

**Returns:**
- Normalized Power value in watts

**Usage:**
- Used in ride analysis for:
  - TSS calculations
  - Intensity Factor calculations
  - Ride comparisons
  - Training load assessment

**Significance:**
- Better represents physiological load than average power
- Accounts for variability in power output
- Key metric for training stress calculation
- Used in combination with FTP for IF and TSS

### `calculate_power_to_weight(power, weight)`

Calculates power-to-weight ratio, a critical metric for comparing performance across different rider weights.

**Implementation:**
```python
def calculate_power_to_weight(power, weight):
    if power is None or weight is None or weight == 0:
        return 0.0
    return power / weight
```

**Parameters:**
- `power`: Power value in watts
- `weight`: Rider weight in kilograms

**Returns:**
- Power-to-weight ratio in watts/kg

**Usage in Analysis:**
- Applied to multiple power metrics:
  - Average Power/kg
  - Maximum Power/kg
  - Normalized Power/kg
  - FTP/kg
  - Real-time power/kg in data processing

**Significance:**
- Key indicator of climbing ability
- Allows performance comparison between riders of different weights
- Useful for tracking fitness improvements while accounting for weight changes
- Typical values for amateur cyclists:
  - 2-3 W/kg: Beginner
  - 3-4 W/kg: Intermediate
  - 4-5 W/kg: Advanced
  - >5 W/kg: Elite

**CSV Export Integration:**
- All power-to-weight metrics are included in the overview CSV:
  - `avg_power_per_kg`
  - `max_power_per_kg`
  - `np_per_kg`
  - Current rider weight at time of analysis

## File Processing

### `process_fit_file(fit_filename, show_plots=False)`

Main function for processing individual .fit files.

**Functionality:**
1. Reads FIT file data
2. Creates DataFrame with available metrics:
   - timestamp
   - power
   - heart rate
   - cadence
   - speed
   - altitude
   - distance

**Calculations:**
1. **Basic Metrics:**
   - Duration
   - Average/Max Power
   - Average/Max Power per kg
   - Average/Max Heart Rate
   - Average/Normalized Speed
   - Total Distance
   - Elevation Gain

2. **Advanced Power Metrics:**
   - Normalized Power (NP)
   - Normalized Power per kg
   - Intensity Factor (IF)
   - Training Stress Score (TSS)

3. **Power-to-Weight Analysis:**
   - Real-time power/kg calculations
   - Average power/kg for the ride
   - Maximum power/kg achieved
   - Normalized power/kg
   - FTP/kg reference value

4. **Visualization:**
   - Creates time-series plot of:
     - Power (blue line)
     - Heart Rate (red line)
     - Speed (green line)
   - Highlights climbs (yellow background)
   - Saves plot as PNG

5. **Data Export:**
   - Saves processed data to CSV
   - Includes all metrics and calculated zones

## Summary Functions

### `weekly_summary(file_list)`

Processes multiple FIT files to create a weekly training summary:
- Iterates through provided files
- Calls `process_fit_file` for each
- Displays individual ride summaries

### `create_overview_csv()`

Creates a comprehensive overview of all training data with the following metrics:

**Included Fields:**
- Filename and Date
- Duration
- Power Metrics:
  - Average Power (W)
  - Maximum Power (W)
  - Average Power/kg
  - Maximum Power/kg
  - Normalized Power (W)
  - Normalized Power/kg
  - Intensity Factor
  - Training Stress Score
- Heart Rate Metrics:
  - Average HR
  - Maximum HR
- Speed Metrics:
  - Average Speed
  - Maximum Speed
  - Normalized Speed
- Distance and Elevation:
  - Total Distance
  - Elevation Gain
- Other:
  - Calories
  - Rider Weight

This comprehensive export allows for:
- Long-term tracking of power-to-weight improvements
- Analysis of performance trends
- Weight change impact assessment
- Training load monitoring
- Fitness progression tracking

## Main Function

### `main()`

Provides an interactive menu interface for accessing all analysis features.

**Menu Options:**
1. **Analyze a single ride**
   - Displays numbered list of all available .fit files
   - User selects file by entering its number
   - Processes selected file with plots enabled
   - Shows detailed metrics and analysis

2. **Get a weekly summary**
   - Processes last 7 .fit files
   - Shows summary for each ride
   - Files sorted by date (newest first)

3. **Process all FIT files**
   - Analyzes all files in the fit-files folder
   - Shows summary for each ride

4. **Create overview CSV**
   - Generates comprehensive CSV with all ride data
   - Includes all available metrics
   - Useful for trend analysis

5. **Compare two rides**
   - Displays numbered list of available .fit files
   - User selects two files by number
   - Shows side-by-side comparison
   - Generates comparison plots

6. **Update rider statistics**
   - Interactive update of rider parameters
   - Validates all inputs
   - Saves to configuration file

**File Selection Interface:**
- Shows numbered list of all .fit files in directory
- Files are sorted alphabetically (by date if using date-based filenames)
- User enters the number corresponding to desired file
- Includes error handling for:
  - Invalid number entries
  - Non-numeric inputs
  - Empty directories
  - Out-of-range selections

**Implementation Details:**
```python
# Example of file selection interface
print("\nAvailable .fit files:")
fit_files = sorted(glob.glob(os.path.join(FIT_FOLDER, "*.fit")))
for i, file in enumerate(fit_files, 1):
    print(f"{i}. {os.path.basename(file)}")
idx = int(input("\nSelect ride number: ")) - 1
```

**Usage:**
1. Run the script
2. Select desired operation from menu
3. For file operations:
   - View numbered list of available files
   - Enter the number of desired file
   - View results or additional prompts
4. For configuration:
   - Enter new values or press Enter to keep current
   - Review saved configuration

## Speed Calculations

### `calculate_normalized_speed(speed_series)`

Calculates normalized average speed by excluding stopped time:

**Implementation:**
1. Converts speed from m/s to km/h
2. Filters out all zero values (stopped time)
3. Calculates mean of remaining values

This provides a more accurate representation of actual moving speed during the ride, as it excludes time spent stopped at traffic lights, rest stops, etc.

**Comparison with Regular Average:**
- Regular Average Speed: Includes all time periods, including stops
- Normalized Average Speed: Only includes time periods where the bike is moving

For example, if you ride for 1 hour at 30 km/h but stop for 20 minutes at traffic lights:
- Regular average would be 24 km/h (30 km covered in 1.33 hours)
- Normalized average would be 30 km/h (actual moving speed)

**Calculations:**
1. **Basic Metrics:**
   - Duration
   - Average/Max Power
   - Average/Max Heart Rate
   - Regular Average Speed (including stops)
   - Normalized Average Speed (excluding stops)
   - Total Distance
   - Elevation Gain

2. **Advanced Power Metrics:**
   - Normalized Power (NP):
     - 30-second rolling average of power
     - Fourth power average calculation
     - Fourth root of the average
   - Intensity Factor (IF):
     - NP / FTP
   - Training Stress Score (TSS):
     - (duration_seconds × NP × IF) / (FTP × 3600) × 100

3. **Visualization:**
   - Creates time-series plot of:
     - Power (blue line)
     - Heart Rate (red line)
     - Speed (green line)
   - Highlights climbs (yellow background)
   - Saves plot as PNG

4. **Data Export:**
   - Saves processed data to CSV
   - Includes all metrics and calculated zones

## Ride Comparison

### `compare_rides(fit_file1, fit_file2)`

Provides detailed comparison analysis between two rides, generating both visual and numerical comparisons of key metrics.

**Basic Metrics Compared:**
1. **Power Metrics:**
   - Average Power and Power/kg
   - Normalized Power and NP/kg
   - Time in each power zone
   - Maximum Power and Power/kg
   - Power distribution over distance

2. **Speed Metrics:**
   - Average Speed
   - Maximum Speed
   - Normalized Speed
   - Speed variations over distance

**Advanced Training Metrics:**

1. **Efficiency Factor (EF):**
   - Calculated as: Normalized Power / Average Heart Rate
   - Higher values indicate better aerobic fitness
   - Useful for comparing aerobic efficiency between rides
   - Can track improvements in fitness over time

2. **Power:HR Decoupling:**
   - Measures cardiovascular drift during long rides
   - Compares first half vs second half of ride
   - Calculation: Percentage change in Power:HR ratio
   - Lower values indicate better endurance
   - Target < 5% for well-executed endurance rides

3. **Training Stress Score (TSS):**
   - Quantifies overall training load
   - Combines intensity (IF) and duration
   - Useful for comparing workout stress between rides
   - Calculation: (duration_hours × NP × IF) / (FTP × 3600) × 100

4. **Power-to-Weight Comparisons:**
   - Compares relative power output accounting for any weight changes
   - Helps identify improvements in power-to-weight ratio
   - Particularly relevant for climbing performance analysis
   - Shows impact of weight changes on performance

**Implementation Details:**
- Resamples data to equal distance intervals (10m) for fair comparison
- Uses 30-second rolling averages for power smoothing
- Handles missing data gracefully
- Generates combined plots for visual analysis
- Saves comparison plots automatically
- Provides detailed text output of all metrics

**Usage:**
1. Select option 5 from the main menu
2. Choose two rides from the displayed list
3. Review the generated comparison metrics and plots
4. Find the comparison plot in the png-files directory

This feature is particularly useful for:
- Tracking improvements over time on the same route
- Comparing different pacing strategies
- Analyzing the effects of fatigue
- Comparing training intensity between different types of rides
- Identifying changes in fitness level

## Rider Statistics Management

### `update_rider_stats()`

Provides an interactive interface to update rider-specific statistics used in calculations.

**Configurable Parameters:**
1. **Weight**
   - Valid range: 40-150 kg
   - Used for power-to-weight calculations
   - Critical for performance tracking
   - Affects all power/kg metrics
   - Stored in configuration file

2. **FTP (Functional Threshold Power)**
   - Valid range: 100-500 watts
   - Used for power zone calculations
   - Critical for TSS and IF
   - FTP/kg calculated and displayed

3. **Maximum Heart Rate**
   - Valid range: 120-220 bpm
   - Used for heart rate zone calculations

4. **Resting Heart Rate**
   - Valid range: 30-100 bpm
   - Must be lower than Maximum Heart Rate
   - Used in Heart Rate Reserve calculations

**Implementation Details:**
- Values are validated to ensure they fall within physiologically reasonable ranges
- Empty input preserves current values
- Configuration is saved to `rider_config.txt`
- Values persist between script executions
- Automatic loading of saved values at startup
- Weight changes affect all power-to-weight calculations

### `load_rider_stats()`

Loads previously saved rider statistics from the configuration file.

**Configuration File Format:**
```
FTP=230
HR_MAX=179
HR_REST=58
WEIGHT=75
```

**Features:**
- Automatically called at script startup
- Gracefully handles missing or corrupt configuration files
- Maintains default values if no configuration exists
- Provides feedback on successful load or any issues
- Loads weight for power-to-weight calculations

**Usage Impact:**
These values affect various calculations throughout the analysis:
- Power Zone distribution
- Heart Rate Zone calculations
- Training Stress Score (TSS)
- Intensity Factor (IF)
- Heart Rate Reserve (HRR) zones
- All power-to-weight metrics 