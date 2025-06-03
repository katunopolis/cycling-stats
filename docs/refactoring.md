# Code Refactoring Plan

## Current Issues
- Single script (`analyze_fit.py`) with over 1000 lines
- Mixed responsibilities (UI, data processing, analysis, visualization)
- Difficult to test individual components
- Limited code reusability
- Complex maintenance and updates

## Proposed Package Structure
```
cycling_analytics/
├── src/
│   └── cycling_analytics/
│       ├── __init__.py
│       ├── main.py              # Main script with menu and program flow
│       ├── data_processing/     # Data processing modules
│       │   ├── __init__.py
│       │   ├── fit_parser.py    # FIT file parsing logic
│       │   └── metrics.py       # Metrics calculation
│       ├── analysis/           # Analysis modules
│       │   ├── __init__.py
│       │   ├── power.py        # Power zone calculations
│       │   ├── heart_rate.py   # Heart rate zone calculations
│       │   └── ride_stats.py   # General ride statistics
│       ├── visualization/      # Visualization modules
│       │   ├── __init__.py
│       │   └── plotting.py     # All plotting functions
│       └── utils/             # Utility modules
│           ├── __init__.py
│           ├── config.py      # Configuration handling
│           └── file_ops.py    # File operations
├── tests/                    # Test directory
├── docs/                     # Documentation
└── setup.py                 # Package setup file
```

## Module Responsibilities

### Main Module (`main.py`)
- Command-line interface
- Menu handling
- User interaction
- High-level program flow
- Coordinating other modules

### Data Processing
#### `fit_parser.py`
- FIT file reading and parsing
- Data extraction
- Basic data validation
- Converting to pandas DataFrame

#### `metrics.py`
- Raw data processing
- Metric calculations
- Data cleaning and validation

### Analysis
#### `power.py`
- Power zone calculations
- Power-based metrics (NP, IF, TSS)
- Power curve analysis

#### `heart_rate.py`
- Heart rate zone analysis
- HR-based metrics
- Recovery calculations

#### `ride_stats.py`
- General ride statistics
- Speed and distance calculations
- Elevation analysis

### Visualization (`plotting.py`)
- Ride data plotting
- Zone distribution charts
- Performance graphs
- Time series visualizations

### Utilities
#### `config.py`
- Configuration file handling
- User settings management
- Constants and defaults

#### `file_ops.py`
- File system operations
- CSV handling
- Directory management

## Refactoring Steps

1. **Initial Setup**
   - Create package structure
   - Set up `__init__.py` files
   - Create empty module files

2. **Code Migration**
   - Identify related functions
   - Move code to appropriate modules
   - Update imports and dependencies
   - Maintain functionality during migration

3. **Interface Updates**
   - Create proper class interfaces
   - Define clear public APIs
   - Add type hints
   - Update function signatures

4. **Testing**
   - Create test directory structure
   - Write unit tests for each module
   - Add integration tests
   - Ensure test coverage

5. **Documentation**
   - Add docstrings to all modules
   - Create module-level documentation
   - Update user guides
   - Add examples

## Benefits of Refactoring

1. **Maintainability**
   - Smaller, focused modules
   - Clear separation of concerns
   - Easier to understand and modify
   - Better error isolation

2. **Testability**
   - Unit testing of individual components
   - Mocking of dependencies
   - Better test coverage
   - Easier debugging

3. **Reusability**
   - Modular components
   - Clear interfaces
   - Easy to extend
   - Potential for external use

4. **Code Quality**
   - Better organization
   - Consistent style
   - Type safety
   - Documentation

## Quality Metrics

The following tools will help maintain code quality:

- **Flake8**: Style guide enforcement (max complexity: 10)
- **Radon**: Code complexity measurement
- **MyPy**: Type checking
- **Pylint**: Code analysis
- **pytest**: Test coverage

## Next Steps

1. Create GitHub repository
2. Set up development environment
3. Create package structure
4. Begin incremental migration
5. Add tests as we go
6. Review and refine 