# Tournament Visualization Tool

This project provides tools for processing and visualizing tournament data, specifically designed for quiz tournaments with multiple tours and questions.

## Project Structure

- `iokovp1.py` - Data transformation script
- `iokoviz.py` - Visualization script
- `tournament_transformed.csv` - Processed tournament data

## Features

### Data Transformation (`iokovp1.py`)
- Converts tournament data from tour-based Excel format to a consolidated CSV
- Handles multiple tours and questions per tour
- Preserves team information (ID, name, city)
- Processes answers for all questions across all tours

### Visualization (`iokoviz.py`)
- Creates animated visualization of tournament progress
- Shows real-time score progression for all teams
- Highlights top 20 teams with different colors
- Displays both tour scores and total scores
- Features dynamic axis scaling and tour separators
- Includes team name abbreviations for better readability

## Configuration

The visualization can be configured using these variables at the start of `iokoviz.py`:
```python
QUESTIONS_PER_TOUR = 12  # Number of questions in each tour
NUMBER_OF_TOURS = 9     # Total number of tours
INPUT_FILENAME = 'tournament_transformed.csv'  # Input file name
```

## Usage

1. Data Transformation:
```python
python iokovp1.py
```
This will create a transformed CSV file from your Excel tournament data.

2. Visualization:
```python
python iokoviz.py
```
This will create an animated GIF showing the tournament progress.

## Input Data Format

### Excel Input Format (for iokovp1.py):
- Required columns: 'Номер команды', 'Название', 'Город', 'Тур'
- Question answers should start from column 5
- Each row represents one team's results in one tour

### CSV Output Format:
- Columns: 'Номер команды', 'Название', 'Город', 'Q1' through 'Q108'
- Each row represents one team's complete tournament results

## Output

- The transformation script produces a CSV file with consolidated data
- The visualization script produces an animated GIF showing the tournament progress
- Output filename is automatically generated based on input filename (adds '_animated.gif' suffix)

## Technical Details

- Supports up to 20 teams in the visualization
- Uses dynamic scaling for better readability
- Includes pauses between tours in the animation
- Automatically abbreviates long team names
- Shows both per-tour and total scores