
# Mouse Hand Movement Detection – Computer Vision Project

**Technion Winter Semester 2024/2025**  
Student: Sharbel Mazzawi  
Mentor: Dr. Sivan Schwartz  

## Overview

This project implements a full pipeline for analyzing mouse right-hand movements in 3D space using a combination of DeepLabCut and custom-built Python algorithms. While DeepLabCut is used to reconstruct 3D coordinates from labeled 2D videos, the actual detection algorithm operates on the 2D coordinates obtained from Camera 1. This decision was made to ensure consistency with the ground truth annotations, which were created based on the Camera 1 video.

The 3D coordinates are only used at the visualization stage, where the system extracts frames corresponding to detected movements and maps them into 3D space for plotting.

---

## Key Features

-Frame extraction and interpolation for missing values.
-2D-based motion detection aligned with camera-ground-truth perspective.
-3D pose estimation used only for visualization.
-Right/Left hand and food movement detection using positional thresholds.
-Custom movement detection logic with trial segmentation.
-Ground truth comparison and accuracy report.
-3D and 2D trajectory visualization.
-Movement-only and full video generation.
---

## Project Structure

```
project/
│
├── src/                               # All core Python scripts
│   ├── Behavioral_analysis_runner.py      # Main entry script
│   ├── config.json                        # Configuration file
│   ├── data_processor_Class_final.py      # Preprocessing 2D DLC data
│   ├── movement_analysis_final.py         # Hand movement detection logic
│   ├── comparsion_final.py                # Ground truth comparison logic
│   ├── plotting_final.py                  # 3D visualization of movements
│
├── camera_1_data.csv                  # Original 2D coordinates from DeepLabCut
├── mouse_video_DLC_3D.csv             # 3D coordinates from DeepLabCut
├── ground_truth.csv                   # Manually labeled ground truth for comparison          
```

---

## How It Works

### 1. Data Preprocessing
Using `DataProcessor`:
- Filters low-confidence keypoints using a likelihood cutoff.
- Applies linear interpolation to fill missing values.

### 2. Movement Detection
Using `MovementAnalyzer`:
- Calculates movement flags based on thresholded distance from reference points.
- Resolves overlaps using confidence (likelihood).
- Filters and merges close or small movements into meaningful trials.
- Classifies valid vs. non-valid trials based on food presence.

### 3. Ground Truth Comparison
Using `CSVComparator`:
- Compares detected results to manually annotated `ground_truth.csv`.
- Outputs mismatched frames and calculates accuracy metrics.

### 4. 3D Visualization
Using `Plotting`:
- Generates 3D or 2D plots for individual and average trajectories per trial.
- Supports both valid and invalid movement plotting.
-Choice is defined in config.json via the plotting_type parameter.

---

## Video Generation

- `generate_movement_video`: Creates a video with only valid movement frames.
- `generate_full_video`: Annotates each frame with frame & trial number.

---

## Requirements

- Python 3.8+
- OpenCV
- pandas
- numpy
- matplotlib
- tqdm
- scipy
- openpyxl

---

## Running the Pipeline

```bash
python Behavioral_analysis_runner.py
```

This script runs the full pipeline: preprocessing, detection, comparison, and visualization.

---

## Evaluation

- **General Accuracy**: 81.59%  
- **Trial-Specific Accuracy**: 97.46%  
- Discrepancies often result from human annotation delays (~±5 frames).

---

## Configuration (`config.json`)

This file allows the user to fully customize the pipeline. Here are the sections and parameters:

### `data_process`
| Parameter | Description |
|----------|-------------|
| `file_name` | Path to the 2D CSV file from Camera 1 used for processing. |
| `coordinates` | Number of preceding coordinate columns to set as NaN if likelihood is below cutoff (usually 2 for x,y). |
| `pcutoff` | Likelihood threshold below which points are considered unreliable and removed. |

### `movement_analysis`
| Parameter | Description |
|----------|-------------|
| `file_path` | Path to the interpolated CSV (output of DataProcessor). |
| `video_path` | Path to the video you of the mouse or.. (the main video) |
| `column_groups` | Dict mapping body parts to their column indices (x, y, likelihood). |
| `thresholds` | Threshold for positional movement beyond reference point to classify as movement. |
| `data_start_row` | Index of the row where actual data starts (after metadata rows). |
| `frames_folder` | Folder where video frames are saved/extracted. |
| `min_length` | Minimum length (in frames) of movement to be considered valid. |
| `padd_trial` | Number of frames to pad before and after detected movement. |
| `image_path` | If not 0, allows user to manually select reference points from image. |
| `reference_idx` | Row index to use as baseline (no movement) when detecting shifts. |

### `comparison`
| Parameter | Description |
|----------|-------------|
| `results_path` | Path to the CSV of results (output of movement_analysis). |
| `ground_truth_path` | Path to the CSV of manual labeling (a file the user build by his sight). |

### `plotting`
| Parameter | Description |
|----------|-------------|
| `valid_path` | Path to the CSV of valid trials. |
| `non_valid_path` | Path to the CSV of invalid trials. |
| `data_3D_file` | Path to the 3D DeepLabCut output CSV file. Used only for plotting. |
| `file` | Choose "valid" or "non_valid" to plot valid or non-valid trials. |
| `trial_num` | Specify a trial number to plot a single trial, or "all" for all the trials at once with average trajectory,
                and each trial in a single plot, one by one. |
| `plotting_type` | Choose "3D" or "2D" to toggle between dimensional plotting. |

---

## Output
- CSV of data after interpolation
- CSV of results (the hand movement of the mouse)
- Csv sheets of valid, non valid movement.
- Visual 3D or 2D trajectory plots with labeled start/end points
- Frame-by-frame annotated movement videos
- Excel sheet with mismatches vs ground truth

---

## License

This project is part of an academic course and is for educational use only.
