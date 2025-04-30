# Mouse Hand Movement Detection: From Annotation to 3D Analysis

**Technion Winter Semester 2024/2025**  
Student: Sharbel Mazzawi  
Mentor: Dr. Sivan Schwartz

---

## Project Overview

This repository presents a complete pipeline for analyzing and detecting right-hand movements in mice using computer vision and deep learning tools. The workflow moves from initial data annotation with DeepLabCut (DLC), through 3D pose estimation and motion detection, to custom analysis and visualization in Python.

The project is divided into three main sections, each with its own notebook (`.ipynb`) and README, guiding the user step-by-step through the full process.

---

## Repository Structure
```
/
├── section1_deeplabcut_gui_napari/
│   ├── 01_Deeplabcut_Gui_Napari.ipynb
│   ├── camera_config.yaml
│   └── README.md
├── section2_3d_pose_estimation/
│   ├── 02_3D_pose_estimation_with_deeplabcut.ipynb
│   ├── pose_estimation.ipynb
│   ├── 3D_config.yaml
│   ├── trial0001.mp4
│   ├── camera-2-01_corner.jpg
│   ├── camera-1-01_corner.jpg
│   └── README.md
├── section3_movement_analysis/
│   ├── camera_1_data.csv
│   ├── mouse_video_DLC_3D.csv
│   ├── ground_truth.csv
│   ├── README.md
│   └── codes
│       ├──Behavioral_analysis_runner.py
│       ├──data_processor_Class_final.py
│       ├──movement_analysis_final.py
│       ├──comparsion_final.py
│       ├──plotting_final.py
│       └── config.json
├── LICENSE
├── .gitignore
└── README.md
```
- **Section 1**: Data Annotation with DeepLabCut and Napari GUI  
  (Preparation and manual labeling of frames for training the pose estimation model.)

- **Section 2**: 3D Pose Estimation with DeepLabCut  
  (Multi-camera calibration, 3D triangulation, and exporting pose data.)

- **Section 3**: Movement Analysis and Visualization  
  (Custom Python pipeline for preprocessing, hand movement detection, comparison to ground truth, and 3D/2D visualization.)

---
- **Outputs:**  
  - CSV files of processed and analyzed data
  - 3D/2D trajectory plots
  - Annotated videos of detected hand movements
  - Accuracy evaluation vs. ground truth

- **Requirements:**  
  - Python 3.8+
  - See `requirements.txt` for a full list

---

## License

This project is for educational use as part of the Technion course.

---

## Contact

For questions, feedback, or collaborations, please contact:
- Sharbel Mazzawi (student): [sharbel.ma2@gmail.com]
- Dr. Sivan Schwartz (mentor): [sivan1schwartz@gmail.com]

---

## Acknowledgements

Special thanks to the Technion and the developers of DeepLabCut for providing the tools and resources for this project.

---
