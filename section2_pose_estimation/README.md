# 🧊 3D Pose Estimation with DeepLabCut

This section demonstrates how to use DeepLabCut for **3D pose estimation** from two synchronized camera videos.

The notebook covers the full 3D workflow, including:
- Creating a 3D DeepLabCut project
- Calibrating cameras using checkerboard images
- Verifying calibration through undistortion
- Triangulating 2D keypoints into 3D coordinates
- Visualizing labeled 3D poses

This pipeline enables accurate tracking of body part movements in space—ideal for behavior analysis and neuroscience experiments.

---

## 📖 Contents

- [`02_3D_pose_estimation_with_deeplabcut.ipynb`](02_3D_pose_estimation_with_deeplabcut.ipynb):  
  Step-by-step notebook for the entire 3D pose estimation process.

---

## 🚀 Quick Start

1. Open the notebook in Jupyter Notebook or Jupyter Lab.
2. Follow each section step by step.
3. Make sure you have synchronized videos from at least two camera angles, and checkerboard images for calibration.

---

## 🗂️ Main Steps

- **Step 1:** Create a 3D DeepLabCut project
- **Step 2:** Capture and process calibration images
    - Option A: Capture synchronized images
    - Option B: Extract frames from videos
- **Step 3:** Calibrate cameras and verify undistortion
- **Step 4:** Triangulate 2D keypoints to obtain 3D coordinates
- **Step 5:** Visualize and analyze results

---

## 💻 Requirements

- Python 3.8+
- [DeepLabCut](https://www.deeplabcut.org/)
- OpenCV
- NumPy
- (See the notebook for full package list and installation instructions.)

---

## 📝 Notes

- All code explanations and detailed instructions are included in the notebook cells.
- Example data structure, troubleshooting tips, and visualization options are provided.
- If you use your own videos/data, adjust the paths and parameters as explained in the notebook.

---

## 📜 License

[Add license information or terms of use, if applicable.]