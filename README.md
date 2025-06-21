# Monocular Visual Localization Project for Autonomous Driving

## 1. Project Overview
Monocular visual localization is crucial for advanced driver assistance systems and autonomous driving, estimating a vehicle's ego - motion from a single pinhole camera. This project addresses the challenges of conventional monocular visual odometry (MVO) in scale estimation, computational complexity, and dynamic object management. It extends prior research, proposing innovative strategies for ego - motion estimation and ground point selection, aiming for a balance between computational efficiency and precision.

## 2. Directory Structure
- **`dataset`**: 
    - Intended to store datasets, like the KITTI dataset used in experiments. Place relevant data files (e.g., image sequences, calibration data) here for the project to access during ego - motion estimation and other processes.
- **`script`**: 
    - Will hold various scripts. These could include data preprocessing scripts (to format dataset files for the project), experiment - related scripts (e.g., for batch - running tests on different dataset subsets), or utility scripts to assist in tasks like result visualization setup.
- **`src`**: 
    - The core source code directory. It contains the implementation of key algorithms:
        - Ego - motion estimation logic, integrating the proposed hybrid method with components like the SegNeXt model usage, dynamic object mask handling, and ground plane mask - based triangulation.
        - Ground point selection algorithms, along with the Geometry - constraint - based road region delineation for scale recovery. Also, it has the integration code with the monocular version of ORB - SLAM3 for road model estimation in scale recovery.
- **`.gitignore`**: 
    - Specifies files and directories that Git should ignore. This keeps the repository clean by excluding unnecessary files like temporary files, build outputs, or local configuration files that don't need to be version - controlled.
- **`README.md`**: 
    - This file, providing an overview of the project, its structure, usage, and key details.
- **`feature.ipynb`**: 
    - A Jupyter Notebook. It can be used for feature exploration, like analyzing the performance of different components (e.g., SegNeXt model output on dataset images, effectiveness of dynamic object masks) interactively. Also useful for visualizing intermediate results during ego - motion estimation and scale recovery processes.

## 3. Key Contributions
- **Innovative Ego - motion and Ground Point Strategies**: Extends prior work, introducing new methods for ego - motion estimation and ground point selection to tackle MVO challenges.
- **Hybrid Method with SegNeXt**: Proposes a hybrid approach leveraging the SegNeXt model for real - time applications, covering ego - motion estimation and ground point selection, balancing efficiency and precision.
- **Dynamic and Ground Mask Utilization**: Uses dynamic object masks to remove unstable features and ground plane masks for accurate triangulation. Applies Geometry - constraint for road region delineation in scale recovery.
- **Integration with ORB - SLAM3**: Integrates with the monocular version of ORB - SLAM3 to estimate the road model, a key part of the scale recovery process.
- **Rigorous Experimentation**: Conducts experiments on the KITTI dataset, comparing with existing MVO algorithms and scale recovery methods, showing superior effectiveness.

## 4. Installation and Usage
### Prerequisites
- Ensure you have installed relevant dependencies such as Python (version [recommended version, e.g., 3.8+]), deep learning frameworks (if needed, like PyTorch for SegNeXt model usage), and libraries for computer vision tasks (e.g., OpenCV).
- Download the KITTI dataset (or other relevant datasets) and place it in the `dataset` directory as per the required structure.

### Running the Project
1. **Data Preparation**: 
    - If needed, use scripts in the `script` directory to preprocess the dataset in the `dataset` directory. For example, format image sequences and calibration data to be compatible with the `src` code.
2. **Code Execution**: 
    - Navigate to the `src` directory. Run the main code files (the entry point depends on how the code is structured, e.g., a `main.py` file) to start ego - motion estimation, ground point selection, and scale recovery processes. The code will utilize the dataset from `dataset`, apply the proposed algorithms, and integrate with ORB - SLAM3 as needed.
3. **Exploring Features**: 
    - Open the `feature.ipynb` notebook. Run the cells to explore features, analyze intermediate results, and visualize outputs related to the project's core algorithms.

## 5. Experiment and Result
- **Dataset**: Experiments are conducted on the KITTI dataset. Place the appropriate dataset files in the `dataset` directory following the required structure (e.g., organizing image sequences, calibration files correctly).
- **Comparison**: The project systematically compares the proposed method with existing MVO algorithms and modern scale recovery methodologies. The results, as mentioned, show the superior effectiveness of our approach over state - of - the - art visual odometry algorithms. You can find result - related scripts in the `script` directory (e.g., for result parsing and visualization) and analyze the outputs, which may include metrics like ego - motion estimation accuracy, scale recovery error, etc.

## 6. Contribution
If you want to contribute to this project:
1. Fork the repository.
2. Create a new branch for your feature or bug fix (e.g., `git checkout -b new - feature - branch`).
3. Make your changes in the relevant directories (`src` for algorithm changes, `script` for script updates, etc.).
4. Write tests if applicable (e.g., to ensure new features in `src` work correctly with different datasets).
5. Commit your changes with a clear and descriptive commit message (e.g., "Enhance ego - motion estimation algorithm in src/ego_motion.py").
6. Push the branch to your forked repository.
7. Open a pull request in the original repository, describing your changes in detail, so that they can be reviewed and merged. 



