# Deep Learning for Vehicle Re-identification in Urban Traffic Monitoring With Visual and Temporal Information

_Official code and resources for the paper:_

> **"Deep Learning for Vehicle Re-identification in Urban Traffic Monitoring With Visual and Temporal Information"**  
> Yura Tak, Robert Fonod, Nikolas Geroliminis <br>
> Accepted in **Communications in Transportation Research**

[//]: # "> Published in **Communicationsin Transporatation Systems**, 2025"
[//]: # "> [📄 PDF](https://infoscience.epfl.ch/entities/publication/a3e71482-1f77-4d10-ad68-556a0d90fa98)  • [🔗 DOI](https://doi.org/10.1109/TITS.2024.3397588)"

---

## 🔍 Overview

This repository contains the official implementation of our paper.  
It includes training scripts, evaluation code, and a pretrained model of our main results.

---

## 📝 Abstract
This paper introduces a novel deep learning framework that enhances vehicle re-identification (ReID) accuracy by integrating visual and temporal data. Vehicle ReID, which identifies target vehicles from large volumes of traffic data, is essential for continuous tracking in large-scale monitoring scenarios involving multiple Unmanned Aerial Vehicles (UAVs). UAV-based monitoring, while offering a comprehensive bird’s-eye view (BEV), faces key challenges: loss of uniquely identifiable features and reliance on visual data, which struggles with vehicles of similar appearance. To overcome these issues, our approach incorporates traffic-oriented features based on shockwave theory to model predictable vehicle travel times. Methods have been tested with data from one of the largest drone experiments with 10 drones monitoring 20 intersections for one week in the city of Songdo in Seoul Area. Experimental results demonstrate a 36.8\% improvement in ReID accuracy over traditional methods, highlighting the potential of UAV-based solutions to complement AV systems for robust and scalable traffic monitoring.

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/yuratak/ReID.git
cd ReID
```

### 2. Install dependencies
```bash
conda create -y -n ReID python==3.7.15 numpy pandas matplotlib
conda activate ReID
conda install pytorch torchvision torchaudio pytorch-cuda -c pytorch -c nvidia
conda install -c conda-forge ignite==0.1.2 scikit-learn
pip install yacs opencv-python geopandas optuna geopy tqdm gdown
```

### 3 Dataset

#### 3.1 Download the data

<pre>
ReID/
├─ data/
  └── KPNEUMA/
      ├── images_gallery/
      ├── images_query/
      ├── images_train/
      ├── images_train_gallery/
      ├── images_train_query/
      ├── temporal_test.txt
      └── temporal_train.txt
</pre>


- [🔗 Download Train/Test Data](https://drive.google.com/file/d/1YX_dDc0Qz6tuDN85PX3qTSpvQUOz4HQF/view?usp=sharing)

Or via command line:

```bash
gdown 1YX_dDc0Qz6tuDN85PX3qTSpvQUOz4HQF
tar -xvzf data.tar.gz
```

#### 3.2 Visual Data Description

The visual dataset contains cropped vehicle images extracted from the drone video sequences.  
Each image corresponds to a single vehicle observation captured by one of the Virtual Loop Detectors (VLDs).  
These images are used to train and evaluate the visual component of the ReID framework.

The visual data files are organized as follows:
- **Training images:** `data/KPNEUMA/images_train`  
- **Query images:** `data/KPNEUMA/images_query`  
- **Gallery images:** `data/KPNEUMA/images_gallery`  
- **Train-query images:** `data/KPNEUMA/images_train_query`  
- **Train-gallery images:** `data/KPNEUMA/images_train_gallery`

<details>
<summary>Data Description</summary>

Each image file is named according to the following format:
XXxxxxxx_yy_zzzz.png, where:
- `X` → Video code identifier  
- `x` → Unique vehicle ID within the dataset  
- `y` → VLD (Virtual Loop Detector) number, corresponding to the **camera ID**  
- `z` → Frame number of the observation in the video sequence  

Example: 04000123_02_0456.png represents **vehicle 000123** observed by **camera/VLD 02** at **frame 456** in video **04**.
</details>

<details>
<summary>Usage Notes</summary>

- **Query and Gallery protocol:**  
  - Each **query** set image corresponds to a **single observation per camera** taken when the vehicle is **entering** a VLD.  
  - Each **gallery** set image corresponds to a **single observation per camera** taken when the vehicle is **exiting** a VLD.

- **Training data population:**  
  - For each vehicle instance, we **sample 5% of its observed frames** across the sequence and **crop the vehicle** from those frames to create the training images.
</details>

#### 3.3 Temporal Data Description

The temporal dataset contains vehicle-level observations extracted from the video sequences.  

Each row corresponds to one vehicle passing through a pair of Virtual Loop Detectors (VLDs).  

The temporal data files can be found at the following locations:
- **Training data:** `data/KPNEUMA/temporal_train.txt`  
- **Testing data:** `data/KPNEUMA/temporal_test.txt`

<details> <summary> Data Description </summary>

| **Column** | **Description** |
|-------------|-----------------|
| `id` | Unique identifier of the vehicle instance. |
| `traj_src` | Historical trajectory of the vehicle before crossing the source VLD, represented as pixel coordinates. |
| `vel_src` | Estimated vehicle velocity at the source VLD (in pixels per second or converted physical units). |
| `category` | Vehicle type or class (e.g., car, bus, truck). |
| `vld_nb_src` | Identifier of the source Virtual Loop Detector where the vehicle is first observed. |
| `vld_nb_dst` | Identifier of the destination Virtual Loop Detector where the vehicle is re-observed. |
| `lane_nb_src` | Lane number at the source VLD, assigned using clustered trajectories. |
| `lane_nb_dst` | Lane number at the destination VLD. |
| `lane_signalized_src` | Binary flag indicating whether the lane at the source VLD is signalized (`1`) or not (`0`). |
| `time_to_green` | Time remaining until the next green signal for the lane when the vehicle is detected at the source VLD (in seconds). |
| `distance` | Estimated distance between the source and destination VLDs (in meters or pixels, depending on calibration). |
| `traj_btwn_vlds` | Intermediate trajectory or travel path between the two VLDs, used for temporal modeling. |
| `avg_travel_time_btwn_vlds` | Average historical travel time between the same pair of VLDs, computed from previous observations. |
| `y` | Ground-truth travel time of the vehicle between the source and destination VLDs (target variable). | 
</details>

### 4. Download the pretrained model

<pre>
ReID/
├─ saved_models/
  ├── visual/
      └── KPNEUMA/
      └── VRAI/
  └── temporal/
</pre>


#### 4.1 Visual and Temporal Component weights (Our Best model)
- [🔗 Download model_weights](https://drive.google.com/file/d/14n4xM2YBiANkdL1jgWOFsj27Uiqpdag8/view?usp=sharing)

Or via command line:

```bash
gdown 14n4xM2YBiANkdL1jgWOFsj27Uiqpdag8
tar -xvzf saved_models.tar.gz
```

## ⚙️ Running Inference and Training

<pre>
ReID/
├── test.py
├── result.ipynb
├── train_visual_component.py
└── train_temporal_component.py
</pre>

### Evaluate with pretrained model

#### Proposed Framework

```bash
python test.py
```

`result.ipynb` provides the code to reproduce the main figures from the paper.


### Train Visual Component
```bash
python train_visual_component.py
```

Note:
Training the visual component requires pretrained baseline model weights.
Before launching the training script, please ensure that you have downloaded the pretrained weights as described in Section 4.1 below.

### Train Temporal Component
```bash
python train_temporal_component.py
```

---

## 📊 Results

| Method   |    mAP   |   CMC-1  |   CMC-5  |
|----------|----------|----------|----------|
| Visual Component (Baseline) | 0.769    | 0.669    | 0.896    |
|Temporal Component| 0.263    | 0.125    | 0.420    |
| **Proposed Framework (Ours)** | **0.949**| **0.915**| **0.984**|

See the paper for more detailed results and visualizations.

---


## 📌 Citation

If you use this code or find our work helpful, please consider citing:

```bibtex
```

---

## 📬 Contact

For questions, feedback, please open an issue on this repository or contact the authors via the corresponding email address provided in the paper.

---

## 🤝 Acknowledgment

The directory **reid_strong_baseline** corresponds to a third-party library.  
The original implementation is available at: https://github.com/michuanhaohao/reid-strong-baseline.git.  
All credit for the code inside that directory goes entirely to the authors of the original repository.

In this project, that directory is included as an unmodified copy of the framework, with the exception of adding a custom dataset for our drone-based experiments. All other parts of this repository are our own work.

---