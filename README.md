# Running ByteTrack - Personal Practice Repository

This repository is a **personal practice project** for running and experimenting with ByteTrack and other related modules on my local machine. It is intended for **learning purposes only** and is **not a copy or plagiarism** of any existing work.

## 📌 Purpose
- Practice cloning, setting up, and running enterprise models on a local machine.
- Gain hands-on experience with tracking algorithms and related dependencies.
- Document troubleshooting steps and solutions encountered while working with these models.

## 📝 Modifications & Learning Process
- Followed the **Jupyter Notebook tutorial** provided by the original ByteTrack repository.
- Edited the **`run.py`** file to incorporate **`detect.py`** from the **YOLOv5** repository for object detection before tracking.
- Experimented with different configurations to understand how ByteTrack integrates with YOLOv5.
- Used the **pretrained model weights provided in the tutorial** for inference.
- **Generated `output_compressed.mp4` as the final result** after running the model.

## ⚠️ Disclaimer
This repository **does not claim ownership** of the original ByteTrack or YOLOv5 code. It is purely for personal learning, and **not for redistribution or publication**.

## 🔧 Getting Started
To run ByteTrack on your local machine, follow these steps:

1. **Clone this repository**:
   git clone https://github.com/Jung0219/Running-ByteTrack.git
   cd Running-ByteTrack
2. Setup environment
  conda create -n bytetrack python=3.8 -y
  conda activate bytetrack
  pip install -r requirements.txt
  must install best.pt from the original paper's jupyter notebooko tutorial.
4. run file
  python run.py
