# Road Damage Detection Using YOLOv8 with CBAM

> Intelligent Road Infrastructure Monitoring System, trained on RDD2022

![Python](https://img.shields.io/badge/Python-3.11-blue)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-yellow)
![Streamlit](https://img.shields.io/badge/Deployed-Streamlit-red)

A lightweight, real-time road damage detection system that identifies four types of surface damage — **Longitudinal Crack (D00), Transverse Crack (D10), Alligator Crack (D20), and Pothole (D40)** — using YOLOv8n, evaluated against two CBAM (Convolutional Block Attention Module) variants.

The live app lets you scan a road photo with any of the three trained models, or run all three side by side to see how attention placement changes detection behaviour.

---

### 🔬 Problem Statement
Manual road inspection is labour-intensive, slow, and subjective. This project builds a lightweight, real-time detection pipeline and rigorously tests whether adding channel/spatial attention (CBAM) to a YOLOv8n backbone improves detection of small, low-contrast damage such as cracks.

### 📂 Dataset: RDD2022
- **47,420 images** from 6 countries (Japan, India, Czech Republic, Norway, USA, China)
- **55,000+ annotations**, reformatted here to YOLO format
- Classes: `D00` Longitudinal, `D10` Transverse, `D20` Alligator, `D40` Pothole
- Source: CRDDC 2022 Challenge — Arya et al., 2022

### 🏗️ Models Compared
| Model | Description | mAP@50 | FPS |
|---|---|---|---|
| **Baseline** | Standard YOLOv8n | 0.521 | 33.6 |
| **CBAM Dual** | 2 CBAM blocks in the backbone | 0.489 | 37.9 |
| **CBAM Single** | 1 CBAM block, immediately before SPPF | 0.495 | 38.9 |

All three were trained for 50 epochs on identical data splits and verified for architectural correctness (parameter counts independently checked against CBAM's theoretical overhead) before comparison. Full methodology and per-class results are in the project report.

### 🚀 App Features
- **Single-model mode** — pick one model, upload a photo, get detections with a confidence/latency/FPS readout
- **Compare-all mode** — run the same photo through all three models side by side
- Adjustable confidence and IoU (NMS) thresholds, and inference resolution
- Detection report table (class, confidence, bounding box)
- Downloadable annotated image
- Built-in benchmark reference (sidebar) pulled from the project's validation results

### 📁 Project Structure
```
├── app.py
├── requirements.txt
├── .streamlit/
│   └── config.toml
├── weights/
│   ├── baseline_best.pt
│   ├── cbam_dual_best.pt
│   └── cbam_single_best.pt
└── README.md
```

### ⚙️ Installation & Run Locally
```bash
git clone https://github.com/YOUR_USERNAME/yolov8-cbam-road-damage-detection.git
cd yolov8-cbam-road-damage-detection
pip install -r requirements.txt
streamlit run app.py
```

### 📦 Requirements
```
--extra-index-url https://download.pytorch.org/whl/cpu
streamlit==1.39.0
ultralytics==8.3.8
Pillow==10.4.0
opencv-python-headless==4.10.0.84
torch==2.4.1
torchvision==0.19.1
numpy==1.26.4
```

### 🔗 Deployment
Deployed on Streamlit Community Cloud: `https://your-app-link.streamlit.app`

To deploy your own copy:
1. Push this repo (including `weights/*.pt` and `.streamlit/config.toml`) to GitHub.
2. On [streamlit.io/cloud](https://streamlit.io/cloud), create a new app pointing at `app.py`.
3. Streamlit Cloud reads `.streamlit/config.toml` automatically — no extra setup needed for the theme.

### 📚 References
- D. Arya et al., "RDD2022: A Multi-National Image Dataset for Automatic Road Damage Detection," arXiv:2209.08538, 2022.
- G. Jocher et al., "Ultralytics YOLOv8," Ultralytics, 2023.
- S. Woo et al., "CBAM: Convolutional Block Attention Module," ECCV 2018.

### 👨‍💻 Team
S. N. Bose Summer Internship Program, National Institute of Technology Silchar
Guided by Dr. Malaya Dutta Borah

Bikash Samanta · Nandan Kumar · Bittoo Kumar · Aditya Kumar Pandey