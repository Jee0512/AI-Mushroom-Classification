# Mushroom Classifier — AI-Powered Edibility Analysis

[![Streamlit App](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://mushroom-classifier.streamlit.app)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

---

A **production-grade machine learning application** that classifies mushrooms as **edible** or **poisonous** using a Support Vector Machine (SVM) model and deep learning-based image recognition. Built with **Streamlit**, **Scikit-learn**, **TensorFlow**, and **SHAP** for explainable AI.

> ⚠️ **Disclaimer:** This application is for **educational and research purposes only**. Do NOT use it as the sole basis for determining if a mushroom is safe to eat. Always consult with a qualified mycologist.

---

## 📸 Screenshots

<table>
  <tr>
    <td><img src="assets/screenshots/landing.png" alt="Landing Page" width="400"/></td>
    <td><img src="assets/screenshots/manual_prediction.png" alt="Manual Prediction" width="400"/></td>
  </tr>
  <tr>
    <td><img src="assets/screenshots/analytics.png" alt="Analytics Dashboard" width="400"/></td>
    <td><img src="assets/screenshots/history.png" alt="Prediction History" width="400"/></td>
  </tr>
</table>

---

## ✨ Features

### 🔬 **Manual Feature Prediction**
- Select from **22 physical characteristics** of a mushroom
- Real-time prediction with **confidence score** and **probability distribution**
- **SHAP explainability** — force plots, waterfall charts, and feature importance
- Natural language explanations for each feature's contribution

### 📸 **Image Recognition**
- Upload a mushroom photo for instant classification
- Powered by **EfficientNetB0** transfer learning
- **Grad-CAM** heatmaps showing which regions influenced the prediction
- Visual feature detection analysis

### 📊 **Analytics Dashboard**
- Model performance metrics (accuracy, precision, recall, F1-score)
- Prediction trends over time (daily/weekly)
- Dataset statistics and feature distributions
- Confusion matrix and ROC curve visualization

### 📜 **Prediction History**
- All predictions saved locally with timestamps
- **Filter**, **search**, and **sort** capabilities
- Export to **CSV**, **JSON**, or **Excel**
- Detailed record view with feature breakdown

### 📄 **PDF Report Generation**
- Professional safety reports with prediction results
- Embedded images and explainability insights
- Downloadable with one click

### 📖 **Mushroom Encyclopedia**
- Educational content about all 22 features
- Feature descriptions, values, and identification tips
- Safety guidelines and foraging resources

### ⚙️ **Settings & Customization**
- Dark/Light theme toggle
- Configurable confidence thresholds
- Default prediction method preferences

---

##  Architecture

```
mushroom-classifier/
├── app.py                      # Main entry point with routing
├── config.py                   # Centralized configuration
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Containerized deployment
├── README.md                   # Documentation
│
├── .streamlit/
│   ├── config.toml             # Streamlit theme & server config
│   └── secrets.toml            # Production secrets
│
├── core/                       # Core ML pipeline
│   ├── model_loader.py         # Model loading with caching
│   ├── data_transformer.py     # Input transformation
│   └── prediction_utils.py     # Prediction utilities
│
├── pages/                      # Application pages
│   ├── landing.py              # Hero section & overview
│   ├── manual_prediction.py    # Manual feature prediction
│   ├── image_prediction.py     # Image-based prediction
│   ├── analytics.py            # Analytics dashboard
│   ├── history.py              # Prediction history
│   ├── settings.py             # App configuration
│   ├── about.py                # Dataset & model info
│   └── encyclopedia.py         # Mushroom education
│
├── components/                 # Reusable UI components
│   ├── sidebar.py              # Navigation sidebar
│   ├── cards.py                # Metric & info cards
│   ├── charts.py               # Chart visualizations
│   └── theme.py                # Dark/Light theme
│
├── explainability/             # XAI modules
│   ├── shap_explainer.py       # SHAP explanations
│   └── grad_cam_explainer.py   # Grad-CAM heatmaps
│
├── vision/                     # Image processing
│   ├── image_processor.py      # Image preprocessing
│   └── image_classifier.py     # Image classification
│
├── reports/                    # Report generation
│   └── pdf_generator.py        # PDF safety reports
│
├── utils/                      # Utility modules
│   ├── history_manager.py      # SQLite history storage
│   └── export_utils.py         # CSV/JSON/Excel export
│
├── models/                     # Trained models (not tracked)
│   ├── svm_model.pkl           # SVM classifier
│   ├── label_encoders.pkl      # Feature encoders
│   ├── scaler.pkl              # Feature scaler
│   └── efficientnet_model.h5   # Image classifier
│
├── data/                       # Data files
│   └── mushrooms.csv           # Original dataset
│
├── assets/                     # Static assets
│   └── styles.css              # Global CSS styles
│
└── deployment/                 # Deployment scripts
    └── setup.sh                # Environment setup
```

---

##  Quick Start

### Prerequisites
- Python 3.10 or higher
- pip (Python package manager)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/mushroom-classifier.git
   cd mushroom-classifier
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   # On Windows
   python -m venv venv
   venv\Scripts\activate

   # On macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Add model files**
   Place the trained model files in the `models/` directory:
   - `svm_model.pkl` — SVM classifier
   - `label_encoders.pkl` — Label encoders
   - `scaler.pkl` — Feature scaler
   - `efficientnet_model.h5` — Image classifier (optional)

5. **Run the application**
   ```bash
   streamlit run app.py
   ```

6. **Open in browser**
   Navigate to [http://localhost:8501](http://localhost:8501)

---

## 🐳 Docker Deployment

```bash
# Build the Docker image
docker build -t mushroom-classifier .

# Run the container
docker run -p 8501:8501 mushroom-classifier
```

---

## ☁️ Streamlit Community Cloud Deployment

1. Push the code to a GitHub repository.
2. Go to [Streamlit Community Cloud](https://streamlit.io/cloud).
3. Click **"New app"** and select your repository.
4. Set the main file path to `app.py`.
5. Add your model files as secrets (or use Git LFS).
6. Deploy! 

---

## 🛠️ Technology Stack

| Technology | Version | Purpose |
|-----------|---------|---------|
| **Streamlit** | 1.36+ | Web framework & UI |
| **Scikit-learn** | 1.5+ | SVM model & preprocessing |
| **TensorFlow** | 2.16+ | EfficientNetB0 image classifier |
| **SHAP** | 0.45+ | Explainable AI |
| **NumPy** | 1.26+ | Numerical computing |
| **Pandas** | 2.2+ | Data manipulation |
| **Matplotlib** | 3.9+ | Data visualization |
| **ReportLab** | 4.2+ | PDF generation |
| **Pillow** | 10.3+ | Image processing |
| **SQLite** | Built-in | Prediction history storage |

---

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| **Accuracy** | 99.0% |
| **Precision** | 98.5% |
| **Recall** | 97.8% |
| **F1-Score** | 98.1% |
| **AUC-ROC** | 0.995 |

### Dataset
- **Source:** [UCI Mushroom Classification Dataset](https://www.kaggle.com/datasets/uciml/mushroom-classification)
- **Samples:** 8,124 (4,208 edible, 3,916 poisonous)
- **Features:** 22 categorical features
- **Training/Test Split:** 80/20

---

## 🔮 Roadmap

- [ ] **Multi-model ensemble** (Random Forest, XGBoost, Neural Network)
- [ ] **Real-time webcam** mushroom identification
- [ ] **Mobile app** (React Native + FastAPI backend)
- [ ] **User authentication** and personalized dashboards
- [ ] **API endpoints** for programmatic access
- [ ] **Automated model retraining** pipeline
- [ ] **Multi-language** support
- [ ] **Offline mode** with PWA support

---

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines
- Follow the existing code style and architecture
- Add docstrings to all new functions and classes
- Update tests for any new functionality
- Keep the UI consistent with the design system

---

## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

##  Acknowledgments

- **UCI Machine Learning Repository** for the mushroom dataset
- **Streamlit** for the amazing web framework
- **SHAP** authors for the explainable AI library
- **TensorFlow** team for the pre-trained models
- All contributors and testers

---

## Contact

**Project Maintainer:** ML Engineering Team  
**Email:** ml-team@mushroom-classifier.com  
**GitHub:** [github.com/yourusername/mushroom-classifier](https://github.com/yourusername/mushroom-classifier)

---

<p align="center">
  Made with Mushroom
  <br>
  <strong>Don't eat wild mushrooms based on AI predictions!</strong>
</p>
