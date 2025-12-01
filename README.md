
# 🌌 Material Hardness & Oxidation Prediction  
### **AI-Driven Microstructure–Property Intelligence Platform for Materials Engineering**

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge">
  <img src="https://img.shields.io/badge/Flask-Web%20Framework-black?style=for-the-badge&logo=flask">
  <img src="https://img.shields.io/badge/Scikit--Learn-ML%20Pipelines-FCC624?style=for-the-badge&logo=scikitlearn">
  <img src="https://img.shields.io/badge/Explainability-SHAP-ff69b4?style=for-the-badge">
  <img src="https://img.shields.io/badge/Deployment-Render-46E3B7?style=for-the-badge&logo=render">
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge">
</p>

---

# 🚀 Live Deployment  
The full production version of this system is deployed on Render:

👉 **https://material-hardness-oxidation-prediction.onrender.com**

This cloud-hosted version runs Gunicorn + Flask with fully packaged ML models.

---

# ⭐ Executive Summary  
Material Hardness & Oxidation Prediction (**MHOC**) is a **research-grade, enterprise-level** materials intelligence system for modeling process–property relationships in Stellite‑6 hardfaced ferrous alloys.

It combines:

- High‑fidelity ML regressors  
- SHAP-based explainability  
- Modern Flask UI  
- Full EDA + diagnostics  
- Microstructure–aware scientific grounding  
- Modular ML pipelines  
- Render-ready deployment  

Built for researchers, engineers, material scientists, and industrial R&D labs.

---

# 🔬 Scientific Foundation  
Based on the peer‑reviewed experimental study:

**“Experimental Studies of Stellite‑6 Hardfaced Layer on Ferrous Materials by TIG Surfacing Process”**  
IOP Conference Series: Materials Science & Engineering (2020).  
DOI: 10.1088/1757‑899X/998/1/012061  

This project converts hardfacing experiments → ML‑based predictive intelligence.

---

# 🏗 System Architecture  

```
                   ┌───────────────────────────┐
                   │     Web UI (Flask)         │
                   │  HTML • CSS • JS • Charts  │
                   └───────────────┬───────────┘
                                   │
                         User Input Validation
                                   │
                   ┌───────────────▼──────────────┐
                   │   Inference Engine (Python)   │
                   │  Pydantic • Feature Builder   │
                   └───────────────┬──────────────┘
                                   │
         ┌─────────────────────────┼──────────────────────────┐
         ▼                         ▼                          ▼
┌────────────────┐      ┌──────────────────┐       ┌──────────────────────┐
│ Hardness Model │      │ Oxidation Model │       │   Metadata System     │
│ LinearReg / RF │      │ Random Forest   │       │ Versioning • Hashing  │
└───────┬────────┘      └──────────┬──────┘       └──────────┬───────────┘
        │                           │                        │
        └────────────┬──────────────┴──────────────┬─────────┘
                     ▼                             ▼
         ┌───────────────────┐          ┌────────────────────────┐
         │ SHAP Explainability│          │ Performance Diagnostics │
         │ Global + Local     │          │ Residuals • R² • MAE    │
         └───────────┬────────┘          └───────────┬────────────┘
                     ▼                             ▼
                   JSON                        UI Charts
                   Plots                       Reports
```

---

# 🖥️ UI Showcase  
(*Image embedding preserved from repo — paths unchanged.*)

```
screenshots/demo-01-home.png
screenshots/demo-02-predict.png
screenshots/demo-03-hardness-shap.png
screenshots/demo-04-oxidation-shap.png
```

---

# 📊 Exploratory Data Analysis  
Relevant correlation plots, histograms, and distribution analytics:

```
src/app/static/plots/eda_hardness_correlation.png
src/app/static/plots/eda_hardness_hist.png
src/app/static/plots/eda_oxidation_correlation.png
src/app/static/plots/eda_oxidation_hist.png
```

---

# 📈 Model Performance Visualization  

Hardness Model:
```
perf_hardness_actual_vs_pred.png
perf_hardness_residuals.png
fi_hardness_coefficients.png
```

Oxidation Model:
```
perf_oxidation_actual_vs_pred.png
perf_oxidation_residuals.png
fi_oxidation_importances.png
```

---

# 🧠 Machine Learning Pipelines  

### Feature Engineering  
- Scaling  
- Derived heat‑input features  
- Composition variable normalization  
- Outlier mitigation  

### Models  
| Task | Models Used |
|------|-------------|
| Hardness | Linear Regression, Random Forest |
| Oxidation Rate | Random Forest |

### Explainability  
- SHAP global importance  
- SHAP per‑sample breakdown  
- Sensitivity mappings  

---

# 📐 Mathematical Formulation  

### Hardness  
\[
\hat{H} = f(X_{\text{process}}, X_{\text{composition}})
\]

### Oxidation  
\[
\hat{O} = g(T, t, X_{\text{alloy}})
\]

### Loss  
\[
\mathcal{L} = \frac{1}{N}\sum (y_i - \hat{y}_i)^2
\]

---

# 🧩 Directory Structure  

```
material-hardness-oxidation-prediction/
├── data/
├── models/
├── screenshots/
├── src/
│   ├── app/
│   ├── inference/
│   ├── models/
│   └── utils/
├── tests/
├── requirements.txt
├── render.yaml
├── Procfile
└── runtime.txt
```

---

# 🔧 Local Development — Clean & Correct  

## 1. Clone Repo  
```bash
git clone https://github.com/TheComputationalCore/Material-Hardness-Oxidation-Prediction
cd Material-Hardness-Oxidation-Prediction
```

## 2. Create Environment  

### Conda  
```bash
conda create -n mhoc python=3.10
conda activate mhoc
```

### OR venv  
```bash
python3 -m venv mhoc
source mhoc/bin/activate   # Linux/Mac
mhoc\Scripts\activate      # Windows
```

## 3. Install Dependencies  
```bash
pip install -r requirements.txt
```

## 4. Run App  
```bash
python src/app/app.py
```

App runs at:  
👉 **http://localhost:5000**

---

# 🚀 Deployment (Render)

### Build Step  
```bash
pip install -r requirements.txt
```

### Start Command  
```bash
gunicorn "app.app:app" --chdir src --bind 0.0.0.0:$PORT --workers 2
```

---

# 🧪 Testing  
```bash
pytest -q
```

---

# 📘 Documentation  
- docs/MODEL_CARD.md  
- docs/ARCHITECTURE.md  
- docs/API_REFERENCE.md  

---

# 📚 Citation  
```
D. Chandra et al.
"Experimental Studies of Stellite-6 Hardfaced Layer on Ferrous Materials by TIG Surfacing Process."
IOP Conference Series: Materials Science and Engineering,
Vol. 998, 012061, 2020.
doi:10.1088/1757-899X/998/1/012061
```

---

# 👤 Author  
**Dinesh Chandra — TheComputationalCore**

---

# 🔒 License  
MIT License
