# 🌲 EcoType: Forest Cover Type Prediction System

## 📌 Overview

EcoType is a machine learning-powered web application that predicts forest cover types based on cartographic and environmental attributes such as elevation, slope, hydrology distance, hillshade values, wilderness areas, and soil types.

The project demonstrates a complete end-to-end machine learning workflow, including data preprocessing, feature engineering, model training, evaluation, model selection, and deployment using Streamlit.

---

## 🎯 Project Objectives

* Predict forest cover types using environmental and geographical features
* Perform data preprocessing and feature engineering
* Train and compare multiple machine learning models
* Select the best-performing model based on evaluation metrics
* Deploy the model through an interactive Streamlit application
* Visualize prediction confidence using probability distributions

---

## 📊 Dataset Information

**Dataset:** UCI Forest CoverType Dataset

### Features Used

* Elevation
* Aspect
* Slope
* Horizontal Distance to Hydrology
* Vertical Distance to Hydrology
* Horizontal Distance to Roadways
* Horizontal Distance to Fire Points
* Hillshade (9 AM, Noon, 3 PM)
* Wilderness Area
* Soil Type

### Target Variable

* Forest Cover Type (7 Classes)

---

## 🧠 Machine Learning Workflow

### Data Preprocessing

* Removed duplicate records
* Handled missing values
* Outlier treatment using IQR method
* Data validation and cleaning

### Feature Engineering

Additional features created:

* Distance_To_Hydrology
* Mean_Hillshade
* Hydro_Road_Distance_Diff

### Feature Transformation

* Log transformation on skewed features
* Feature scaling using StandardScaler

### Class Balancing

* Random Oversampling applied to handle class imbalance

---

## 🤖 Models Evaluated

| Model               | Status       |
| ------------------- | ------------ |
| Random Forest       | ✅ Trained    |
| Decision Tree       | ✅ Trained    |
| Logistic Regression | ✅ Trained    |
| K-Nearest Neighbors | ⭐ Best Model |
| XGBoost             | ✅ Trained    |

### Best Model

**K-Nearest Neighbors (KNN)** was selected as the final model based on overall performance and validation results.

---

## 🌐 Streamlit Application Features

### User Input Interface

Users can provide:

* Terrain information
* Hydrology-related measurements
* Roadway and fire-point distances
* Hillshade values
* Wilderness area
* Soil type

### Prediction Output

The application displays:

✅ Predicted Forest Cover Type

### Confidence Analysis

The application also provides:

* Prediction Confidence Table
* Probability Bar Chart
* Probability Distribution Pie Chart

---

## 🏗️ Project Structure

```bash
Eco_Type_Forest_Cover_Classification/
│
├── ecotype_pipeline.py
├── app.py
├── cover_type.csv
├── requirements.txt
│
├── models/
│   ├── best_model.pkl
│   ├── scaler.pkl
│   ├── encoders.pkl
│   ├── top_features.pkl
│   └── skewed_features.pkl
│
└── README.md
```

---

## ⚙️ Installation

```bash
git clone https://github.com/vikas160405/Eco_Type_Forest_Cover_Classification.git

cd Eco_Type_Forest_Cover_Classification

pip install -r requirements.txt
```

---

## ▶️ Running the Project

### Train the Model

```bash
python ecotype_pipeline.py
```

### Launch the Streamlit Application

```bash
streamlit run app.py
```

Default URL:

```bash
http://localhost:8501
```

---

## 📊 Application Output

The deployed application provides:

* Forest Cover Type Prediction
* Prediction Confidence Scores
* Interactive Probability Visualizations
* User-Friendly Dashboard

---

---

## 📊 Project Presentation (PPT)

📥 View / Download PowerPoint Presentation:

https://drive.google.com/file/d/14M2_kzINH4Qs50dqvAD2De0Kx8bjsrCz/view?usp=sharing

The presentation covers:

- Problem Statement
- Dataset Overview
- Data Preprocessing
- Feature Engineering
- Model Training & Evaluation
- Best Model Selection
- Streamlit Application
- Prediction Results
- Future Enhancements

---

## 🚀 Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* Streamlit
* Matplotlib
* Seaborn

---

## 📈 Future Enhancements

* Deep Learning-based classification
* Cloud deployment (AWS/Azure/GCP)
* Docker containerization
* GIS-based map visualization
* Advanced dashboard analytics
* Real-time environmental data integration

---

## 👨‍💻 Author

**Vikas Relangi**

B.Tech – Artificial Intelligence & Data Science

Passionate about Machine Learning, Data Science, Artificial Intelligence, and Predictive Analytics.

---

## 📄 License

This project is intended for educational and academic purposes.
