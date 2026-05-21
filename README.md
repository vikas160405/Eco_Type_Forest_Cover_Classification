# 🌲 EcoType: Forest Cover Type Classification

## 📌 Overview
EcoType is a machine learning-based application that predicts forest cover types using environmental and cartographic features such as elevation, slope, and distances to hydrology.

This project demonstrates a complete end-to-end workflow including data preprocessing, feature engineering, model training, evaluation, and deployment using Streamlit.

---

## 🎯 Objectives
- Build a classification model to predict forest cover types  
- Apply data preprocessing and feature engineering techniques  
- Compare multiple machine learning algorithms  
- Deploy the model with a user-friendly web interface  

---

## 🧠 Methodology

### Data Preprocessing
- Removed duplicate values  
- Handled missing values using median and mode  
- Treated outliers using IQR method  

### Feature Engineering
- Distance to hydrology  
- Mean hillshade  
- Hydrology-road distance difference  

### Data Transformation
- Log transformation for skewed features  
- Feature scaling using StandardScaler  

### Class Imbalance Handling
- Applied Random Oversampling  

### Model Training
Models used:
- Random Forest  
- Decision Tree  
- Logistic Regression  
- K-Nearest Neighbors  
- XGBoost  

Best model selected based on accuracy and cross-validation.

---

## 🏗️ Project Structure

```bash
Eco_Type_Forest_Cover_Classification/
│
├── ecotype_pipeline.py
├── app.py
├── cover_type.csv
├── requirements.txt
├── models/
│   ├── best_model.pkl
│   ├── scaler.pkl
│   ├── encoders.pkl
│   ├── top_features.pkl
│   └── skewed_features.pkl
```

---

## ⚙️ Installation

```bash
git clone https://github.com/vikas160405/Eco_Type_Forest_Cover_Classification.git
cd Eco_Type_Forest_Cover_Classification
pip install -r requirements.txt
```

---

## ▶️ Run the Project

### Step 1: Train the Model

```bash
python ecotype_pipeline.py
```

### Step 2: Run the Streamlit Application

```bash
streamlit run app.py
```

Open in browser:

```bash
http://localhost:8502/
```

---

## 📊 Project Presentation (PPT)

📥 View / Download PowerPoint Presentation:  
https://drive.google.com/file/d/1jSxIvthxzv9EJhWpVxJYDG14qhQGeVM9/view?usp=sharing

---

## 🚀 Technologies Used
- Python  
- Pandas  
- NumPy  
- Scikit-learn  
- XGBoost  
- Streamlit  
- Matplotlib  
- Seaborn  

---

## 📌 Features
- Interactive Streamlit web interface  
- Real-time forest cover prediction  
- Multiple ML model comparison  
- Data visualization and analysis  
- End-to-end ML workflow  

---

## 📈 Future Improvements
- Add deep learning models  
- Deploy using Docker and cloud platforms  
- Add live GIS map visualization  
- Improve UI/UX with advanced dashboards  

---
