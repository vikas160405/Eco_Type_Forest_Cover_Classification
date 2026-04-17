#🌲 EcoType: Forest Cover Type Classification

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


## ⚙️ Installation

git clone https://github.com/vikas160405/Eco_Type_Forest_Cover_Classification.git
cd Eco_Type_Forest_Cover_Classification
pip install -r requirements.txt


Step 1: Train Model
python ecotype_pipeline.py

Step 2: Run Application
streamlit run app.py

Open in browser:
http://localhost:8502/
