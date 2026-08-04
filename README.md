# Customer Churn Prediction Using ANN

This project is a beginner-friendly deep learning application that predicts whether a bank customer is likely to leave the bank, also known as customer churn. It uses an Artificial Neural Network (ANN) built with TensorFlow/Keras, along with scikit-learn preprocessing tools and a simple Streamlit web interface for interactive predictions.

The repository combines three parts of a typical machine learning workflow:

1. Data exploration and experimentation in Jupyter notebooks
2. Model training and preprocessing artifact generation
3. A lightweight deployment interface for end users through Streamlit

## Project Objective

The goal of this project is to learn how to build and deploy a binary classification model using deep learning. Given a set of customer attributes such as geography, age, balance, credit score, and account activity, the model estimates the probability that the customer will churn.

This type of problem is common in banking, telecom, SaaS, and subscription-based businesses, where retaining existing customers is often more cost-effective than acquiring new ones.

## Problem Statement

Customer churn prediction is a supervised learning problem where:

- Input: Customer features such as credit score, age, tenure, balance, salary, and region
- Output: A prediction indicating whether the customer is likely to churn

In this project, the output is represented as a probability between `0` and `1`. A probability above `0.5` is currently treated as a churn prediction in the Streamlit app.

## Dataset

The project uses the `Churn_Modelling.csv` dataset, which contains bank customer information commonly used for churn prediction practice projects.

The model uses the following features during inference:

- `CreditScore`
- `Gender`
- `Age`
- `Tenure`
- `Balance`
- `NumOfProducts`
- `HasCrCard`
- `IsActiveMember`
- `EstimatedSalary`
- `Geography` encoded into:
  - `Geography_France`
  - `Geography_Germany`
  - `Geography_Spain`

The geography categories stored in the trained encoder are:

- `France`
- `Germany`
- `Spain`

The gender encoder stores:

- `Female`
- `Male`

## Tech Stack

- Python
- TensorFlow / Keras
- scikit-learn
- pandas
- NumPy
- Streamlit
- Jupyter Notebook

## Repository Structure

```text
.
|-- app.py                     # Streamlit app for live churn prediction
|-- Churn_Modelling.csv        # Source dataset
|-- exp.ipynb                  # Main experimentation/training notebook
|-- prediction.ipynb           # Notebook focused on prediction workflow
|-- model.h5                   # Trained ANN model
|-- label_encoder_gender.pkl   # Saved LabelEncoder for gender
|-- ohe_geo.pkl                # Saved OneHotEncoder for geography
|-- scaler.pkl                 # Saved StandardScaler for numeric/model inputs
|-- requirements.txt           # Python dependencies
|-- logs/                      # Training logs
```

## How the Project Works

### 1. Data Preprocessing

Before training or prediction, categorical and numerical features are transformed into a format suitable for the ANN.

#### Gender Encoding

`Gender` is label-encoded into numeric form:

- `Female` / `Male` -> integer values through `LabelEncoder`

#### Geography Encoding

`Geography` is one-hot encoded using a saved `OneHotEncoder`. This converts a single categorical feature into multiple binary columns:

- `Geography_France`
- `Geography_Germany`
- `Geography_Spain`

This is why code such as `ohe_geo.categories_[0]` or `ohe_geo.get_feature_names_out(['Geography'])` is used in the app. It ensures that inference uses the same category order and feature names that were learned during training.

#### Feature Scaling

After categorical encoding, all model input columns are scaled using the saved `StandardScaler`.

The scaler expects exactly these input columns:

```text
CreditScore
Gender
Age
Tenure
Balance
NumOfProducts
HasCrCard
IsActiveMember
EstimatedSalary
Geography_France
Geography_Germany
Geography_Spain
```

Using the saved scaler at inference time is important because the ANN was trained on scaled features, not raw values.

### 2. Model Training

The ANN model is stored in `model.h5`. While the exact architecture is defined in the training notebook, the overall workflow is:

- Load and clean the dataset
- Encode categorical features
- Split data into training and testing sets
- Scale the features
- Train an ANN for binary classification
- Save the trained model and preprocessing objects

### 3. Streamlit Prediction App

The `app.py` file provides a simple interface where the user can enter customer details and receive a churn probability.

The app workflow is:

1. Load the trained ANN model
2. Load the saved encoders and scaler
3. Collect user input from the Streamlit UI
4. Encode `Gender` and `Geography`
5. Merge all features into one DataFrame
6. Scale the input
7. Run model prediction
8. Display churn probability and final decision

## Setup Instructions

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd "DL(ANN) Project"
```

### 2. Create and Activate a Virtual Environment

On Windows PowerShell:

```powershell
python -m venv annvenv
.\annvenv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit App

```bash
streamlit run app.py
```

After running the command, Streamlit will open a local web page where you can enter customer information and view the model's prediction.

## Example Prediction Flow

A user selects or enters:

- Geography
- Gender
- Age
- Balance
- Credit Score
- Estimated Salary
- Tenure
- Number of Products
- Credit Card status
- Active Member status

The app converts this information into the exact numeric format required by the ANN and then predicts the churn probability.

## Saved Model Artifacts

This project includes the following reusable artifacts:

- `model.h5`: trained neural network model
- `label_encoder_gender.pkl`: saved encoder for the `Gender` column
- `ohe_geo.pkl`: saved one-hot encoder for the `Geography` column
- `scaler.pkl`: saved feature scaler for final model inputs

These files are essential for keeping training-time preprocessing consistent with prediction-time preprocessing.

## Learning Outcomes

This project is useful for understanding:

- End-to-end binary classification with deep learning
- Categorical encoding using `LabelEncoder` and `OneHotEncoder`
- Numerical feature scaling using `StandardScaler`
- Saving and loading preprocessing objects with `pickle`
- Deploying a basic ML model with Streamlit
- Keeping inference preprocessing aligned with training preprocessing

## Current Notes and Limitations

- The trained preprocessing objects were created with a newer scikit-learn version than the one currently detected in one local test environment. If you see version warnings while loading `.pkl` files, use a matching scikit-learn version to avoid compatibility issues.
- The app currently uses a fixed threshold of `0.5` to convert probability into a churn / non-churn label.
- This project is designed primarily as a learning and practice project rather than a production-ready banking system.
- The ANN model file is saved in the legacy `.h5` format. It works, but newer Keras projects often prefer the `.keras` format.

## Future Improvements

- Add evaluation metrics such as accuracy, precision, recall, F1-score, and ROC-AUC
- Display confidence and richer explanations in the Streamlit UI
- Add input validation and better default values in the app
- Export the training pipeline into a dedicated Python module
- Add model retraining instructions
- Add unit tests for preprocessing and inference
- Deploy the application to Streamlit Community Cloud, Render, or another hosting platform

## Conclusion

This project demonstrates a complete beginner-to-intermediate machine learning workflow using an Artificial Neural Network for churn prediction. It is a solid example of how data preprocessing, model training, saved artifacts, and a simple deployment layer come together in a real project structure.

If you are learning deep learning, TensorFlow, or ML deployment with Streamlit, this repository is a good hands-on reference for building and serving a tabular classification model.
