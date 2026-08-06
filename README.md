# Deep Learning ANN Projects

This repository now contains multiple deep learning notebooks built with TensorFlow/Keras. The main app predicts customer churn with an Artificial Neural Network (ANN), and the project also includes experiments for hyperparameter tuning and a separate salary regression workflow.

## What Changed

- The churn model artifact in the repository is now `model.h5`
- A new `hyperparametertuning.ipynb` notebook was added
- A new `salaryregression.ipynb` notebook was added
- Regression TensorBoard outputs are stored in `regressionlogs/`
- The repository no longer includes `model.keras`

## Main Churn Prediction Project

The churn prediction workflow uses `Churn_Modelling.csv` and predicts whether a bank customer is likely to churn based on:

- `CreditScore`
- `Gender`
- `Age`
- `Tenure`
- `Balance`
- `NumOfProducts`
- `HasCrCard`
- `IsActiveMember`
- `EstimatedSalary`
- `Geography`

The current churn-related assets are:

- `exp.ipynb` for ANN training and experimentation
- `prediction.ipynb` for loading the saved model and testing predictions
- `hyperparametertuning.ipynb` for tuning the churn model
- `model.h5` for the trained churn model
- `label_encoder_gender.pkl` for gender encoding
- `ohe_geo.pkl` for geography one-hot encoding
- `scaler.pkl` for feature scaling
- `app.py` for the Streamlit interface
- `logs/` for TensorBoard training logs

## Salary Regression Project

The repository also includes `salaryregression.ipynb`, which explores a separate ANN-based regression workflow. Its TensorBoard outputs are stored in `regressionlogs/`.

## Tech Stack

- Python
- TensorFlow / Keras
- pandas
- NumPy
- scikit-learn
- Streamlit
- Jupyter Notebook
- TensorBoard
- Matplotlib
- SciKeras

## Repository Structure

```text
.
|-- app.py
|-- Churn_Modelling.csv
|-- exp.ipynb
|-- hyperparametertuning.ipynb
|-- prediction.ipynb
|-- salaryregression.ipynb
|-- model.h5
|-- label_encoder_gender.pkl
|-- ohe_geo.pkl
|-- scaler.pkl
|-- requirements.txt
|-- logs/
|-- regressionlogs/
```

## Preprocessing Pipeline

For churn inference, the preprocessing artifacts are reused exactly as they were saved during training:

- `label_encoder_gender.pkl` encodes `Gender`
- `ohe_geo.pkl` expands `Geography`
- `scaler.pkl` scales the final feature set before prediction

The one-hot encoded geography columns are:

- `Geography_France`
- `Geography_Germany`
- `Geography_Spain`

## Running The App

### 1. Create and activate a virtual environment

```powershell
python -m venv annvenv
.\annvenv\Scripts\Activate.ps1
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Start Streamlit

```bash
streamlit run app.py
```

## Important Note

The repository currently stores the churn model as `model.h5`, while `app.py` still loads `model.keras`. Update the model path in `app.py` before running the Streamlit app, or save the trained model again using the filename expected by the app.

## Learning Outcomes

- Binary classification with ANNs
- Regression with ANNs
- Hyperparameter tuning workflows
- Reusing saved preprocessing artifacts
- TensorBoard-based training inspection
- Building a simple Streamlit interface for inference

## Future Improvements

- Align `app.py` with the current `model.h5` artifact
- Add evaluation metrics and visualizations
- Move training logic from notebooks into reusable Python modules
- Add tests for preprocessing and inference
- Document the salary regression dataset and prediction target more explicitly

## Conclusion

This repository has grown from a single churn prediction example into a small collection of ANN experiments covering classification, tuning, and regression workflows.
