# Multiple-Disease-Prediction-System
This project is a web application built using Streamlit to predict the presence of three diseases based on user input:

- Diabetes
- Heart Disease
- Parkinson's Disease

The models were trained using machine learning algorithms and saved using joblib for reuse.

---

## Features

- Streamlit-based user interface
- Input form for each disease
- Predictions using trained models
- Modular code structure for easy maintenance

---

## Technologies Used

- Python 3.11
- Streamlit
- scikit-learn
- NumPy
- joblib
- streamlit-option-menu

---

## Files Included

- `app.py` - Main Streamlit app file
- `ensemble_model_diabetes.pkl` - Trained diabetes prediction model
- `Heart_disease_prediction.pkl` - Trained heart disease model
- `Parkinson_Disease_Predictor.pkl` - Trained Parkinson's disease model
- `requirements.txt` - List of dependencies

---

## How to Run

### Option 1: Run Locally

1. Clone the repository:

```bash
git clone https://github.com/your-username/Multiple-Disease-Prediction-System.git
cd Multiple-Disease-Prediction-System
````

2. Install the required libraries:

```bash
pip install -r requirements.txt
```

3. Start the Streamlit app:

```bash
streamlit run app.py
```

---

### Option 2: Run on Replit

1. Import the project to [Replit](https://replit.com/)
2. Make sure `replit.nix` is set up correctly to run Streamlit
3. Upload the `.pkl` model files
4. Click **Run**

---

## Disclaimer

This application is for educational purposes only and not intended for actual medical use.

```
