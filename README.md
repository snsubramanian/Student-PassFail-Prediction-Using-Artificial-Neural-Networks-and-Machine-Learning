# Student Pass/Fail Prediction System

## About the Project
This project is a machine learning web application that predicts whether a student is likely to pass or fail based on academic performance and attendance data. It uses a Random Forest classifier and an interactive Streamlit interface to generate predictions.

## Features
- Student academic performance and attendance analysis
- Data preprocessing and feature scaling
- Pass/fail classification using Random Forest
- Interactive Streamlit web interface
- Model evaluation using accuracy, precision, recall, F1-score, and confusion matrix
- Saved model for reuse with Joblib

## Technologies Used
- Python
- Pandas and NumPy
- Scikit-learn
- Random Forest Classifier
- Streamlit
- Joblib

## Dataset
The project uses a student performance dataset containing academic marks, previous GPA, and attendance information.

The target variable is created using the assumption that:
- **Pass (1):** Final marks are greater than or equal to 35
- **Fail (0):** Final marks are below 35

Note: The passing threshold is an assumption used for this project and may not represent the dataset's official grading policy.

## Model Performance
The Random Forest model achieved **66.67% test accuracy** on the current train-test split. Performance may vary depending on the data split and model configuration.

## Project Structure
```text
Student-Pass-Fail-Prediction/
├── app.py
├── student_pass_fail_model.pkl
└── README.md
```

## Installation and Usage

### 1. Clone the Repository
```bash
git clone <your-github-repository-url>
cd Student-Pass-Fail-Prediction
```

### 2. Install Dependencies
```bash
pip install streamlit pandas numpy scikit-learn joblib
```

### 3. Run the Application
```bash
streamlit run app.py
```

The application will open in your browser, allowing you to enter student details and view the predicted result.

## Future Improvements
- Improve model performance through hyperparameter tuning
- Evaluate models using cross-validation
- Add visualizations for student performance
- Improve the user interface and prediction reports
- Test the model on larger datasets

## Disclaimer
This project is developed for educational purposes. Predictions are estimates and should not be used as the sole basis for academic decisions.
