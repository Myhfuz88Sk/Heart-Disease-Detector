# Heart-Disease-Detector
A machine learning-based system that predicts the likelihood of heart disease based on medical input features like age, cholesterol, blood pressure, and more. It helps in early detection and risk assessment for heart disease using trained classification models.


Features:

Predicts the likelihood of heart disease using medical data.

Uses classification algorithms like Logistic Regression, Random Forest, and XGBoost.

Includes exploratory data analysis (EDA) and feature engineering.

Optionally supports a web app for user interaction and prediction.

Project Structure:

Heart-Disease-Detector/
├── dataset/ -> CSV dataset (e.g., heart.csv)
├── model/ -> Trained models (pickle or h5 files)
├── notebooks/ -> Jupyter notebooks for training, EDA, testing
├── app/ -> Flask or Streamlit web app
├── utils/ -> Preprocessing and prediction scripts
├── requirements.txt -> Python dependencies
└── README.txt -> Project documentation

Dataset:

Source: UCI Heart Disease Dataset or Kaggle

Contains patient records with features like:

Age

Sex

Chest Pain Type

Resting Blood Pressure

Cholesterol

Fasting Blood Sugar

Maximum Heart Rate

Exercise-Induced Angina

Target (0 = No Disease, 1 = Disease)

Tech Stack:

Python (Pandas, NumPy, Scikit-learn, XGBoost)

Matplotlib / Seaborn for data visualization

Flask or Streamlit (for frontend app)

Installation:

Clone the project:
git clone https://github.com/yourusername/Heart-Disease-Detector.git

Navigate to the folder:
cd Heart-Disease-Detector

Install the dependencies:
pip install -r requirements.txt

Training the Model:

Run the notebook or script to train the model:

Example:
python train_model.py
or
Open notebooks/train_heart_model.ipynb

Running the Web App:

For Flask:
cd app
python app.py

For Streamlit:
streamlit run app/app.py

You can enter medical input values and get instant prediction results.

Results:

Accuracy: ~85–90% on test data

Balanced precision and recall

Helpful for quick health screening
