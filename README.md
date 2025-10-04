#Heart Disease Prediction Using Machine Learning

📌 Introduction

Cardiovascular diseases are one of the leading causes of death worldwide. Early detection can greatly improve treatment outcomes.
In this project, we use machine learning techniques to predict the presence of heart disease based on medical attributes such as age, blood pressure, cholesterol levels, chest pain type, and more.

The project is implemented in Python using a Jupyter Notebook, with popular libraries like pandas, scikit-learn, seaborn, matplotlib.

🎯 Objectives

Explore and preprocess the heart disease dataset.

Perform exploratory data analysis (EDA) to identify trends and correlations.

Apply feature engineering (scaling, encoding categorical variables).

Build and evaluate machine learning models (K-Nearest Neighbors, Random Forest).

Compare performance and discuss results.

Provide insights for future improvements.

📊 Dataset

The dataset contains 303 samples and 14 attributes, including the target variable:

Feature	Description
| Feature  | Description                                           |
| -------- | ----------------------------------------------------- |
| age      | Age of the patient                                    |
| sex      | Sex (0 = female, 1 = male)                            |
| cp       | Chest pain type                                       |
| trestbps | Resting blood pressure (mm Hg)                        |
| chol     | Serum cholesterol (mg/dl)                             |
| fbs      | Fasting blood sugar > 120 mg/dl (1 = true; 0 = false) |
| restecg  | Resting ECG results                                   |
| thalach  | Maximum heart rate achieved                           |
| exang    | Exercise induced angina (1 = yes; 0 = no)             |
| oldpeak  | ST depression (exercise vs rest)                      |
| slope    | Slope of the ST segment                               |
| ca       | Number of major vessels colored by fluoroscopy        |
| thal     | Thalassemia type                                      |
| target   | Presence of heart disease (1 = yes, 0 = no)           |

🔎 Methodology
1. Data Preprocessing

Missing values check: The dataset had no missing values.

Encoding categorical features: Variables like cp, thal, and slope were converted into dummy variables using pd.get_dummies.

Feature scaling: Numerical features (age, chol, trestbps, thalach, oldpeak) were standardized using StandardScaler for better model performance.

2. Exploratory Data Analysis (EDA)

Correlation heatmaps were generated to see relationships among features.

Count plots were used to check the distribution of the target variable (balanced classes).

Distribution plots showed how features varied for patients with/without heart disease.

3. Model Building

K-Nearest Neighbors (KNN)

Tested values of k from 1 to 20.

Used 10-fold cross-validation to find the best k.

Optimal performance was at k = 12.

Random Forest Classifier

Built a random forest with 10 estimators.

Evaluated using 10-fold cross-validation.

4. Model Evaluation

Metrics used: Cross-validation mean accuracy.

KNN achieved ~84.48% accuracy.

Random Forest achieved ~81.14% accuracy.

📈 Results & Discussion

KNN performed better than Random Forest in this dataset.

Shows that simple models can work effectively with proper preprocessing.

However, in healthcare, relying only on accuracy is insufficient. Metrics like precision, recall, F1-score, ROC-AUC must also be considered.

Limitations: small dataset (303 rows), limited algorithms tested, no external validation.

✅ Conclusion

Machine learning can assist in predicting heart disease with promising accuracy.

The KNN model with k=12 outperformed Random Forest in this study.

With more data and advanced models, prediction accuracy can be further improved.

🚀 Future Scope

Try other models (Logistic Regression, SVM, Gradient Boosting, Neural Networks).

Perform hyperparameter tuning (GridSearchCV, RandomizedSearchCV).

Evaluate with precision, recall, F1-score to avoid false negatives.

Use SHAP/LIME for explainability.

Deploy the best model as a Flask/Django web app for clinicians.

🛠️ Tech Stack

Python 3.x

pandas, numpy (data handling)

matplotlib, seaborn (visualization)

scikit-learn (machine learning models & evaluation)

Jupyter Notebook
