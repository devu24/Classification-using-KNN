# Diabetes Prediction Using Machine Learning

This project demonstrates the use of machine learning techniques to predict diabetes based on various health metrics. The analysis is performed using Python and several popular data science libraries.

## Dataset

The dataset used in this project is the Pima Indians Diabetes Database. It includes several health-related features such as:

- Pregnancies
- Glucose
- Blood Pressure
- Skin Thickness
- Insulin
- BMI (Body Mass Index)
- Diabetes Pedigree Function
- Age
- Outcome (1 indicates diabetes, 0 indicates no diabetes)

## Data Preprocessing

1. Data Cleaning: The initial step involved handling missing values. Zero values in certain columns (Glucose, BloodPressure, SkinThickness, Insulin, BMI) were replaced with NaN and then imputed using mean or median values.

2. Exploratory Data Analysis (EDA): Various visualization techniques were used to understand the data distribution and relationships between features.

3. Feature Scaling: StandardScaler was applied to normalize the feature set.

## Model Development

The primary model used in this project is K-Nearest Neighbors (KNN). The process included:

1. Train-Test Split: The data was split into training and testing sets.

2. Model Training: A KNN classifier was trained on the data.

3. Hyperparameter Tuning: GridSearchCV was used to find the optimal number of neighbors (k) for the KNN model.

## Model Evaluation

Several techniques were used to evaluate the model's performance:

1. Confusion Matrix: Visualized using a heatmap to show true positives, true negatives, false positives, and false negatives.

2. Classification Report: Provides precision, recall, and F1-score for each class.

3. ROC Curve and AUC Score: Used to assess the model's ability to distinguish between classes.

## Key Findings

- The optimal number of neighbors for the KNN model was found to be 25.
- The model achieved an accuracy of approximately 77.2% after hyperparameter tuning.
- The ROC-AUC score of about 0.819 indicates good discriminative ability.

## Visualizations

The project includes several visualizations:

- Histograms of feature distributions
- Pair plots to show relationships between features
- Heatmap of feature correlations
- Confusion matrix heatmap
- ROC curve

## Future Work

Potential areas for improvement and expansion:

1. Try other machine learning algorithms (e.g., Random Forest, SVM)
2. Feature engineering to create more predictive variables
3. Handling class imbalance in the dataset
4. Deployment of the model as a web application

## Dependencies

- Python 3.x
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

## How to Run

1. Ensure all dependencies are installed.
2. Load the Jupyter Notebook or Python script.
3. Run all cells/script to see the analysis and results.

This project demonstrates a complete machine learning workflow from data preprocessing to model evaluation, providing insights into diabetes prediction based on health metrics.
