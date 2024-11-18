# GDSC Drug Response Prediction with Machine Learning

This repository demonstrates a comprehensive approach to predicting drug responses of cancer cell lines using machine learning techniques. The dataset contains information on cancer cell lines, drug responses, and various biological features, with the target variable being the **LN_IC50**, which quantifies the drug's half-maximal inhibitory concentration. This project explores different machine learning models to predict the drug response and evaluates their performance using several metrics.

## Project Overview

The goal of this project is to predict the drug response across various cancer cell lines. The models are evaluated based on how well they predict the target column, **LN_IC50**, using machine learning algorithms. This process involves data preprocessing, model training, and comparison of results to identify the best performing model.

### Key Steps in the Project:
1. **Data Preprocessing**:
   - **Handling Missing Data**: Missing values in the dataset are addressed through imputation or removal.
   - **Encoding Categorical Features**: Non-numeric variables (such as `GDSC Tissue descriptor 1`, `Cancer Type`, etc.) are encoded into numerical representations.
   - **Feature Scaling**: Standardization and normalization of numeric features to ensure uniform scaling for model training.

2. **Feature Engineering**:
   - Important features are selected based on their relevance to the target variable, **LN_IC50**. 
   - Additional transformations are applied to enhance the dataset for better model performance.

3. **Model Training**:
   - Multiple machine learning algorithms are trained, including **Random Forest Regressor**, **Linear Regression**, and **XGBoost**.
   
4. **Model Evaluation**:
   - Models are evaluated using performance metrics such as **Mean Squared Error (MSE)**, **Root Mean Squared Error (RMSE)**, **Mean Absolute Error (MAE)**, and **R² score**.
   - A comparison of these models helps to identify which one provides the most accurate predictions for the **LN_IC50**.

5. **Model Comparison**:
   - The results of each model are compared to determine the most effective algorithm for predicting drug responses.
   - Visualizations, such as residual plots and performance comparison charts, are used to demonstrate the results.

---

## Key Technologies Used

- **Python**: The main programming language for this project.
- **Pandas**: For data manipulation and preprocessing.
- **Scikit-learn**: For machine learning model training and evaluation.
- **XGBoost**: For gradient boosting techniques.
- **Matplotlib/Seaborn**: For creating visualizations and performance plots.

---

## Data Description

The dataset contains the following columns:

1. **COSMIC_ID**: Unique identifier for each cell line.
2. **CELL_LINE_NAME**: Name of the cell line.
3. **TCGA_DESC**: Description of the TCGA label for cancer types.
4. **DRUG_ID**: Unique identifier for each drug.
5. **DRUG_NAME**: Name of the drug.
6. **LN_IC50**: Logarithm of the half-maximal inhibitory concentration (target column).
7. **AUC**: Area Under the Curve representing the drug's efficacy.
8. **Z_SCORE**: Standardized value for the drug's response.
9. **GDSC Tissue descriptors**: Tissue type descriptors for cancer.
10. **Cancer Type (matching TCGA label)**: Specific cancer type for the cell line.
11. **Microsatellite instability Status (MSI)**: MSI status for cancer cells.
12. **Screen Medium**: Medium used for drug screening.
13. **Growth Properties**: Properties describing cell growth.
14. **CNA**: Copy number alterations for the cell lines.
15. **Gene Expression**: Gene expression data for cell lines.
16. **Methylation**: DNA methylation data for cell lines.
17. **TARGET**: The putative molecular target for the drug.
18. **TARGET_PATHWAY**: The biological pathway related to the target.

---

## Results and Analysis

### Model Evaluation

#### 1. **Random Forest Regressor Evaluation**:
   - **MSE**: 0.08252052733612429
   - **RMSE**: 0.28726386360996453
   - **MAE**: 0.1586659554659184
   - **R²**: 0.9897738225597844
   
   The **Random Forest Regressor** model showed exceptional performance with an R² score of 0.99, indicating that it explains most of the variance in the target variable. The low MSE, RMSE, and MAE further confirm that this model provides highly accurate predictions.

#### 2. **Linear Regression Evaluation**:
   - **MSE**: 2.498517638627196
   - **RMSE**: 1.5806699967504907
   - **MAE**: 1.0896249837375305
   - **R²**: 0.690376618583177
   
   The **Linear Regression** model, while still providing reasonable results, performed less well compared to Random Forest. The lower R² score and higher error metrics suggest that linear relationships are not sufficient to capture the complexities of the data.

#### 3. **XGBoost Evaluation**:
   - **MSE**: 0.1314089752598891
   - **RMSE**: 0.362503758959668
   - **MAE**: 0.25991761196150165
   - **R²**: 0.9837154276441923
   
   **XGBoost** performed well with an R² score of 0.98, showing strong predictive power. However, it did not outperform Random Forest in terms of the R² score and the error metrics.

### Model Comparison

A comparison of the models reveals that **Random Forest Regressor** outperforms both **Linear Regression** and **XGBoost** in terms of accuracy. The results suggest that ensemble methods, like Random Forest, are particularly well-suited for this type of regression task, where complex, non-linear relationships exist between the features and the target variable.

### Visualizations

- **Performance Comparison Bar Plot**: Visual comparison of MSE, RMSE, MAE, and R² score for each model.
- **Residual Plots**: Plots showing the residuals for each model to evaluate the error distribution. Random Forest and XGBoost showed more normally distributed residuals, indicating better generalization.

---

## Conclusion

- The **Random Forest Regressor** emerged as the best-performing model for predicting drug responses with an impressive R² score of **0.99**. This model is highly effective in predicting **LN_IC50**, showing minimal error in predictions.
- **XGBoost** also performed well, with an R² score of **0.98**, but did not surpass Random Forest in overall performance.
- **Linear Regression** underperformed in comparison, with a much lower R² score of **0.69**, suggesting that linear models are not suitable for this type of dataset.

Given the results, **Random Forest Regressor** is recommended as the preferred model for predicting drug responses in cancer cell lines.

