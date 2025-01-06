# Video Game Sales Regression Analysis

## Introduction
In this project, we analyzed video game sales data to estimate linear regression parameters, 
applying techniques to preprocess the data, verify the model, and validate results statistically 
using confidence intervals, hypothesis testing, and p-values. This report emphasizes data 
preprocessing, exploratory data analysis (EDA), statistical validation, and regression analysis.

---

## Data Preprocessing

### Missing Value Handling
- **Year:** Missing values were imputed with the median year (2007) to prevent skewing due to outliers.
- **Publisher:** Missing values were replaced with "Unknown," retaining all rows for analysis.

### Feature Engineering
- **Total_Sales:** Calculated as the sum of `NA_Sales`, `EU_Sales`, `JP_Sales`, and `Other_Sales`, validated against `Global_Sales` for consistency.
- **Age_of_Game:** Created as `2024 - Year` to represent the age of each game numerically.

### Categorical Encoding
- Encoded categorical variables (`Platform`, `Genre`, and `Publisher`) for modeling compatibility using label encoding.
- Smaller publishers were grouped into "Other" (-1), simplifying the analysis.

### Statistical Validation
1. **Central Tendencies:**
   - **Total_Sales:**
     - Mean: ~0.54 million units
     - Median: ~0.17 million units
     - Max: 82.74 million units (notable outlier: Wii Sports)
   - **Year:**
     - Median: 2007
     - Range: 1980–2020

2. **Distribution Analysis:**
   - **Total_Sales:** Right-skewed due to highly successful games.
   - **Year:** Clustered around 2000–2010, reflecting peak growth in the video game industry.

3. **Validation of Transformed Columns:** Verified logical consistency of `Total_Sales` and `Age_of_Game` calculations.

4. **Outlier Check:** Retained notable outliers (e.g., Wii Sports) as they represent real-world phenomena.

---

## Exploratory Data Analysis (EDA)

### Sales Distribution
#### Log Transformation
- **Why Applied:** Sales data often exhibits skewness, with a long tail due to a few games having extremely high sales.
    Applying a log transformation (`log(1+x)`) reduces skewness, making the data more normally distributed.
- **How to Check:**
  - Plot histograms before and after the transformation.
  - Check skewness values (ideal skewness for normality is close to 0).
- **Conclusion:** The log transformation successfully reduced skewness, improving the suitability of the data for linear regression.

#### Visualizations
- **Histogram:** Observed the spread and distribution of sales after transformation.
- **Boxplot:** Identified remaining outliers in regional sales data.

### Correlation Analysis
1. **Pearson Correlation Coefficients:**
   - Strong positive correlations were observed between `Total_Sales`, `NA_Sales`, and other regional sales (`EU_Sales`, `JP_Sales`),
     justifying their inclusion in the regression model.

2. **Heatmap:**
   - Provided a visual summary of correlation strengths and potential multicollinearity issues.
   - Regional sales showed strong intercorrelations, but further tests (e.g., VIF) indicated that multicollinearity was not problematic.

---

## Linear Regression Analysis

### Model Description
- **Selected Features:**
  - `EU_Sales` and `JP_Sales` were included due to their strong correlations with `NA_Sales`.
  - `Year` was included to capture temporal trends in sales.
- **Ordinary Least Squares (OLS):**
  - Chosen for its ability to minimize the sum of squared residuals and provide interpretable coefficient estimates.

### Model Performance
1. **Mean Absolute Error (MAE):**
   - Measures average absolute prediction error.
   - Model’s MAE of 0.112 indicates good prediction accuracy.

2. **Mean Squared Error (MSE) & Root Mean Squared Error (RMSE):**
   - RMSE of 0.134 confirms that the model can predict sales within a narrow margin.

3. **R-Squared & Adjusted R-Squared:**
   - **R-Squared:** 0.845 (84.5% of variance in `NA_Sales` explained by the model).
   - **Adjusted R-Squared:** 0.841 (minimal overfitting).

---

## Estimation of Linear Regression Parameters

### Coefficient Estimation
- **Intercept:** Represents the baseline `NA_Sales` when all predictors are zero.
- **Coefficients:**
  - `EU_Sales`: Strong positive relationship with `NA_Sales`.
  - `JP_Sales`: Positive but weaker relationship (cultural and market differences may explain the weaker influence).
  - `Year`: Negative coefficient suggests declining sales trends over time.

### Variance Inflation Factor (VIF)
- **Why Calculated:** To detect multicollinearity, which inflates standard errors.
- **Conclusion:** VIF < 5 for all variables confirms minimal multicollinearity.

---

## Validation of Parameters

### Method 1: Confidence Intervals
- Confidence intervals provide a range of plausible values for each coefficient.
- All intervals (e.g., `EU_Sales`: [0.76, 0.84]) exclude zero, confirming statistical significance.

### Method 2: Hypothesis Testing
- **p-values < 0.05** indicate statistical significance.
- All predictors were statistically significant (e.g., `EU_Sales`: p < 0.001).

### Method 3: P-Values
- Double-checks the significance of predictors.
- Results aligned with hypothesis testing, confirming statistical significance.

---

## Conclusions for the Model

1. **Model Performance:**
   - High R-Squared (0.845) and low errors (MAE, RMSE) indicate that the model explains a significant portion of variance in `NA_Sales` and makes accurate predictions.

2. **Predictors’ Impact:**
   - `EU_Sales` and `JP_Sales` positively impact `NA_Sales`, with `EU_Sales` having a stronger influence.
   - The negative coefficient for `Year` highlights a declining sales trend over time.

3. **Robustness:**
   - Minimal multicollinearity (VIF < 5) ensures reliable coefficient estimates.
   - Statistical significance of all predictors (p-values < 0.05) indicates strong evidence for their effects on `NA_Sales`.

4. **Limitations:**
   - The model captures linear relationships but may not account for non-linear trends or interactions.
   - Temporal variables like `Year` may require additional context or interaction terms for better interpretation.

---

## License
This project is licensed under the Apache-2.0 license.
