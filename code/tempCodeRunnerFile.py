import pandas as pd
import statsmodels.api as sm

# Load your dataset
data = pd.read_csv("D:/code/DS221/data/cleaned_vgsales_no_outliers.csv")  # Replace with your actual file path

# Step 1: Correlation analysis for NA_Sales, EU_Sales, and JP_Sales
regional_sales_correlation = data[['NA_Sales', 'EU_Sales', 'JP_Sales']].corr()
print("Correlation Matrix:")
print(regional_sales_correlation)

# Step 2: Prepare data for linear regression
X_regional = data[['EU_Sales', 'JP_Sales']]  # Independent variables
y_na_sales = data['NA_Sales']  # Dependent variable

# Add a constant to the independent variables (for the intercept)
X_regional = sm.add_constant(X_regional)

# Step 3: Fit the linear regression model
regional_model = sm.OLS(y_na_sales, X_regional).fit()

# Step 4: Display model summary
print("\nLinear Regression Model Summary:")
print(regional_model.summary())

# Step 5: Extract confidence intervals for parameters
confidence_intervals = regional_model.conf_int()
print("\nConfidence Intervals for Coefficients:")
print(confidence_intervals)

# Step 6: Evaluate model metrics
r_squared = regional_model.rsquared
mae = abs(regional_model.resid).mean()
mse = (regional_model.resid ** 2).mean()
rmse = mse ** 0.5

print("\nModel Evaluation Metrics:")
print(f"R-squared: {r_squared}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
