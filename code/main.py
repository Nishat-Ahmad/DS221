import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

dataCSV = pd.read_csv('../code/DS221/data/cleaned_vgsales_no_outliers.csv')

# Step 2: Prepare features
# Adding Platform, Genre, Year
dataCSV = pd.get_dummies(dataCSV, columns=['Platform', 'Genre'], drop_first=True)  # Encode categorical variables
X = dataCSV[['EU_Sales', 'JP_Sales', 'Year']]
y = dataCSV['NA_Sales']

# Step 3: Add a constant and check Variance inflation factor.
X = sm.add_constant(X)
vif = pd.DataFrame()
vif['Feature'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

# print(f"Variance Inflation Factor (VIF): \n {vif}")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)   # Split data between training & testing.

enhanced_model = sm.OLS(y_train, X_train).fit()     # Step 5: Fit the enhanced linear regression model

# Step 6: Evaluate model on the test set
y_pred = enhanced_model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5

# Print evaluation metrics
print("\nEnhanced Model Summary:")
print(enhanced_model.summary())

print("\nModel Evaluation Metrics:")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")

# Step 7: Confidence intervals for parameters
confidence_intervals = enhanced_model.conf_int()
print("\nConfidence Intervals for Coefficients:")
print(confidence_intervals)