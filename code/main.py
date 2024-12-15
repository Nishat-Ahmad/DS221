import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import t

# Load the dataset
df = pd.read_csv('D:/code/DS221/data/cleaned_vgsales.csv')

# Step 1: Data Preparation
features = ['Year', 'Platform', 'Genre', 'Publisher']
target = 'Total_Sales'

# Encode categorical variables using one-hot encoding
df_encoded = pd.get_dummies(df[features], drop_first=True)
X = pd.concat([df_encoded, df[['Year']]], axis=1)  # Include numerical features
y = df[target]

# Check for multicollinearity
vif_data = pd.DataFrame()
vif_data["Feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
print("Variance Inflation Factor (VIF):")
print(vif_data)

# Drop features with high VIF if necessary (example placeholder):
# X.drop(columns=['Highly_Correlated_Feature'], inplace=True)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 2: Linear Regression Model
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# Predict on the test set
y_pred_linear = linear_model.predict(X_test)

# Evaluate the linear regression model
def evaluate_model(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return mae, mse, rmse, r2

mae_linear, mse_linear, rmse_linear, r2_linear = evaluate_model(y_test, y_pred_linear)

# Output linear regression evaluation
print("Linear Regression Evaluation Metrics:")
print(f"MAE: {mae_linear:.3f}")
print(f"MSE: {mse_linear:.3f}")
print(f"RMSE: {rmse_linear:.3f}")
print(f"R^2 Score: {r2_linear:.3f}")

# Step 3: Estimation of Linear Regression Parameters
coefficients = linear_model.coef_
intercept = linear_model.intercept_

# Output coefficients and intercept
print("\nLinear Regression Coefficients:")
for feature, coef in zip(X_train.columns, coefficients):
    print(f"{feature}: {coef:.3f}")
print(f"Intercept: {intercept:.3f}")

# Confidence intervals for coefficients
alpha = 0.05  # Significance level
n, p = X_train.shape  # Number of samples, number of features
t_stat = t.ppf(1 - alpha / 2, df=n - p - 1)

# Calculate standard errors
residuals = y_train - linear_model.predict(X_train)
residual_var = np.sum(residuals ** 2) / (n - p - 1)
X_design = np.hstack([np.ones((X_train.shape[0], 1)), X_train])  # Add intercept term

# Use pseudo-inverse to avoid singular matrix error
cov_matrix = np.linalg.pinv(X_design.T @ X_design) * residual_var
std_errors = np.sqrt(np.diag(cov_matrix))

# Output confidence intervals for coefficients
print("\nConfidence Intervals for Coefficients:")
for i, feature in enumerate(['Intercept'] + list(X_train.columns)):
    coef = intercept if i == 0 else coefficients[i - 1]
    ci_lower = coef - t_stat * std_errors[i]
    ci_upper = coef + t_stat * std_errors[i]
    print(f"{feature}: [{ci_lower:.3f}, {ci_upper:.3f}]")
