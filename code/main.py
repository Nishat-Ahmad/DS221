import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import t

# Load the dataset
df = pd.read_csv('D:/code/DS221/data/cleaned_vgsales.csv')

# Step 1: Data Preparation
features = ['Year', 'Platform', 'Genre', 'Publisher']
target = 'Total_Sales'

# Encode categorical variables using one-hot encoding
df_encoded = pd.get_dummies(df[features], drop_first=True)

# Handle missing values (fill with 0 or use another strategy)
df_encoded = df_encoded.fillna(0)  # You can also use df_encoded.dropna() to drop rows with missing values

# Include the 'Year' as a numerical feature
X = pd.concat([df_encoded, df[['Year']]], axis=1)
y = df[target]

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 2: Linear Regression Model (with Ridge regularization)
ridge_model = Ridge(alpha=1.0)  # Use Ridge to handle multicollinearity
ridge_model.fit(X_train, y_train)

# Predict on the test set
y_pred_ridge = ridge_model.predict(X_test)

# Evaluate the Ridge regression model
def evaluate_model(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return mae, mse, rmse, r2

mae_ridge, mse_ridge, rmse_ridge, r2_ridge = evaluate_model(y_test, y_pred_ridge)

# Output Ridge regression evaluation metrics
print("Ridge Regression Evaluation Metrics:")
print(f"MAE: {mae_ridge:.3f}")
print(f"MSE: {mse_ridge:.3f}")
print(f"RMSE: {rmse_ridge:.3f}")
print(f"R^2 Score: {r2_ridge:.3f}")

# Step 3: Feature Importance using Random Forest
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)

# Get feature importances and plot
feature_importances = pd.Series(rf_model.feature_importances_, index=X_train.columns)
sorted_importances = feature_importances.sort_values(ascending=False)

plt.figure(figsize=(10, 6))
sorted_importances.plot(kind='bar', color='skyblue')
plt.title('Feature Importances (Random Forest)')
plt.ylabel('Importance')
plt.xlabel('Features')
plt.show()

# Step 4: Random Forest Model Evaluation
y_pred_rf = rf_model.predict(X_test)
mae_rf, mse_rf, rmse_rf, r2_rf = evaluate_model(y_test, y_pred_rf)

# Output Random Forest evaluation
print("Random Forest Evaluation Metrics:")
print(f"MAE: {mae_rf:.3f}")
print(f"MSE: {mse_rf:.3f}")
print(f"RMSE: {rmse_rf:.3f}")
print(f"R^2 Score: {r2_rf:.3f}")

# Step 5: Estimation of Linear Regression Parameters (with Confidence Intervals)
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# Get the coefficients and intercept of the linear regression model
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
epsilon = 1e-5  # Small value for numerical stability
cov_matrix = np.linalg.inv(X_design.T @ X_design + epsilon * np.eye(X_design.shape[1])) * residual_var
std_errors = np.sqrt(np.diag(cov_matrix))

# Output confidence intervals for coefficients
print("\nConfidence Intervals for Coefficients:")
for i, feature in enumerate(['Intercept'] + list(X_train.columns)):
    ci_lower = coefficients[i - 1] - t_stat * std_errors[i] if i > 0 else intercept - t_stat * std_errors[i]
    ci_upper = coefficients[i - 1] + t_stat * std_errors[i] if i > 0 else intercept + t_stat * std_errors[i]
    print(f"{feature}: [{ci_lower:.3f}, {ci_upper:.3f}]")
