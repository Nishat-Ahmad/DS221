import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

dataCSV = pd.read_csv('../code/DS221/data/cleaned_vgsales_no_outliers.csv')

# Preparing features, adding Platform, Genre, Year
dataCSV = pd.get_dummies(dataCSV, columns=['Platform', 'Genre'], drop_first=True)
X = dataCSV[['EU_Sales', 'JP_Sales', 'Year']]
y = dataCSV['NA_Sales']

# Adding a constant (so that the line is not forced to go through y=0, x=0)
# Check Variance inflation factor (checks multicollineary).
X = sm.add_constant(X)
vif = pd.DataFrame()
vif['Feature'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
print(vif)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
enhanced_model = sm.OLS(y_train, X_train).fit()

# Evaluating model on the test set
y_pred = enhanced_model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5

# Enhanced Model Summary and Coefficient Table Display
print("=" * 30)
print("\nEnhanced Model Summary:")
print(f"Dependent Variable: {enhanced_model.model.endog_names} \nModel: Ordinary least squares(OLS)")
print(f"Method: Least Squares \nNumber of Observations: {int(enhanced_model.nobs)}")
print(f"R-squared: {enhanced_model.rsquared:.3f} \nAdjusted R-squared: {enhanced_model.rsquared_adj:.3f}")
print("=" * 78)
print("                 coef    std err          t      P>|t|      [0.025      0.975]")
print("-" * 78)
coef_table = enhanced_model.summary2().tables[1]
for _, row in coef_table.iterrows():
    print(f"{row.name:<12} {row['Coef.']:>9.4f} {row['Std.Err.']:>10.4f} {row['t']:>10.4f} {row['P>|t|']:>10.4f} {row['[0.025']:>12.4f} {row['0.975]']:>12.4f}")
print("=" * 78)

print("\nModel Evaluation Metrics (in millions) (the lower the better):")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")

confidence_intervals = enhanced_model.conf_int()
print("\nConfidence Intervals for Coefficients:")
print(confidence_intervals)