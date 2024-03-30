import pandas as pd
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel, SelectKBest, f_regression
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# List of hyperparameters for each method
k_values = [25, 50, 75, 100, 125, 150, 175, 206]  # SelectKBest
n_estimators = [100, 200, 300, 400, 500]  # Random Forest
alpha_values = [0.01, 0.05, 0.1, 0.2, 0.5]  # Lasso

# Initialize empty lists to store results
r_squared_values = []
mse_values = []

# Load data (replace with your data loading logic)
df = pd.read_csv('train.csv')
X = df.drop(columns=['SMILES', 'MtbH37Rv-Inhibition', 'Active', 'MoleculeName', 'RDKitMol', 'molStripped', 'strippedSalts'])
y = df['MtbH37Rv-Inhibition']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Loop through each feature selection method
for method_name, hyperparameter_list, selection_method in [
    ('SelectKBest', k_values, SelectKBest(f_regression)),
    ('Random Forest', n_estimators, SelectFromModel(RandomForestRegressor(random_state=42))),
    ('Lasso', alpha_values, SelectFromModel(Lasso(random_state=42))),
]:

    # Loop through each hyperparameter value
    for hyperparameter in hyperparameter_list:
        # Feature selection with chosen method
        selector = selection_method(hyperparameter=hyperparameter)
        X_new = selector.fit_transform(X_train, y_train)

        # Define CatBoost model parameters (consider hyperparameter tuning)
        params = {
            'iterations': 894,
            'depth': 10,
            # ... other parameters as needed
        }

        # Create CatBoostRegressor model
        model = CatBoostRegressor(**params)

        # Train the model
        model.fit(X_new, y_train)

        # Make predictions
        y_pred = model.predict(selector.transform(X_test))  # Transform test data

        # Evaluate model performance
        mse = mean_squared_error(y_test, y_pred)
        r_squared = r2_score(y_test, y_pred)

        # Store results
        r_squared_values.append(r_squared)
        mse_values.append(mse)

# Create DataFrame with method, hyperparameter, and performance metrics
results_df = pd.DataFrame({
    'Method': [method_name] * (len(k_values) + len(n_estimators) + len(alpha_values)),
    'Hyperparameter': k_values + n_estimators + alpha_values,
    'R-squared': r_squared_values,
    'MSE': mse_values
})
print(results_df)
