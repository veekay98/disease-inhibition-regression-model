import pandas as pd
from catboost import CatBoostRegressor
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import mean_squared_error, r2_score
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# List of alpha values for L1 regularization (Lasso)
# alpha_values = [0.01, 0.05, 0.1, 0.2, 0.5]  # Adjust as needed
alpha_values = [0.0001, 0.00001, 0.00005]

# Initialize empty lists to store results
r_squared_values = []
mse_values = []

df = pd.read_csv('train.csv')
X = df.drop(columns=['SMILES', 'MtbH37Rv-Inhibition', 'Active', 'MoleculeName', 'RDKitMol', 'molStripped', 'strippedSalts'])
y = df['MtbH37Rv-Inhibition']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Loop through each alpha value
for alpha in alpha_values:
    # Feature selection with Lasso
    lasso = Lasso(alpha=alpha, random_state=42)  # Set random_state for reproducibility
    selector = SelectFromModel(lasso)
    X_new = selector.fit_transform(X_train, y_train)

    X_new = selector.fit_transform(X_train, y_train)

    # Define model parameters (consider hyperparameter tuning)
    params = {
        'iterations': 894,
    'depth': 10,
    'learning_rate': 0.14102831275341693,  # commented out as hyperparameter tuning is recommended
    'l2_leaf_reg': 6.158019820677719,   # commented out as hyperparameter tuning is recommended
    'border_count': 17
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

# Create a DataFrame to display results
results_df = pd.DataFrame({'alpha': alpha_values, 'R-squared': r_squared_values, 'MSE': mse_values})
print(results_df)
