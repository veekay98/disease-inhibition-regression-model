import pandas as pd
from catboost import CatBoostRegressor
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# List of hyperparameters for PCA
n_components_list = [50, 75, 100, 150 , 200]  # Adjust as needed
# n_components_list = [5, 10, 15, 20, 25]  # Adjust as needed

# Initialize empty lists to store results
r_squared_values = []
mse_values = []

# Load data (replace with your data loading logic)
df = pd.read_csv('morg.csv')
X = df.drop(columns=['SMILES', 'MtbH37Rv-Inhibition', 'Active', 'MoleculeName', 'RDKitMol', 'molStripped', 'strippedSalts'])
y = df['MtbH37Rv-Inhibition']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Loop through each number of components
for n_components in n_components_list:
    # PCA for dimensionality reduction
    pca = PCA(n_components=n_components, random_state=42)  # Set random_state for reproducibility
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    # Define CatBoost model parameters (consider hyperparameter tuning)
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
    model.fit(X_train_pca, y_train)

    # Make predictions
    y_pred = model.predict(X_test_pca)

    # Evaluate model performance
    mse = mean_squared_error(y_test, y_pred)
    r_squared = r2_score(y_test, y_pred)

    # Store results
    r_squared_values.append(r_squared)
    mse_values.append(mse)

# Create DataFrame with method (PCA), parameter (number of components), and performance metrics
results_df = pd.DataFrame({
    'Method': ['PCA'] * len(n_components_list),
    'Hyperparameter': n_components_list,
    'R-squared': r_squared_values,
    'MSE': mse_values
})
print(results_df)
