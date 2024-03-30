# CATBOOST WITH MORGAN FP

import pandas as pd
from catboost import CatBoostRegressor
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

# df = pd.read_csv('train.csv')
df = pd.read_csv('morg3.csv')
df['MtbH37Rv-Inhibition'] = df['MtbH37Rv-Inhibition'].apply(lambda x: max(x, 0))

lower_bound = df['MtbH37Rv-Inhibition'].quantile(0.05)
upper_bound = df['MtbH37Rv-Inhibition'].quantile(0.95)
undersample_ratio = 1/9

print(f"The range containing 90% of the data is: {lower_bound} to {upper_bound}")

df_target_range = df[(df['MtbH37Rv-Inhibition'] >= lower_bound) & (df['MtbH37Rv-Inhibition'] <= upper_bound)]
df_outside_range = df[(df['MtbH37Rv-Inhibition'] < lower_bound) | (df['MtbH37Rv-Inhibition'] > upper_bound)]

undersampled_size = int(len(df_target_range) * undersample_ratio)
df_target_range_undersampled = df_target_range.sample(n=undersampled_size, random_state=42)
df_undersampled = pd.concat([df_outside_range, df_target_range_undersampled])

df = df_undersampled

y = df['MtbH37Rv-Inhibition']
X = df.drop(columns=['SMILES', 'MtbH37Rv-Inhibition', 'Active', 'MoleculeName', 'RDKitMol', 'molStripped', 'strippedSalts'])

# Ensure all columns in X are numeric
for col in X.columns:
    if X[col].dtype == 'object':
        X = X.drop(columns=[col])
        
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = CatBoostRegressor(
    iterations=894,
    depth=10,
    learning_rate=0.14102831275341693,
    l2_leaf_reg=6.158019820677719,
    border_count=17,
    loss_function='RMSE',
    random_seed=42,
    verbose=0
)


model.fit(X_train, y_train)

preds = model.predict(X_test)
r2 = r2_score(y_test, preds)
rmse = np.sqrt(mean_squared_error(y_test, preds))

print(f"R squared: {r2}")
print(f"RMSE: {rmse}")
