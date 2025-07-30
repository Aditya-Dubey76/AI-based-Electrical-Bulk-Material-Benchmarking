import pandas as pd
import pickle
from sklearn.ensemble import GradientBoostingRegressor
from catboost import CatBoostRegressor
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load dataset
data = pd.read_csv('Your dataset')

# Features (Independent Variables)
X = data[['Plot size (sq m)', 'Total consumer', 'Total Equipment (Mechanical + Electrical)']]

# Multi-Output Targets (Dependent Variables)
y = data[[
    'Total Power Cable',
    'LCS',
    'Total Light Fixture',
    'Total Cable Tray',
    'Total Earthing Material (GI Strip)',
    'Total Earthing Material (Equipment Earthing)',
]]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features for Ridge
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Models Dictionary
models = {
    'Gradient Boosting': MultiOutputRegressor(GradientBoostingRegressor(n_estimators=100, random_state=42)),
    'CatBoost': MultiOutputRegressor(CatBoostRegressor(verbose=0, random_state=42)),
    'Ridge': MultiOutputRegressor(Ridge(alpha=1.0)),
}

trained_models = {}
accuracies = {}

for name, model in models.items():
    if name == 'Ridge':
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    elif name == 'CatBoost':
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    
    acc = r2_score(y_test, y_pred, multioutput='uniform_average')  # Average R2 across all outputs
    trained_models[name] = model
    accuracies[name] = acc

# Save all models and scaler
with open('forecast_multioutput_models.pkl', 'wb') as f:
    pickle.dump({'models': trained_models, 'scaler': scaler, 'accuracies': accuracies}, f)

print("All multi-output models trained and saved successfully.")
