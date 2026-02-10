from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import root_mean_squared_error, mean_squared_error, r2_score
import pickle
import os

def split_train_test(X, y, test_size=0.2, random_state=42):
    """
    Split the dataset into train and test sets.
    Returns:
        X_train, X_test, y_train, y_test
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test


def train_models(X_train, X_test, y_train, y_test):
    """
    Train the best performing model (XGBoost) and return evaluation metrics.
    Returns:
        results: dict with MSE, RMSE, R2
        model: trained XGBoost model
    """
    model = XGBRegressor(n_estimators=1000, learning_rate=0.01, random_state=42)
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    results = {'MSE': mse, 'RMSE': rmse, 'R2': r2}
    
    return results, model


def save_model_and_scaler(model, scaler, model_path='models/xgb_model.pkl', scaler_path='models/scaler.pkl'):
    """
    Save the trained model and scaler to disk
    """
    os.makedirs('models', exist_ok=True)
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)