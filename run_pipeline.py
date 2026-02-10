from src.data.data_loader import load_raw_data
from src.features.preprocessing import *
from src.models.train import split_train_test, train_models, save_model_and_scaler

def run_pipeline():
    # Load
    df = load_raw_data()

    # Preprocess
    df = drop_unnecessary_columns(df)
    df = drop_missing_values_in_target(df)
    df = impute_bathrooms_with_KNN(df)
    df = impute_waterfront_with_mode(df)
    df = encode_waterfront(df)
    df = encode_condition(df)

    # Scale features
    X, y, scaler = scale_features(df)

    # Split train/test
    X_train, X_test, y_train, y_test = split_train_test(X, y)

    # Train & evaluate
    results, best_model = train_models(X_train, X_test, y_train, y_test)
    
    # Save model and scaler
    save_model_and_scaler(best_model, scaler)

    return results

if __name__ == "__main__":
    results = run_pipeline()
    print(results)