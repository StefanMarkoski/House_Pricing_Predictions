
# House Price Predictions

Machine learning project for predicting house prices using XGBoost.

## Setup

### 1. Run the Pipeline

First, save the trained model and scaler:

```bash
python run_pipeline.py
```

### 2. Start the API

Run the application locally (using SwaggerUI is recommended):

```bash
uvicorn api.predict:app --reload --port 5000
```
Then open **http://localhost:5000/docs** in your browser.

## API Usage

The API provides a single endpoint that accepts house features in JSON format.

### Request Format

```json
{
    "bedrooms": 3,
    "bathrooms": 2.5,
    "sqft_living": 2000,
    "sqft_lot": 5000,
    "floors": 2,
    "waterfront": 0,
    "view": 0,
    "condition": 3,
    "grade": 7,
    "sqft_above": 1500,
    "sqft_basement": 500,
    "yr_built": 1990,
    "yr_renovated": 0
}
```

The model will return a price prediction based on these parameters.

## Project Contents

- **Notebook**: Experimentation and model comparison sandbox. Includes model selection  showing XGBoost as the best performer.
