import pandas as pd
import os

def load_raw_data(path='data/raw/house_prices.csv'):
    """
    Load the raw CSV file containing the data and return
    pd.DataFrame
    """

    if not os.path.exists(path):
        raise FileNotFoundError(f"This files does not exist: {path}")

    df = pd.read_csv(path)

    return df