import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

def drop_unnecessary_columns(df):
    """
    Dropping the id and date since they are unnecessary for this task
    return a pd.DataFrame
    """

    drop_columns = ['id','date','sqft_living15','sqft_lot15','lat','long']

    df = df.drop(columns=drop_columns,errors='ignore')

    return df


def drop_missing_values_in_target(df):
    """
    Drop the missing values in target since we almost never want to 
    impute missing values in the target column
    """
    target = 'price'

    df = df.dropna(subset=[target])

    return df


def impute_bathrooms_with_KNN(df,n_neighbors=5):
    """
    Since I caught a good correlation between bathrooms column and 
    a couple of other columns I will use KNNImputer here to capitalize
    on those correlations for more precise imputing
    """
    numeric_columns = ['bathrooms','price','bedrooms','sqft_living','grade']
    imputer = KNNImputer(n_neighbors=n_neighbors)
    df[numeric_columns] = imputer.fit_transform(df[numeric_columns])
    return df


def impute_waterfront_with_mode(df):
    """
    I feel like this might be a very important feature beacause a house
    having a waterfront should have bigger price than a house who doesnt
    have it, we will be imputing it with mode (check notebook), since more houses
    dont have any waterfront than houses who have it. 
    I feel like mode here should be perfect.
    """
    df['waterfront'] = df['waterfront'].fillna(df['waterfront'].mode()[0])

    return df

def encode_waterfront(df):
    """
    We will be encoding waterfront with LabelEncoder so we can retain some of the
    bias that the model retain from this column, this is important beacause we want
    to make the model think that the houses with waterfront should be more expensive
    (all things considered).
    """
    encoder = LabelEncoder()
    df['waterfront'] = encoder.fit_transform(df['waterfront'])
    return df


def encode_condition(df):
    """
    Same thing here as in waterfront, condition matters in price, we want to have bias
    so it can properly affect and make the model think that the bigger(since its numeric)
    condition the higher the house price will be. 
    """
    df['condition'] = df['condition'].map({
        "Poor" : 0,
        "Fair" : 1,
        "Average" : 2,
        "Good" : 3,
        "Very Good" : 4
    }).astype(int)

    return df


def scale_features(df):
    """
    Scale X
    Target:y
    Features:x
    """
    scaler = MinMaxScaler()

    X = df.drop(columns='price')
    y = df['price']

    X = scaler.fit_transform(X)
    return X,y,scaler