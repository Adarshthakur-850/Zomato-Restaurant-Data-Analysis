import pandas as pd
from sklearn.preprocessing import LabelEncoder

def engineer_features(df):
    
    # Feature: Cuisine Count
    if 'cuisines' in df.columns:
        df['cuisine_count'] = df['cuisines'].astype(str).apply(lambda x: len(x.split(',')))
    else:
        df['cuisine_count'] = 1
        
    # Feature: Cost per Rating
    # Avoid division by zero
    df['cost_per_rating'] = df['cost'] / (df['rating'] + 0.1)
    
    # Encode Categorical
    # Ideally should use OneHotEncoding for nominal, but sticking to LabelEncoding for simplicity/speed regarding prompt
    le = LabelEncoder()
    categorical_cols = ['name', 'city', 'location', 'rest_type', 'cuisines']
    
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype(str)
            df[col] = le.fit_transform(df[col])
            
    # Select features for modeling
    features = ['online_order', 'book_table', 'votes', 'cost', 'cuisine_count', 'cost_per_rating']
    # Add encoded cols if they exist
    for col in categorical_cols:
        if col in df.columns:
            features.append(col)
            
    # Filter only available columns
    features = [f for f in features if f in df.columns]
    
    return df, features
