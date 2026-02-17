import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def preprocess_data(df):
    print("Preprocessing data...")
    
    # Drop duplicates and irrelevant columns
    df.drop_duplicates(inplace=True)
    cols_to_drop = ['url', 'address', 'phone', 'reviews_list', 'menu_item', 'listed_in(type)', 'listed_in(city)']
    # Depending on the specific csv version, names might differ, handling common ones or just generic cleaning
    # For the specific rashida048 dataset, common cols are: 
    # url, address, name, online_order, book_table, rate, votes, phone, location, rest_type, dish_liked, cuisines, approx_cost(for two people), reviews_list, menu_item, listed_in(type), listed_in(city)
    
    # Let's keep it robust by checking columns existence
    existing_cols = [c for c in cols_to_drop if c in df.columns]
    df.drop(columns=existing_cols, inplace=True)
    
    # Rename columns based on actual dataset header
    rename_map = {
        'Restaurant Name': 'name',
        'City': 'city',
        'Cuisines': 'cuisines',
        'Average Cost for two': 'cost',
        'Has Table booking': 'book_table',
        'Has Online delivery': 'online_order',
        'Aggregate rating': 'rating',
        'Votes': 'votes',
        'Locality': 'location'
    }
    df.rename(columns=rename_map, inplace=True)
    
    # Cleaning 'rating'
    # In this dataset, 'rating' (Aggregate rating) is typically a float, but let's be safe
    if 'rating' in df.columns:
        df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
        df['rating'].fillna(df['rating'].mean(), inplace=True)
        
    # Cleaning 'cost'
    if 'cost' in df.columns:
        if df['cost'].dtype == 'object':
            df['cost'] = df['cost'].astype(str).apply(lambda x: x.replace(',', ''))
        df['cost'] = pd.to_numeric(df['cost'], errors='coerce')
        df['cost'].fillna(df['cost'].median(), inplace=True)
        
    # Handle Categorical
    # Encoding binary
    for col in ['online_order', 'book_table']:
        if col in df.columns:
            df[col] = df[col].map({'Yes': 1, 'No': 0})
            
    # Dropping remaining nulls for simplicity in this artifact
    df.dropna(inplace=True)
    
    return df
