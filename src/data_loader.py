import os
import requests
import pandas as pd

DATA_URL = "https://raw.githubusercontent.com/dsrscientist/dataset4/main/zomato.csv"
DATA_PATH = os.path.join("data", "zomato.csv")

def load_data():
    if not os.path.exists("data"):
        os.makedirs("data")
        
    if not os.path.exists(DATA_PATH):
        print(f"Downloading data from {DATA_URL}...")
        try:
            response = requests.get(DATA_URL)
            response.raise_for_status()
            with open(DATA_PATH, "wb") as f:
                f.write(response.content)
            print("Download complete.")
        except Exception as e:
            print(f"Error downloading data: {e}")
            raise
            
    print("Loading dataset...")
    # Trying different encodings because zomato dataset often has encoding issues
    try:
        df = pd.read_csv(DATA_PATH, encoding='latin-1')
    except:
        df = pd.read_csv(DATA_PATH, encoding='utf-8')
        
    print(f"Dataset shape: {df.shape}")
    return df
