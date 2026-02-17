from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import joblib
import os

def train_models(df, feature_cols, target_col='rating'):
    X = df[feature_cols]
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    models = {
        'LinearRegression': LinearRegression(),
        'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42)
    }
    
    if not os.path.exists("models"):
        os.makedirs("models")
        
    trained_models = {}
    best_score = -float('inf')
    best_model = None
    best_name = ""
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        print(f"{name} R2 Score: {score:.4f}")
        
        trained_models[name] = model
        
        if score > best_score:
            best_score = score
            best_model = model
            best_name = name
            
    print(f"Best Model: {best_name} (R2: {best_score:.4f})")
    joblib.dump(best_model, f"models/best_model.pkl")
    
    return trained_models, X_test, y_test, best_name
