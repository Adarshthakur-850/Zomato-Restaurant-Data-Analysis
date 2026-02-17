import sys
import os

sys.path.append(os.path.join(os.getcwd(), 'src'))

from src.data_loader import load_data
from src.preprocessing import preprocess_data
from src.eda import perform_eda
from src.feature_engineering import engineer_features
from src.model_trainer import train_models
from src.evaluation import evaluate_models
from src.visualization import plot_feature_importance

def main():
    print("Starting Zomato Restaurant Data Analysis Pipeline...")
    
    df = load_data()
    
    df = preprocess_data(df)
    
    perform_eda(df)
    
    df, features = engineer_features(df)
    print(f"Features used: {features}")
    
    models, X_test, y_test, best_name = train_models(df, features)
    
    evaluate_models(models, X_test, y_test)
    
    if 'RandomForest' in models:
        plot_feature_importance(models['RandomForest'], features)
        
    print("Pipeline completed successfully.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Pipeline Failed: {e}")
