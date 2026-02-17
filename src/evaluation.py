from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

def evaluate_models(models, X_test, y_test):
    if not os.path.exists("plots"):
        os.makedirs("plots")
        
    metrics = []
    
    plt.figure(figsize=(10, 6))
    
    for name, model in models.items():
        y_pred = model.predict(X_test)
        
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        metrics.append({
            'Model': name,
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2
        })
        
        plt.scatter(y_test, y_pred, alpha=0.5, label=name)
        
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel("Actual Rating")
    plt.ylabel("Predicted Rating")
    plt.title("Actual vs Predicted Ratings")
    plt.legend()
    plt.savefig("plots/actual_vs_predicted.png")
    plt.close()
    
    metrics_df = pd.DataFrame(metrics)
    print("\nModel Evaluation:")
    print(metrics_df)
    
    return metrics_df
