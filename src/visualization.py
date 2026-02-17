import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_feature_importance(model, feature_names):
    if not os.path.exists("plots"):
        os.makedirs("plots")
        
    if hasattr(model, 'feature_importances_'):
        plt.figure(figsize=(10, 8))
        importances = model.feature_importances_
        indices = importances.argsort()[::-1]
        
        plt.title("Feature Importances")
        plt.bar(range(len(importances)), importances[indices], align="center")
        plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
        plt.tight_layout()
        plt.savefig("plots/feature_importance.png")
        plt.close()
