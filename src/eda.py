import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd

def perform_eda(df):
    if not os.path.exists("plots"):
        os.makedirs("plots")
        
    plt.figure(figsize=(8, 5))
    sns.histplot(df['rating'], bins=20, kde=True)
    plt.title("Rating Distribution")
    plt.savefig("plots/rating_distribution.png")
    plt.close()
    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='cost', y='rating', data=df, alpha=0.5)
    plt.title("Cost vs Rating")
    plt.savefig("plots/cost_vs_rating.png")
    plt.close()
    
    plt.figure(figsize=(6, 4))
    sns.countplot(x='online_order', data=df)
    plt.title("Online Order Availability")
    plt.savefig("plots/online_order_count.png")
    plt.close()
    
    plt.figure(figsize=(6, 4))
    sns.boxplot(x='book_table', y='rating', data=df)
    plt.title("Table Booking vs Rating")
    plt.savefig("plots/booking_vs_rating.png")
    plt.close()
