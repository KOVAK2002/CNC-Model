import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import zscore

def load_features(file_path):
    data = np.load(file_path)
    return data['features'], data['labels']

def plot_distribution(features):
    # Pretpostavimo da su MFCC značajke pohranjene kao redovi u 'features'
    plt.figure(figsize=(15, 6))
    sns.histplot(features.flatten(), kde=True)
    plt.title('Distribucija MFCC značajki')
    plt.xlabel('Vrijednost MFCC-a')
    plt.ylabel('Frekvencija')
    plt.show()

def plot_correlation(features):
    # Pretvaramo značajke u dataframe radi lakše obrade
    import pandas as pd
    df = pd.DataFrame(features, columns=[f'MFCC_{i+1}' for i in range(features.shape[1])])

    plt.figure(figsize=(12, 8))
    corr = df.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Korelacijska matrica MFCC značajki')
    plt.show()

if __name__ == "__main__":
    # Učitavamo značajke i oznake
    features, labels = load_features("features_multiclass.npz")
    
    # Prikaz distribucije MFCC značajki
    plot_distribution(features)
    
    # Prikaz korelacijske matrice MFCC značajki
    plot_correlation(features)
