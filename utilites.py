import unicodedata
import pandas as pd
import numpy as np

def to_ascii(text):
    return unicodedata.normalize('NFKD', str(text)).encode('ascii', 'ignore').decode('ascii')

def clean_text(text):
    if pd.isna(text):
        return text
    text = str(text)
    text = text.replace('\t', ' ').replace('\n', ' ').replace('\r', ' ')
    # Remove multiple spaces
    text = ' '.join(text.split())
    # Strip leading/trailing whitespace
    text = text.strip()
    text = to_ascii(text)
    return text

from sklearn.model_selection import KFold

# Target encoding for tranche_effectif_ord with K-fold to prevent overfitting
def target_encode_kfold(df, column, target, n_splits=5):
    """
    Perform out-of-fold target encoding to prevent data leakage
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    encoded_col = np.zeros(len(df))
    
    for train_idx, val_idx in kf.split(df):
        # Calculate mean target for each category on training fold
        target_means = df.iloc[train_idx].groupby(column)[target].mean()
        
        # Apply to validation fold
        encoded_col[val_idx] = df.iloc[val_idx][column].map(target_means)
    
    # For any missing values, use global mean
    global_mean = df[target].mean()
    encoded_col = np.where(np.isnan(encoded_col), global_mean, encoded_col)
    
    return encoded_col