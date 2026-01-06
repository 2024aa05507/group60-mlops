import pandas as pd
import os
from ucimlrepo import fetch_ucirepo 

def download_data(output_path="data/raw/heart.csv"):
    # Fetch dataset (ID 45 is Heart Disease UCI)
    heart_disease = fetch_ucirepo(id=45) 
    
    X = heart_disease.data.features 
    y = heart_disease.data.targets 
    
    # Combine for unified processing
    df = pd.concat([X, y], axis=1)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Data saved to {output_path}")
    return df

if __name__ == "__main__":
    download_data()