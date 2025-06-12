import pandas as pd

def load_data(path):
    df = pd.read_csv(path)
    print(f"dataset with shape: {df.shape}")
    print("feature types:\n", df.dtypes)
    return df