import pandas as pd
import scipy.stats as stats

DATA_PATH = "./data/data.txt"
if __name__ == "__main__":
    df = pd.read_table(DATA_PATH)
    ds = df.to_numpy()
    ds = stats.zscore(ds)
    print(ds)
