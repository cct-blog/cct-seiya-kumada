import pandas as pd

# co2 --- 二酸化炭素濃度の月平均値(綾里)[ppm]
# 11-Aprに欠損値がある。上下の平均値で置き換える。

if __name__ == "__main__":
    SRC_PATH = "/home/kumada/projects/cct-seiya-kumada/sarima/data/co2_monthave_ryo.csv"
    # load data from column 1 to 2
    df = pd.read_csv(SRC_PATH, usecols=[0, 1, 2], encoding="shift-jis")
    # print(df.head())
    df["co2"] = df["co2"].astype(float)
    df["date"] = df["year"].astype(str) + "-" + df["month"].astype(str)
    print(df.head())

    DST_PATH = "/home/kumada/projects/cct-seiya-kumada/sarima/data/co2_monthave_ryo_concated.csv"
    df.to_csv(DST_PATH, columns=["date", "co2"], index=False)
