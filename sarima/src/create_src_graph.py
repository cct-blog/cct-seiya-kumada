import matplotlib.pyplot as plt
import pandas as pd

SRC_PATH = "/home/kumada/projects/cct-seiya-kumada/sarima/data/co2_monthave_ryo_concated.csv"
DST_PATH = "/home/kumada/projects/cct-seiya-kumada/sarima/images/src_graph.jpg"
if __name__ == "__main__":
    # Load the CSV file
    data = pd.read_csv(SRC_PATH)

    # Convert the 'date' column to a datetime format for better plotting
    # data["date"] = pd.to_datetime(data["date"], format="%b-%y")
    data["date"] = pd.to_datetime(data["date"], infer_datetime_format=True)

    # Plot the graph
    plt.figure(figsize=(12, 6))
    plt.plot(data["date"], data["co2"], marker="o")
    plt.title("Monthly CO2 Concentration Over Time")
    plt.xlabel("Date")
    plt.ylabel("CO2 Concentration (ppm)")
    plt.grid(True)
    # plt.show()
    plt.savefig(DST_PATH)
