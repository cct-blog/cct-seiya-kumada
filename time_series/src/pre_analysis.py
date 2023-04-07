import argparse
import os

from pmdarima import arima

import src.utils as utils

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/vgsales.csv")
    parser.add_argument("--publisher", type=str, default="Nintendo")
    parser.add_argument("--output_dir_path", type=str, default="data/")
    parser.add_argument("--lags", type=int, default=10)
    parser.add_argument("--xticks", type=int, default=5)
    args = parser.parse_args()

    # load data
    df = utils.load_data(args.data_path, args.publisher)

    # make output directory
    output_sub_dir_path = os.path.join(args.output_dir_path, args.publisher)
    if not os.path.exists(output_sub_dir_path):
        os.makedirs(output_sub_dir_path)

    # draw plot
    output_path = os.path.join(output_sub_dir_path, f"{args.publisher}_raw_data.jpg")
    utils.draw_plot(df, args.publisher, output_path)

    # evaluate kpss
    kpss_0, kpss_1 = utils.evaluate_kpss(df)

    d = arima.ndiffs(df["Global_Sales"])

    # save kpss result
    output_path = os.path.join(output_sub_dir_path, f"{args.publisher}_kpss.txt")
    with open(output_path, "w") as f:
        f.write(f"kpss_0: {kpss_0}\n")
        f.write(f"kpss_1: {kpss_1}\n")
        f.write(f"ndiffs: {d}\n")

    # calculate auto correlation
    output_path = os.path.join(output_sub_dir_path, f"{args.publisher}_auto_corr.jpg")
    utils.calculate_auto_correlation(
        args.lags, args.xticks, df["Global_Sales"], output_path, args.publisher
    )

    if kpss_0 > 0.05:
        print("The data is stationary.")
        exit()

    df_diff_1 = utils.calculate_diff(df, 1)
    output_path = os.path.join(output_sub_dir_path, f"{args.publisher}_auto_corr_diff_1.jpg")
    utils.calculate_auto_correlation(
        args.lags, args.xticks, df_diff_1["Global_Sales"], output_path, args.publisher
    )
