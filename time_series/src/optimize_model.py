import argparse
import os

from pmdarima import model_selection

import src.utils as utils

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/vgsales.csv")
    parser.add_argument("--output_dir_path", type=str, default="data/vgsales.csv")
    parser.add_argument("--publisher", type=str, default="Nintendo")
    parser.add_argument("--train_rate", type=float, default=0.9)
    parser.add_argument("--d", type=int, default=1)
    parser.add_argument("--m", type=int, default=0)
    args = parser.parse_args()

    # load data
    df = utils.load_data(args.data_path, args.publisher)

    # split train and test data
    df_train, df_test = model_selection.train_test_split(df, train_size=args.train_rate)
    print(f"> train data size: {len(df_train)}")
    print(f"> test data size: {len(df_test)}")

    # optimize model
    best_result = utils.execute_grid_search(df_train, args.d, args.m)
    print(f"> best model(p,d,q),(P,D,Q,m),AIC: {best_result}")

    # make output directory
    output_sub_dir_path = os.path.join(args.output_dir_path, args.publisher)
    if not os.path.exists(output_sub_dir_path):
        os.makedirs(output_sub_dir_path)

    # save best parameters
    output_path = os.path.join(output_sub_dir_path, f"{args.publisher}_best_params.txt")
    (p, d, q) = best_result[0]
    (P, D, Q, m) = best_result[1]
    with open(output_path, "w") as f:
        f.write(f"{p},{d},{q},{P},{D},{Q},{m}\n")
