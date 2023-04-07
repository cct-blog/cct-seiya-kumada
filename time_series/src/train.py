import argparse
import os

from pmdarima import model_selection

import src.utils as utils


def print_args(args):
    print(f"> data_path: {args.data_path}")
    print(f"> publisher: {args.publisher}")
    print(f"> output_dir_path: {args.output_dir_path}")
    print(f"> train_rate: {args.train_rate}")
    print(f"> params_path: {args.params_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/vgsales.csv")
    parser.add_argument("--publisher", type=str, default="Nintendo")
    parser.add_argument("--output_dir_path", type=str, default="data/")
    parser.add_argument("--train_rate", type=float, default=0.8)
    parser.add_argument("--params_path", type=str)
    args = parser.parse_args()

    print_args(args)
    assert os.path.exists(args.data_path), f"File not found: {args.data_path}"
    assert os.path.exists(args.params_path), f"File not found: {args.params_path}"

    # load data
    df = utils.load_data(args.data_path, args.publisher)
    df_train, df_test = model_selection.train_test_split(df, train_size=args.train_rate)

    # load best parameters
    a, A = utils.read_params(args.params_path)

    # train model
    model = utils.train(df_train, a, A)

    # make output path
    output_sub_dir_path = os.path.join(args.output_dir_path, args.publisher)
    if not os.path.exists(output_sub_dir_path):
        os.makedirs(output_sub_dir_path)

    # save model
    output_path = os.path.join(output_sub_dir_path, f"{args.publisher}_model.pkl")
    model.save(output_path)
