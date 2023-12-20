import argparse
import os
import pickle

from pmdarima import model_selection

import src.utils as utils


def print_args(args: argparse.Namespace) -> None:
    print(f"> data_path: {args.data_path}")
    print(f"> model_path: {args.model_path}")
    print(f"> output_dir_path: {args.output_dir_path}")
    print(f"> publisher: {args.publisher}")
    print(f"> train_rate: {args.train_rate}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/vgsales.csv")
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--output_dir_path", type=str, default="data/")
    parser.add_argument("--publisher", type=str, default="Nintendo")
    parser.add_argument("--train_rate", type=float, default=0.8)
    args = parser.parse_args()

    print_args(args)
    assert os.path.exists(args.data_path), f"File not found: {args.data_path}"
    assert os.path.exists(args.model_path), f"File not found: {args.model_path}"

    # load data
    df = utils.load_data(args.data_path, args.publisher)

    # split train and test data
    df_train, df_test = model_selection.train_test_split(df, train_size=args.train_rate)

    # load model
    with open(args.model_path, "rb") as f:
        model = pickle.load(f)

    # predict
    train_pred, test_pred, test_pred_ci = utils.predict(model, df_test)

    # make output path
    output_sub_dir_path = os.path.join(args.output_dir_path, args.publisher)
    if not os.path.exists(output_sub_dir_path):
        os.makedirs(output_sub_dir_path)

    # save prediction images
    pred_path = os.path.join(output_sub_dir_path, f"{args.publisher}_pred.jpg")
    utils.draw_results(df_train, df_test, train_pred, test_pred, test_pred_ci, pred_path)

    pred_part_path = os.path.join(output_sub_dir_path, f"{args.publisher}_pred_part.jpg")
    utils.draw_results_part(df_test, test_pred, test_pred_ci, pred_part_path)
