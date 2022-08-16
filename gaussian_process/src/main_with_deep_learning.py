import src.util as util
import numpy as np

# import src.pytorch_model as pytorch_model
import src.pyro_model as pyro_model


if __name__ == "__main__":
    pass
    (
        train_xs,
        train_ys,
        test_xs,
        test_ys,
    ) = util.load_dataset(util.DATA_PATH, util.TRAIN_SIZE)
    train_ys = train_ys.astype(np.float32)
    test_ys = test_ys.astype(np.float32)

    # pytorch_model.execute(
    #     train_xs,
    #     train_ys,
    #     test_xs,
    #     test_ys,
    #     "loss_pytorch.jpg",
    #     "pred_pytorch.jpg",
    #     "pytorch_model.pth",
    # )

    pyro_model.execute(
        train_xs,
        train_ys,
        test_xs,
        test_ys,
        "loss_pytorch.jpg",
        "pred_pytorch.jpg",
        "pytorch_model.pth",
    )
