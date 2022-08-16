from sklearn.datasets import fetch_california_housing
import numpy as np
from typing import Tuple, List
from numpy.typing import NDArray


def load_california_housing() -> Tuple[
    NDArray[np.float32], NDArray[np.float32], List[str]
]:
    housing = fetch_california_housing(as_frame=False)
    data = housing["data"]
    print("> data: ", type(data), data.shape)
    target = housing["target"]
    print("> target: ", type(target), target.shape)
    feature_names = housing["feature_names"]
    print("> feature_names: ", type(feature_names))
    # MedInc:所得の中央値
    # HouseAge: 築年数の中央値
    # AveRooms: 部屋数の中央値
    # AveBedrms: 寝室数
    # Population: 住居人の総人数
    # AveOccup: 世帯人数の平均値
    # Latitude: 緯度
    # Longitude: 経度
    # MedHouseVal: 住宅価格の中央値
    return data, target, feature_names


def make_heat_map(
    data: NDArray[np.float32],
    target: NDArray[np.float32],
    feature_names: List[str],
) -> None:
    import seaborn as sns
    import matplotlib.pyplot as plt

    ds: NDArray[np.float32] = np.concatenate([data, target], axis=1)
    cs = np.corrcoef(ds.transpose())
    names = feature_names.copy()
    names.append("MedHouseVal")
    sns.heatmap(cs, annot=True, xticklabels=names, yticklabels=names)

    # グラフを保存する
    plt.savefig("heatmap_house.jpg")


if __name__ == "__main__":
    data, target, feature_names = load_california_housing()
    print(data.shape)
    target = target[:, np.newaxis]
    print(target.shape)
    make_heat_map(data, target, feature_names)

    lat_max = np.max(data[:, 6])
    lat_min = np.min(data[:, 6])

    lon_max = np.max(data[:, 7])
    lon_min = np.min(data[:, 7])

    print(lat_min, lat_max)
    print(lon_min, lon_max)
