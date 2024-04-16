import matplotlib.pyplot as plt
import numpy as np
from pyclustering.cluster.gmeans import gmeans
from sklearn.cluster import DBSCAN, KMeans
from sklearn.datasets import make_moons


def execute_GMeans(X: np.ndarray):
    # pyclusteringの入力形式にデータを変換
    data = X.tolist()

    # GMeansクラスタリングの実行
    gmeans_instance = gmeans(data)
    gmeans_instance.process()
    clusters = gmeans_instance.get_clusters()
    centers = gmeans_instance.get_centers()

    colors = ["b", "g", "r", "c", "m", "y", "k"]

    for cluster_index, cluster in enumerate(clusters):  # type:ignore
        cluster_points = np.array([data[index] for index in cluster])
        plt.scatter(
            cluster_points[:, 0], cluster_points[:, 1], s=50, edgecolor="k", c=colors[cluster_index % len(colors)]
        )

    # クラスタの中心をプロット
    center_points = np.array(centers)
    plt.scatter(
        center_points[:, 0],
        center_points[:, 1],
        s=200,
        c="yellow",
        marker="*",  # type:ignore
        label="Centers",
    )

    plt.title("G-means Clustering")
    plt.legend()

    # 結果を画像として保存
    plt.savefig("gmeans.jpg")


def execute_DBSCAN(X: np.ndarray):
    # DBSCANクラスタリングの実行
    dbscan = DBSCAN(eps=0.2, min_samples=5)
    dbscan.fit(X)

    # クラスタリング結果のラベルを取得
    labels = dbscan.labels_

    # ノイズのデータポイントを特定
    # 全ての要素をFalseに設定する。
    core_samples_mask = np.zeros_like(dbscan.labels_, dtype=bool)
    # coreの箇所だけTrueにする。
    core_samples_mask[dbscan.core_sample_indices_] = True
    unique_labels = set(labels)

    # クラスタリング結果の可視化
    colors = ["b", "g", "r", "c", "m", "y"]
    for k in unique_labels:
        if k == -1:
            # ノイズは黒色で表示
            c = "k"

        # kに等しい箇所がTrue、他はFalseになる。
        class_member_mask = labels == k

        c = colors[k % len(colors)]
        # コアサンプルのプロット
        # kの個所かつコアの個所
        xy = X[class_member_mask & core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], "o", c=c, markeredgecolor="k", markersize=7)

        # ノンコアサンプルのプロット
        # kの箇所かつノンコアの箇所
        xy = X[class_member_mask & ~core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], "o", c=c, markeredgecolor="k", markersize=3)

        if k == -1:
            print(f"the number of noises: {len(xy)}")
    plt.title("DBSCAN Clustering")
    plt.savefig("dbscan.jpg")
    plt.clf()


def execute_KMeans(X: np.ndarray):
    # KMeansクラスタリングの適用
    kmeans = KMeans(n_clusters=2, random_state=42)
    y_kmeans = kmeans.fit_predict(X)

    # クラスタリング結果のプロット
    # plt.figure(figsize=(8, 5))
    plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, cmap="viridis", marker="o", edgecolor="k", s=50)  # type:ignore
    plt.scatter(
        kmeans.cluster_centers_[:, 0],
        kmeans.cluster_centers_[:, 1],
        s=200,
        c="red",
        marker="*",  # type:ignore
        label="Centroids",
    )
    plt.title("K-means Clustering")
    plt.legend()
    plt.savefig("kmeans.jpg")
    plt.clf()


def draw_samples(X: np.ndarray) -> None:
    plt.title("Samples")
    plt.scatter(X[:, 0], X[:, 1], c="b", marker="o", edgecolor="k", s=50)  # type:ignore
    plt.savefig("samples.jpg")


if __name__ == "__main__":
    # サンプルデータセットの生成 (半月形データ)
    X, _ = make_moons(n_samples=300, noise=0.1, random_state=42)

    draw_samples(X)

    # DBSCAN法を実行する。
    execute_DBSCAN(X)

    # KMeans法を実行する。
    execute_KMeans(X)

    # GMeans法を実行する。
    execute_GMeans(X)
