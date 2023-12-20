# 訓練（バッシュスクリプト名: run_train）
最初にデータセットで訓練する。


```
python -m src.train \
	--dataset_dir_path /home/ubuntu/data/synthetic_scenes/train/shoe \
	--dataset_dir_name small_rgb \
	--output_dir_path /home/ubuntu/data/synthetic_scenes/outputs/shoe \
	--pose_dir_name pose \
	--image_ext png \
	--width 256 \
	--height 256 \
	--saving_interval 2  \
	--t_n 0 \
	--t_f 2.5 \
	--epochs 1 \
	--batch_size 2048
```
引数の説明
- dataset_dir_path: 入力ディレクトリのパス
- dataset_dir_name: 画像を格納したフォルダ名
- output_dir_path: 出力ディレクトリのパス
- pose_dir_name: カメラパラメータを格納したフォルダ名
- image_ext: 画像の拡張子
- width: 画像の幅
- height: 画像の高さ
- saving_interval: モデルを保存するタイミング
- t_n: レンダリングの下限（光線の下限）
- t_f: レンダリングの上限（光線の上限）
- epochs: エポック数
- batch_size: バッチサイズ

入力ディレクトリの構成

[ここ](https://drive.google.com/drive/folders/1ScsRlnzy9Bd_n-xw83SP-0t548v63mPH)からダウンロードしたファイルsynthetic_scenes.zipのshoeを使用した。shoeの中にあるディレクトリrgbに格納された画像のサイズは512x512である。これを256x256に変更した画像をsmall_rgbに収めた。

```
shoe/
    small_rgb/
    pose/
    intrinsics.txt
```

# 予測（run_predict）
視点（pose内のパラメータ）を与えて画像を描画する。
```
for ind in {0..0}; do
	python -m src.predict \
		--dataset_dir_path /home/ubuntu/data/synthetic_scenes/train/shoe \
		--dataset_dir_name small_rgb \
		--pose_dir_name pose \
		--image_ext png \
		--width 256 \
		--height 256 \
		--ind ${ind} \
		--model_path /home/ubuntu/data/synthetic_scenes/outputs/shoe/epoch_1.state \
		--t_n 0.0 \
		--t_f 2.5 \
		--view_path /home/ubuntu/data/synthetic_scenes/outputs/shoe/views/epoch_1.jpg
done
```
引数の説明
- dataset_dir_path: 入力ディレクトリのパス
- dataset_dir_name: 画像を格納したフォルダ名
- pose_dir_name: カメラパラメータを格納したフォルダ名
- image_ext: 画像の拡張子
- width: 画像の幅
- height: 画像の高さ
- ind: 視点（カメラパラメータ）
- model_path: 訓練済みモデルへのパス
- t_n: レンダリングの下限（光線の下限）
- t_f: レンダリングの上限（光線の上限）
- view_path: 出力画像のパス

たまにcoarseかfineのどちらかの絵が真っ白になることがある。理由は不明。訓練し直すと直る。

# 予測 with 回転（run_predict_rotation）
任意の軸周りに回転させた画像を描画する。
```
AXIS=z
python -m src.predict_rotation \
	--dataset_dir_path /home/ubuntu/data/synthetic_scenes/train/shoe \
	--dataset_dir_name small_rgb \
	--pose_dir_name pose \
	--image_ext png \
	--width 256 \
	--height 256 \
	--ind 0 \
	--model_path /home/ubuntu/data/synthetic_scenes/outputs/shoe/epoch_1.state \
	--t_n 0.0 \
	--t_f 2.5 \
	--output_path /home/ubuntu/data/synthetic_scenes/outputs/shoe/views/rotation_${AXIS}.jpg \
	--axis ${AXIS}
```
引数の説明
- AXIZ: 軸名
- dataset_dir_path: 入力ディレクトリのパス
- dataset_dir_name: 画像を格納したフォルダ名
- pose_dir_name: カメラパラメータを格納したフォルダ名
- image_ext: 画像の拡張子
- width: 画像の幅
- height: 画像の高さ
- ind: 視点（カメラパラメータ）
- model_path: 訓練済みモデルへのパス
- t_n: レンダリングの下限（光線の下限）
- t_f: レンダリングの上限（光線の上限）
- output_path: 出力ディレクトリのパス

# 描画範囲の観察（run_draw_rendering_region）
点群の存在範囲を見積もる。
```
ELEV=90
AZIM=0
python -m src.draw_rendering_region \
    --dataset_dir_path /home/ubuntu/data/synthetic_scenes/train/shoe \
    --dataset_dir_name small_rgb \
    --output_path /home/ubuntu/data/synthetic_scenes/outputs/shoe/rendering_region/shoe.jpg \
    --pose_dir_name pose \
    --image_ext png \
    --width 256 \
    --height 256 \
    --t_n 0.0 \
    --t_f 2.5 \
    --elev ${ELEV} \
    --azim ${AZIM} \
    --camera_interval 10
```
引数の説明
- dataset_dir_path: 入力ディレクトリのパス
- dataset_dir_name: 画像を格納したフォルダ名
- output_path: 出力ファイル名（このファイル名にELEVとAZIMが付加される）
- pose_dir_name: カメラパラメータを格納したフォルダ名
- image_ext: 画像の拡張子
- width: 画像の幅
- height: 画像の高さ
- t_n: レンダリングの下限（光線の下限）
- t_f: レンダリングの上限（光線の上限）
- elev: elevation angle（度）
- azim: azimuthal angle（度）
- camera_interval: pose内のファイルを何個おきに読み込むか。

標準出力に出力されるaverage focus（平均焦点座標）が点群生成時の視点になる。また、保存される画像を見て点群の存在範囲を適当に見積もる。

# 点群の生成（run_generate_points）
上の結果をもとに点群を生成する。
```
N=200
SIGMA_THRESHOLD=100
python -m src.generate_points \
	--output_path /home/ubuntu/data/synthetic_scenes/outputs/shoe/points/points_size_${N}_sigma_${SIGMA_THRESHOLD}.txt \
	--size ${N} \
	--sigma_threshold ${SIGMA_THRESHOLD} \
	--model_path /home/ubuntu/data/synthetic_scenes/outputs/shoe/epoch_1.state \
	--t_n 0.0 \
	--t_f 2.5 \
	--ox 0.02144805 \
	--oy -0.03721913 \
	--oz 0.7881192 \
	--xmin -1.5 \
	--xmax 1.5 \
	--ymin -1.5 \
	--ymax 1.5 \
	--zmin -1.5  \
	--zmax 1.5
```
引数の説明
- output_path: 出力ファイル名（点群の座標と色を記載したテキストファイル）
- size: x,y,z方向の分割数
- sigma_threshold: sigma（密度）の閾値
- model_path: 訓練済みモデルのパス
- t_n: レンダリングの下限（光線の下限）
- t_f: レンダリングの上限（光線の上限）
- ox: 平均焦点座標のx（1つ前のコマンドで得た値）
- oy: 平均焦点座標のy
- oz: 平均焦点座標のz
- xmin: 点群存在範囲（1つ前のコマンドで得た画像から判断）
- xmax: 点群存在範囲
- ymin: 点群存在範囲
- ymax: 点群存在範囲
- zmin: 点群存在範囲
- zmax: 点群存在範囲

出力ファイルは[Dai-Con Viewer](https://www.dai-con.net/dai-con-viewer/)で表示できる。
