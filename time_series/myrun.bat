@echo off
chcp 65001

set DATA_PATH="C:\\data\\time_series_data\\archive\\vgsales.csv"
set OUTPUT_DIR_PATH="C:\\data\\time_series_data\\archive\\outputs"
set TRAIN_RATE=0.8

set PUBLISHER="Nintendo"
set LAGS=32
set XTICKS=5
set d=0
set PARAMS_NAME="Nintendo_best_params.txt"
set MODEL_NAME="Nintendo_model.pkl"

rem set PUBLISHER="Namco Bandai Games"
rem set LAGS=31
rem set XTICKS=5
rem set d=1
rem set PARAMS_NAME="Namco Bandai Games_best_params.txt"
rem set MODEL_NAME="Namco Bandai Games_model.pkl"

rem set PUBLISHER="Microsoft Game Studios"
rem set LAGS=18
rem set XTICKS=5
rem set d=0
rem set PARAMS_NAME="Microsoft Game Studios_best_params.txt"
rem set MODEL_NAME="Microsoft Game Studios_model.pkl"

rem set PUBLISHER="Sony Computer Entertainment"
rem set LAGS=22
rem set XTICKS=5
rem set d=0
rem set PARAMS_NAME="Sony Computer Entertainment_best_params.txt"
rem set MODEL_NAME="Sony Computer Entertainment_model.pkl"

python -m src.pre_analysis ^
    --data_path %DATA_PATH% ^
    --publisher %PUBLISHER% ^
    --output_dir_path %OUTPUT_DIR_PATH% ^
    --lags %LAGS% ^
    --xticks %XTICKS%

python -m src.optimize_model ^
    --data_path %DATA_PATH% ^
    --output_dir_path %OUTPUT_DIR_PATH% ^
    --publisher %PUBLISHER% ^
    --train_rate %TRAIN_RATE% ^
    --d %d%

python -m src.train ^
    --data_path %DATA_PATH% ^
    --output_dir_path %OUTPUT_DIR_PATH% ^
    --publisher %PUBLISHER% ^
    --train_rate %TRAIN_RATE% ^
    --params_path %OUTPUT_DIR_PATH%\\%PUBLISHER%\\%PARAMS_NAME%

python -m src.predict ^
    --data_path %DATA_PATH% ^
    --model_path %OUTPUT_DIR_PATH%\\%PUBLISHER%\\%MODEL_NAME% ^
    --output_dir_path %OUTPUT_DIR_PATH% ^
    --publisher %PUBLISHER% ^
    --train_rate %TRAIN_RATE%