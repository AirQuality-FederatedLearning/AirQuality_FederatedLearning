@echo off

REM --------------------------------------------------
REM 1) Activate the environment and launch the server
REM --------------------------------------------------
echo Starting the FL server...
start cmd /k "call venv\Scripts\activate && cd Flower && python server.py  --rounds 10 --save_dir  './saved_models"

REM --------------------------------------------------
REM 2) Launch each client on a single line
REM --------------------------------------------------
echo Starting client for site_1431_Patparganj_Delhi...
start cmd /k "call venv\Scripts\activate && cd Flower && python client.py --config client_config.yaml --dataset ..\Non_Null_Datasets\preprocessed_Raw_data_15Min_2024_site_1431_Patparganj_Delhi_DPCC_15Min.csv"

echo Starting client for site_1434_Wazirpur_Delhi...
start cmd /k "call venv\Scripts\activate && cd Flower && python client.py --config client_config.yaml --dataset ..\Non_Null_Datasets\preprocessed_Raw_data_15Min_2024_site_1434_Wazirpur_Delhi_DPCC_15Min.csv"

echo Starting client for site_1561_Mundka_Delhi...
start cmd /k "call venv\Scripts\activate && cd Flower && python client.py --config client_config.yaml --dataset ..\Non_Null_Datasets\preprocessed_Raw_data_15Min_2024_site_1561_Mundka_Delhi_DPCC_15Min.csv"

echo Starting client for site_1563_Pusa_Delhi...
start cmd /k "call venv\Scripts\activate && cd Flower && python client.py --config client_config.yaml --dataset ..\Non_Null_Datasets\preprocessed_Raw_data_15Min_2024_site_1563_Pusa_Delhi_DPCC_15Min.csv"

echo Starting client for site_5024_Alipur_Delhi...
start cmd /k "call venv\Scripts\activate && cd Flower && python client.py --config client_config.yaml --dataset ..\Non_Null_Datasets\preprocessed_Raw_data_15Min_2024_site_5024_Alipur_Delhi_DPCC_15Min.csv"

REM --------------------------------------------------
echo All processes started. Press any key to end.
pause
