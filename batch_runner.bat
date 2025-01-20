@echo off

REM --------------------------------------------------
REM 1) Activate the environment and launch the server
REM --------------------------------------------------
echo Starting the FL server...
start cmd /k "call venv\Scripts\activate && cd Flower && python server.py"

REM --------------------------------------------------
REM 2) Launch each client on a single line
REM --------------------------------------------------
echo Starting client for site_103_CRRI_Mathura_Road...
start cmd /k "call venv\Scripts\activate && cd Flower && python client.py --config client_config.yaml --dataset ..\Dataset_Dehli\Raw_data_15Min_2017_site_103_CRRI_Mathura_Road_Delhi_IMD_15Min.csv --idx 0"

echo Starting client for site_105_North_Campus_DU...
start cmd /k "call venv\Scripts\activate && cd Flower && python client.py --config client_config.yaml --dataset ..\Dataset_Dehli\Raw_data_15Min_2017_site_105_North_Campus_DU_Delhi_IMD_15Min.csv --idx 0"

echo Starting client for site_108_Aya_Nagar...
start cmd /k "call venv\Scripts\activate && cd Flower &&    "

echo Starting client for site_115_NSIT_Dwarka...
start cmd /k "call venv\Scripts\activate && cd Flower && python client.py --config client_config.yaml --dataset ..\Dataset_Dehli\Raw_data_15Min_2017_site_115_NSIT_Dwarka_Delhi_CPCB_15Min.csv --idx 0"

echo Starting client for site_117_ITO...
start cmd /k "call venv\Scripts\activate && cd Flower && python client.py --config client_config.yaml --dataset ..\Dataset_Dehli\Raw_data_15Min_2017_site_117_ITO_Delhi_CPCB_15Min.csv --idx 0"

echo Starting client for site_118_DTU...
start cmd /k "call venv\Scripts\activate && cd Flower && python client.py --config client_config.yaml --dataset ..\Dataset_Dehli\Raw_data_15Min_2017_site_118_DTU_Delhi_CPCB_15Min.csv --idx 0"

echo Starting client for site_122_Mandir_Marg...
start cmd /k "call venv\Scripts\activate && cd Flower && python client.py --config client_config.yaml --dataset ..\Dataset_Dehli\Raw_data_15Min_2017_site_122_Mandir_Marg_Delhi_DPCC_15Min.csv --idx 0"

echo Starting client for site_124_R_K_Puram...
start cmd /k "call venv\Scripts\activate && cd Flower && python client.py --config client_config.yaml --dataset ..\Dataset_Dehli\Raw_data_15Min_2017_site_124_R_K_Puram_Delhi_DPCC_15Min.csv --idx 0"

echo Starting client for site_125_Punjabi_Bagh...
start cmd /k "call venv\Scripts\activate && cd Flower && python client.py --config client_config.yaml --dataset ..\Dataset_Dehli\Raw_data_15Min_2017_site_125_Punjabi_Bagh_Delhi_DPCC_15Min.csv --idx 0"

echo -----------------------------------------------
echo All processes started. Press any key to end.
pause
