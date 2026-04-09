@echo off
setlocal
set "FORECAST_PY=%USERPROFILE%\.conda\envs\forecastapp\python.exe"
if exist "%FORECAST_PY%" (
    "%FORECAST_PY%" -m streamlit run "%~dp0streamlit_app.py"
) else (
    python -m streamlit run "%~dp0streamlit_app.py"
)
