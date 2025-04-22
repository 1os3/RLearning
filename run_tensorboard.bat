@echo off
REM 一键启动TensorBoard，监控训练日志
call .\venv\Scripts\activate
start tensorboard --logdir=runs --port=6006
