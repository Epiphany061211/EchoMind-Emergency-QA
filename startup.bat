@echo off
chcp 65001 >nul
cd /d %~dp0
call D:\software\anaconda\Scripts\activate.bat emergency-echomind
echo 正在启动应急预案助手后端...
echo 启动成功后，双击打开本目录下的"应急预案助手前端.html"即可使用。
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
pause
