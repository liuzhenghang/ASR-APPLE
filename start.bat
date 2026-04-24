@echo off
setlocal enabledelayedexpansion

rem ---------- 切到脚本所在目录 ----------
set "SCRIPT_DIR=%~dp0"
if "%SCRIPT_DIR:~-1%"=="\" set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"
cd /d "%SCRIPT_DIR%"

rem ---------- 平台子目录：Windows/Intel 走 windows-intel ----------
set "SUBDIR=%SCRIPT_DIR%\windows-intel"
set "REQ_FILE=%SUBDIR%\requirements.txt"

rem ---------- 可调参数（保留外部已设置的值） ----------
if not defined VENV_DIR        set "VENV_DIR=%SCRIPT_DIR%\.venv"
if not defined PYTHON_BIN      set "PYTHON_BIN=python"
if not defined ASR_HOST        set "ASR_HOST=0.0.0.0"
if not defined ASR_PORT        set "ASR_PORT=8000"
rem bweng/whisper-large-v3-turbo-int8-ov 是老版 optimum-intel 导的，新 openvino-genai 跑会报
rem "Port for tensor name beam_idx was not found"。FluidInference 这版是新导出的 stateful IR，
rem 名字虽带 -npu 但 CPU/GPU 也能跑。
if not defined ASR_MODEL_ID    set "ASR_MODEL_ID=FluidInference/whisper-large-v3-turbo-int8-ov-npu"
if not defined ASR_ALIGNER_ID  set "ASR_ALIGNER_ID="
if not defined ASR_ENABLE_ALIGN set "ASR_ENABLE_ALIGN=0"
if not defined ASR_WORD_TIMESTAMPS set "ASR_WORD_TIMESTAMPS=0"
if not defined ASR_DEVICE      set "ASR_DEVICE=GPU"
if not defined ASR_SEG_GAP_SEC set "ASR_SEG_GAP_SEC=0.8"
if not defined ASR_SEG_MAX_DURATION set "ASR_SEG_MAX_DURATION=30"
if not defined ASR_SEG_MAX_CHARS    set "ASR_SEG_MAX_CHARS=120"
if not defined ASR_ALIGN_MAX_CHARS  set "ASR_ALIGN_MAX_CHARS=3000"
if not defined ASR_MAX_QUEUE   set "ASR_MAX_QUEUE=5"
if not defined ASR_MAX_CONCURRENCY set "ASR_MAX_CONCURRENCY=1"
if not defined ASR_TIMEOUT     set "ASR_TIMEOUT=180"
if not defined ASR_ALIGN_TIMEOUT   set "ASR_ALIGN_TIMEOUT=60"
if not defined ASR_WORKER_READY_TIMEOUT set "ASR_WORKER_READY_TIMEOUT=600"

rem ---------- 模型缓存目录：项目内 models/ ----------
set "MODELS_DIR=%SCRIPT_DIR%\models"
if not exist "%MODELS_DIR%" mkdir "%MODELS_DIR%"
set "HF_HOME=%MODELS_DIR%"
set "HUGGINGFACE_HUB_CACHE=%MODELS_DIR%\hub"

rem ---------- 检测/创建虚拟环境 ----------
if not exist "%VENV_DIR%\Scripts\activate.bat" (
    echo [start.bat] 未检测到虚拟环境，使用 %PYTHON_BIN% 创建: %VENV_DIR%
    where %PYTHON_BIN% >nul 2>&1
    if errorlevel 1 (
        echo [start.bat] 找不到 %PYTHON_BIN%，请先安装 Python 3 1>&2
        exit /b 1
    )
    %PYTHON_BIN% -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo [start.bat] 创建虚拟环境失败 1>&2
        exit /b 1
    )
)

call "%VENV_DIR%\Scripts\activate.bat"
if errorlevel 1 (
    echo [start.bat] 激活虚拟环境失败 1>&2
    exit /b 1
)

rem ---------- 处理参数：install 代表强制重装依赖，其余原样透传给 app.py ----------
set "FORCE_INSTALL=0"
set "EXTRA_ARGS="
:parse_args
if "%~1"=="" goto after_args
if /i "%~1"=="install" (
    set "FORCE_INSTALL=1"
) else (
    set "EXTRA_ARGS=!EXTRA_ARGS! %1"
)
shift
goto parse_args
:after_args

for /f "tokens=*" %%v in ('python --version 2^>^&1') do set "PY_VER=%%v"
for /f "tokens=*" %%p in ('where python') do (
    set "PY_PATH=%%p"
    goto after_py_path
)
:after_py_path
echo [start.bat] python=%PY_VER% at %PY_PATH%

rem ---------- 安装依赖（对比 stamp 与 requirements.txt 修改时间） ----------
set "STAMP_FILE=%VENV_DIR%\.requirements.windows-intel.stamp"

set "NEED_INSTALL=0"
if "%FORCE_INSTALL%"=="1" (
    echo [start.bat] 强制重新安装依赖...
    set "NEED_INSTALL=1"
) else if not exist "%STAMP_FILE%" (
    set "NEED_INSTALL=1"
) else (
    rem 用 PowerShell 比较 mtime：REQ_FILE 比 STAMP 新就重装
    for /f %%r in ('powershell -NoProfile -Command "if ((Get-Item -LiteralPath '%REQ_FILE%').LastWriteTime -gt (Get-Item -LiteralPath '%STAMP_FILE%').LastWriteTime) { Write-Output 1 } else { Write-Output 0 }"') do set "NEED_INSTALL=%%r"
)

if "%NEED_INSTALL%"=="1" (
    echo [start.bat] 安装/更新依赖 ^(%REQ_FILE%^)...
    python -m pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple
    if errorlevel 1 exit /b 1
    python -m pip install -r "%REQ_FILE%" -i https://pypi.tuna.tsinghua.edu.cn/simple
    if errorlevel 1 exit /b 1
    echo. > "%STAMP_FILE%"
) else (
    echo [start.bat] 依赖已是最新，跳过安装 ^(删除 %STAMP_FILE% 可强制重装^)
)

rem ---------- 把子目录塞进 PYTHONPATH，让 app.py 能 import worker ----------
if defined PYTHONPATH (
    set "PYTHONPATH=%SUBDIR%;%PYTHONPATH%"
) else (
    set "PYTHONPATH=%SUBDIR%"
)

echo [start.bat] ========================================
echo [start.bat] PLATFORM     : windows-intel ^(openvino-genai^)
echo [start.bat] MODEL        : %ASR_MODEL_ID%
echo [start.bat] DEVICE       : %ASR_DEVICE% ^(GPU=iGPU / CPU^)
echo [start.bat] ALIGN        : enable=%ASR_ENABLE_ALIGN% ^(windows-intel 不支持 align^)
echo [start.bat] WORD_TS      : %ASR_WORD_TIMESTAMPS%
echo [start.bat] SEG          : gap=%ASR_SEG_GAP_SEC%s max_dur=%ASR_SEG_MAX_DURATION%s max_chars=%ASR_SEG_MAX_CHARS%
echo [start.bat] MODELS_DIR   : %MODELS_DIR%
echo [start.bat] PYTHONPATH   : %PYTHONPATH%
echo [start.bat] HOST:PORT    : %ASR_HOST%:%ASR_PORT%
echo [start.bat] MAX_QUEUE    : %ASR_MAX_QUEUE%
echo [start.bat] MAX_CONCURR. : %ASR_MAX_CONCURRENCY%
echo [start.bat] TIMEOUT      : asr=%ASR_TIMEOUT%s align=%ASR_ALIGN_TIMEOUT%s
echo [start.bat] ========================================

rem ---------- 启动 ----------
python "%SCRIPT_DIR%\app.py" %EXTRA_ARGS%
exit /b %ERRORLEVEL%
