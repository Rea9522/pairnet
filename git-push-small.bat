@echo off
:: git-push-small.bat
:: 在 Windows 下自动只上传小于指定大小的文件

:: 设置远程仓库地址
git remote set-url origin https://github.com/Rea9522/pairnet.git

setlocal enabledelayedexpansion

:: 默认参数
set DEFAULT_SIZE=5
set DEFAULT_BRANCH=main
set DEFAULT_MSG=Upload small files

echo =======================================
echo    ?? Git 自动上传小文件脚本
echo =======================================

:: 询问文件大小阈值（单位MB）
set /p SIZE=请输入文件大小阈值(MB，默认 %DEFAULT_SIZE%): 
if "%SIZE%"=="" set SIZE=%DEFAULT_SIZE%

:: 询问分支名
set /p BRANCH=请输入推送的分支名(默认 %DEFAULT_BRANCH%): 
if "%BRANCH%"=="" set BRANCH=%DEFAULT_BRANCH%

:: 询问提交信息
set /p MSG=请输入提交信息(默认 "%DEFAULT_MSG%"): 
if "%MSG%"=="" set MSG=%DEFAULT_MSG%

echo.
echo ?? 正在查找小于 %SIZE%MB 的文件...

:: 用 PowerShell 查找小文件
set FILELIST=files_to_add.txt
del %FILELIST% 2>nul

powershell -command "Get-ChildItem -Recurse | Where-Object { -not $_.PSIsContainer -and $_.Length -lt (%SIZE%MB) } | Select-Object -ExpandProperty FullName" > %FILELIST%

:: 检查是否有文件
for /f %%C in ('find /c /v "" ^< %FILELIST%') do set COUNT=%%C

if %COUNT%==0 (
    echo ? 没有找到小于 %SIZE%MB 的文件。
    pause
    exit /b
)

echo.
echo ? 以下文件将被上传：
type %FILELIST%

echo =======================================
set /p CONFIRM=是否继续提交这些文件? (y/n): 
if /i not "%CONFIRM%"=="y" (
    echo ? 已取消上传
    del %FILELIST%
    pause
    exit /b
)

:: 添加文件到暂存区
echo ?? 添加文件中...
for /f "delims=" %%F in (%FILELIST%) do (
    git add "%%F"
)

:: 提交更改
git commit -m "%MSG%"

:: 推送到远程分支
echo ?? 正在推送到分支: %BRANCH% ...
git push origin %BRANCH%

echo.
echo ?? 推送完成！
del %FILELIST%
pause
