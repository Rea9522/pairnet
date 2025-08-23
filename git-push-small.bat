@echo off
:: git-push-small.bat
:: �� Windows ���Զ�ֻ�ϴ�С��ָ����С���ļ�

:: ����Զ�ֿ̲��ַ
git remote set-url origin https://github.com/Rea9522/pairnet.git

setlocal enabledelayedexpansion

:: Ĭ�ϲ���
set DEFAULT_SIZE=5
set DEFAULT_BRANCH=main
set DEFAULT_MSG=Upload small files

echo =======================================
echo    ?? Git �Զ��ϴ�С�ļ��ű�
echo =======================================

:: ѯ���ļ���С��ֵ����λMB��
set /p SIZE=�������ļ���С��ֵ(MB��Ĭ�� %DEFAULT_SIZE%): 
if "%SIZE%"=="" set SIZE=%DEFAULT_SIZE%

:: ѯ�ʷ�֧��
set /p BRANCH=���������͵ķ�֧��(Ĭ�� %DEFAULT_BRANCH%): 
if "%BRANCH%"=="" set BRANCH=%DEFAULT_BRANCH%

:: ѯ���ύ��Ϣ
set /p MSG=�������ύ��Ϣ(Ĭ�� "%DEFAULT_MSG%"): 
if "%MSG%"=="" set MSG=%DEFAULT_MSG%

echo.
echo ?? ���ڲ���С�� %SIZE%MB ���ļ�...

:: �� PowerShell ����С�ļ�
set FILELIST=files_to_add.txt
del %FILELIST% 2>nul

powershell -command "Get-ChildItem -Recurse | Where-Object { -not $_.PSIsContainer -and $_.Length -lt (%SIZE%MB) } | Select-Object -ExpandProperty FullName" > %FILELIST%

:: ����Ƿ����ļ�
for /f %%C in ('find /c /v "" ^< %FILELIST%') do set COUNT=%%C

if %COUNT%==0 (
    echo ? û���ҵ�С�� %SIZE%MB ���ļ���
    pause
    exit /b
)

echo.
echo ? �����ļ������ϴ���
type %FILELIST%

echo =======================================
set /p CONFIRM=�Ƿ�����ύ��Щ�ļ�? (y/n): 
if /i not "%CONFIRM%"=="y" (
    echo ? ��ȡ���ϴ�
    del %FILELIST%
    pause
    exit /b
)

:: ����ļ����ݴ���
echo ?? ����ļ���...
for /f "delims=" %%F in (%FILELIST%) do (
    git add "%%F"
)

:: �ύ����
git commit -m "%MSG%"

:: ���͵�Զ�̷�֧
echo ?? �������͵���֧: %BRANCH% ...
git push origin %BRANCH%

echo.
echo ?? ������ɣ�
del %FILELIST%
pause
