@echo off

REM Edit these values for the actual learner host

REM change this to some short unique string to identify yourself among the other contributors
set WORKER_NAME=Your-Name-Here

REM leave empty for localhost
set REDIS_HOST=

set REDIS_PASSWORD=Some-Fancy-Password


REM Using sets inside of ifs annoys windows, this Setlocal fixes that
Setlocal EnableDelayedExpansion


REM optional argument to launch multiple workers at once
set instance_num=1
if %1.==. goto :endparams
set instance_num=%1
:endparams


WHERE git
if !errorlevel! neq 0 echo git required, download here: https://git-scm.com/download/win
if !errorlevel! neq 0 pause & exit

if exist !APPDATA!\bakkesmod\ (
    echo Bakkesmod located at !APPDATA!\bakkesmod\ 
    goto :done
    
) else (
    echo \nBakkesmod not found at !APPDATA!\bakkesmod\ 
    echo * If you've already installed it elsewhere, you're fine *
    set /p choice=Download Bakkesmod[y/n]?:
    
    if /I "!choice!" EQU "Y" goto :install
    if /I "!choice!" EQU "N" goto :no_install
)

    :install
    echo Downloading Bakkesmod
    curl.exe -L --output !USERPROFILE!\Downloads\BakkesModSetup.zip --url https://github.com/bakkesmodorg/BakkesModInjectorCpp/releases/latest/download/BakkesModSetup.zip
    tar -xf !USERPROFILE!\Downloads\BakkesModSetup.zip -C !USERPROFILE!\Downloads\
    !USERPROFILE!\Downloads\BakkesModSetup.exe
    
    if !errorlevel! neq 0 echo \n*** Problem with Bakkesmod installation. Manually install and try again ***\n
    if !errorlevel! neq 0 pause & exit /b !errorlevel!
    
    echo Bakkesmod installed!
    goto :done

    :no_install
    goto :done    
    
    :done

python -m venv !LocalAppData!\nathanbot\venv

CALL  !LocalAppData!\nathanbot\venv\Scripts\activate.bat
echo %cd%


@REM python -m pip install -U git+https://github.com/some-rando-rl/nathan-bot-philips.git
python -m pip install --upgrade pip
python -m pip install -U -r requirements.txt

if !errorlevel! neq 0 pause & exit /b !errorlevel!


REM Automatically pull latest version, avoid stashing if no changes to avoid errors
for /f %%i in ('call git status --porcelain --untracked-files=no') do set stash=%%i
if not [%stash%] == [] (
    git stash
    git checkout main
    git pull origin main
    git stash apply
) else (
    git checkout main
    git pull origin main
)

echo.
echo #########################
echo ### Launching Worker! ###
echo #########################


for /L %%i in (1, 1, !instance_num!) do (
    start cmd /c python worker.py ^& pause
    timeout 90 >nul
)
