@echo off
set name=train
set PROJECT_PATH=E:\PythonProjects\Scraping Dataset\generating-annotations

@REM echo ============= Extracting frames =============
@REM python ".\5 - Get frames from videos.py" ^
@REM   -i "F:\Videos\retro-games\%name%" ^
@REM   -o "%PROJECT_PATH%\images\%name%-frames" ^
@REM   -f 1 -s 0.13

@REM echo ============= Transforming images =============
@REM python ".\5.1 - Transform frames.py" ^
@REM   -i "%PROJECT_PATH%\images\%name%-frames" ^
@REM   -o "%PROJECT_PATH%\images\transformed-%name%-frames"

@REM echo ============= Filtering images =============
@REM python ".\5.2 - Filtering frames.py" ^
@REM   -i "%PROJECT_PATH%\images\transformed-%name%-frames" ^
@REM   -o "%PROJECT_PATH%\images\retro-games-gameplay-frames-30k-512p\%name%" ^
@REM   -v "%PROJECT_PATH%\data\%name%_frames_varience.json"

echo ============= Generating captions =============
python ".\2 - Generate captions.py" ^
  -m "noamrot/FuseCap" ^
  -i "%PROJECT_PATH%\images\retro-games-gameplay-frames-30k-512p\%name%" ^
  -o "%PROJECT_PATH%\images\retro-games-gameplay-frames-30k-512p\%name%_captions.txt" ^
  -b 32
