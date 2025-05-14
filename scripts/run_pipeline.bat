@echo off
set name=train
set PROJECT_PATH=E:\PythonProjects\Scraping Dataset\generating-annotations

echo ============= Extracting frames =============
python ".\5 - Get frames from videos.py" ^
  -i "F:\Videos\retro-games\%name%" ^
  -o "%PROJECT_PATH%\images\%name%-frames" ^
  -f 1 -s 0.13

echo ============= Transforming images =============
python ".\5.1 - Transform frames.py" ^
  -i "%PROJECT_PATH%\images\%name%-frames" ^
  -o "%PROJECT_PATH%\images\transformed-%name%-frames"

echo ============= Filtering images =============
python ".\5.2 - Filtering frames.py" ^
  -i "%PROJECT_PATH%\images\transformed-%name%-frames" ^
  -o "%PROJECT_PATH%\images\retro-games-gameplay-frames-30k-512p\%name%" ^
  -v "%PROJECT_PATH%\data\%name%_frames_varience.json"

echo ============= Generating captions =============
python ".\2 - Generate captions.py" ^
  -m "noamrot/FuseCap" ^
  -i "%PROJECT_PATH%\images\retro-games-gameplay-frames-30k-512p\%name%" ^
  -o "%PROJECT_PATH%\images\retro-games-gameplay-frames-30k-512p\%name%_captions.txt" ^
  -b 32
