@echo off
set name=train

echo ============= Extracting frames =============
python ".\5 - Get frames from videos.py" ^
  -i "F:\Videos\retro-games\%name%" ^
  -o "..\images\%name%-frames" ^
  -f 1 -s 0.13

echo ============= Transforming images =============
python ".\5.1 - Transform frames.py" ^
  -i "..\images\%name%-frames" ^
  -o "..\images\transformed-%name%-frames"

echo ============= Filtering images =============
python ".\5.2 - Find blurry frames.py" ^
  -i "..\images\transformed-%name%-frames" ^
  -o "..\images\retro-games-gameplay-frames-30k-512p\%name%\"

echo ============= Generating captions =============
python ".\2 - Generate captions.py" ^
  -m "noamrot/FuseCap" ^
  -i "..\images\retro-games-gameplay-frames-30k-512p\%name%" ^
  -o "..\images\retro-games-gameplay-frames-30k-512p"
