@echo off
setlocal enabledelayedexpansion

REM 设置输入和输出文件夹路径
set "input_folder=D:\server\home\songgaochao\codes\gof_a6k\new_exps\exp_mip360\release\garden\test_track\ours_30000\test_preds_-1"
set "output_folder=%input_folder%\down"

REM 创建输出文件夹
if not exist "%output_folder%" mkdir "%output_folder%"

REM 遍历输入文件夹下的所有视频文件
for %%f in ("%input_folder%\*.mp4") do (
    REM 获取文件名和扩展名
    set "filename=%%~nf"
    set "extension=%%~xf"

    REM 压缩视频并存储在输出文件夹
    ffmpeg -i "%%f" -vcodec libx265 -crf 28 "%output_folder%\!filename!_compressed!extension!"
)

echo 压缩完成！
pause
