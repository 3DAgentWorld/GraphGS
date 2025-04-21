@echo off
setlocal enabledelayedexpansion

REM 设置视频文件所在的目录
set "video_dir=D:\server\home\songgaochao\datasets_a6000\flame_steak\flame_steak"

REM 设置输出图像序列的根目录
set "output_dir=D:\server\home\songgaochao\datasets_a6000\flame_steak\flame_steak"

REM 切换到视频文件目录
cd /d "%video_dir%"

REM 遍历视频文件
for %%f in (*.mp4) do (
    REM 获取视频文件名（不带扩展名）
    set "filename=%%~nf"

    REM 创建对应的输出文件夹
    mkdir "%output_dir%\!filename!"

    REM 导出图像序列到对应的文件夹
    ffmpeg -i "%%f" "%output_dir%\!filename!\images\frame_%%04d.png"
)

echo 完成！
pause
