# pose-estimation
YOLOv7 для детекций скелета человека

Для работы с кодом этого репозитория необходимо сначала склонировать себе код [репозитория `yolov7`](https://github.com/WongKinYiu/yolov7)

```sh
git clone https://github.com/WongKinYiu/yolov7.git
```

Перейти в директорию репозитория и установить неибходимые библиотеки из [requirements.txt](https://github.com/WongKinYiu/yolov7/blob/main/requirements.txt)
```
cd yolov7
pip install -r requirements.txt
```

Для обработки видео необходимо запустить скрипт video_pose из директории репозитория командой
```sh
python video_pose.py -i path/to/original_video.mp4 -o path/to/processed_video.mp4 
```

Пример:
https://www.youtube.com/watch?v=5-cnUadP2YA 