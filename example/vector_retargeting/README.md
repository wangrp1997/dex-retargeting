## Retarget Robot Motion from Human Hand Video

![teaser](teaser.webp)

### Generate the robot joint pose trajectory from our pre-recorded video

```shell
cd example/vector_retargeting
python3 detect_from_video.py \
  --robot-name allegro \
  --video-path data/human_hand_video.mp4 \
  --retargeting-type dexpilot \
  --hand-type right \
  --output-path data/allegro_joints.pkl 
```

This command will output the joint trajectory as a pickle file at the `output_path`.

The pickle file is a python dictionary with two keys: `meta_data` and `data`. `meta_data`, a dictionary, includes
details about the robot, while `data`, a list, contains the robotic joint positions for each frame. For additional
options, refer to the help information. Note that the time cost here includes both the hand pose detection from video,
and the hand pose retargeting in single process mode.

```shell
python3 detect_from_video.py --help
```

### Utilize the pickle file to produce a video of the robot

```shell
python3 render_robot_hand.py \
  --pickle-path data/allegro_joints.pkl \
  --output-video-path data/allegro.mp4 \
  --headless
```

This command uses the data saved from the previous step to create a rendered video.

### Capture a Video Using Your Webcam

*The following instructions assume that your computer has a webcam connected.*

```bash
python3 capture_webcam.py --video-path data/my_human_hand_video.mp4
```

This command enables you to use your webcam to record a video saved in MP4 format. To stop recording, press `Esc` on your
keyboard.

### Real-time Visualization of Hand Retargeting via Webcam

```bash
pip install loguru
python3 show_realtime_retargeting.py \
  --robot-name allegro \
  --retargeting-type dexpilot \
  --hand-type right 
```

This process integrates the tasks described above. It involves capturing your hand movements through the webcam and
instantaneously displaying the retargeting outcomes in the SAPIEN viewer. Special thanks
to [@xbkaishui](https://github.com/xbkaishui) for contributing the initial pull request.

![realtime_example](data/realtime_example.webp)


### Capture Video Using OAK-D Camera

*The following instructions assume you have an OAK-D camera connected.*

First, ensure you have the proper permissions set up for the OAK-D camera:
```bash
echo 'SUBSYSTEM=="usb", ATTRS{idVendor}=="03e7", MODE="0666"' | sudo tee /etc/udev/rules.d/80-movidius.rules
sudo udevadm control --reload-rules && sudo udevadm trigger
```

#### Basic RGB Video Recording
```bash
python3 capture_oak.py \
  --video-path data/my_human_hand_video.mp4 \
  --no-save-depth \
  --resolution full
```

#### Recording with Depth Data
```bash
python3 capture_oak.py \
  --video-path data/my_human_hand_video.mp4 \
  --save-depth \
  --resolution full
```
This will generate two files:
- `data/my_human_hand_video.mp4`: RGB video
- `data/my_human_hand_video_depth.mp4`: Depth video

#### Available Options
- `--resolution`: Choose between "full" (1280x800) or "ultra" (3840x2160)
- `--save-depth`: Enable depth recording
- `--no-save-depth`: Disable depth recording
- `--internal-fps`: Set custom frame rate
- `--crop`: Enable square crop mode
- `--internal-frame-height`: Set internal processing height (default: 640)

To stop recording, press `Esc` on your keyboard.

Note: The depth data is automatically normalized and saved in a viewable format. When using the depth video for further processing, please be aware that the values have been compressed to 8-bit format.


