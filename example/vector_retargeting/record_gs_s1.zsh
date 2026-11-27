#!/usr/bin/env zsh
# Run from any directory. Extra arguments are ROS launch arguments such as hand:=left.
set -e
example_dir=${0:A:h}
source /opt/ros/humble/setup.zsh
source "$HOME/botyard_ws/install/setup.zsh"
command -v ffmpeg >/dev/null || { print -u2 'ffmpeg is required to export the recording'; exit 1; }
output_root="$HOME/Videos/botyard"
mkdir -p "$output_root"
session_dir=$(mktemp -d "$output_root/$(date +%Y%m%d-%H%M%S)-XXXXXX")
print -r -- "Recording directory: $session_dir"
# The launch receives Ctrl+C as well. Allow it to close the frame log before export.
trap ':' INT
launch_status=0
ros2 launch "$example_dir/gs_s1_preview.launch.py" backend:=mujoco objective:=botyard \
    "$@" record_dir:="$session_dir" || launch_status=$?
trap - INT
if [[ -s "$session_dir/frames/00000000.jpg" ]]; then
    print 'Exporting synchronized MP4 files...'
    MUJOCO_GL=${MUJOCO_GL:-osmesa} "$HOME/miniconda3/envs/dex_retargeting/bin/python" "$example_dir/gs_s1_recording.py" "$session_dir"
else
    print -u2 'No frames were recorded.'
    exit 1
fi
exit "$launch_status"
