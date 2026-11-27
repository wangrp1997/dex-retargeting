"""Record synchronized Botyard preview frames and export four views plus a comparison."""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
from threading import Event, Thread
from time import monotonic
import xml.etree.ElementTree as ET


def fit_view(image, width, height):
    import cv2
    import numpy as np

    h, w = image.shape[:2]
    scale = min(width / w, height / h)
    w, h = max(1, round(w * scale)), max(1, round(h * scale))
    tile = np.zeros((height, width, 3), dtype=np.uint8)
    x, y = (width - w) // 2, (height - h) // 2
    tile[y:y+h, x:x+w] = cv2.resize(image, (w, h))
    return tile


class SessionRecorder:
    def __init__(self, node, directory, scene, cleanup):
        from std_msgs.msg import String
        from rclpy.executors import SingleThreadedExecutor

        self.node = node
        self.latest = None
        self.ready = Event()
        self.sample = None
        self.directory = directory
        self.frames = directory / 'frames'
        self.frames.mkdir(parents=True, exist_ok=False)
        with (directory / 'scene.xml').open('xb') as xml:
            ET.ElementTree(scene).write(xml, encoding='utf-8')
        self.states = cleanup.enter_context((directory / 'sim_states.jsonl').open('x'))
        self.targets = cleanup.enter_context((directory / 'retargeting.jsonl').open('x'))
        log = cleanup.enter_context((directory / 'frames.csv').open('x', newline=''))
        self.log = log
        self.rows = csv.writer(log)
        self.rows.writerow(('frame', 'time_s', 'camera_read_ros_ns', 'sim_view_ros_ns', 'sim_age_ms',
                            'skeleton_detected', 'tracking', 'mano_visible'))
        self.count = 0
        self.start = None
        sub = node.create_subscription(String, 'record_state', self.receive, 1)
        cleanup.callback(node.destroy_subscription, sub)
        # The tracker can spend several hundred ms in a solve. Receive simulated
        # states throughout so the next camera read can use the newest sample.
        self.executor = SingleThreadedExecutor(context=node.context)
        self.executor.add_node(node)
        self.receiver = Thread(target=self.listen, name='gs-s1-recording-state', daemon=True)
        self.receiver.start()
        cleanup.callback(self.stop_receiver)

    def listen(self):
        from rclpy.executors import ExternalShutdownException, ShutdownException
        try:
            self.executor.spin()
        except (ExternalShutdownException, ShutdownException):
            pass

    def stop_receiver(self):
        self.executor.shutdown()
        self.receiver.join()

    def receive(self, msg):
        self.latest = json.loads(msg.data)
        self.ready.set()

    def wait_for_simulator(self):
        import rclpy

        self.node.get_logger().info('Waiting for the MuJoCo simulation state')
        deadline = monotonic() + 45
        while rclpy.ok() and self.latest is None and monotonic() < deadline:
            self.ready.wait(.1)
        if self.latest is None:
            raise RuntimeError('No recording state received from the MuJoCo simulator')
        self.node.get_logger().info(f'Recording in {self.directory}')

    def sample_simulator(self):
        self.read_ns = self.node.get_clock().now().nanoseconds
        self.sample = self.latest

    def write_frame(self, frame, timestamp, detector, landmarks, mano, tracking, command, trace):
        import cv2
        import numpy as np

        skeleton, surface = frame.copy(), frame.copy()
        detected = landmarks is not None
        if detected:
            detector.draw_skeleton_on_image(skeleton, landmarks, style='default')
        visible = tracking and mano.draw_overlay(surface)
        for image, ok, label in ((skeleton, detected, 'Hand not tracked'), (surface, visible, 'MANO unavailable')):
            if not ok:
                cv2.putText(image, label, (10, 48), cv2.FONT_HERSHEY_SIMPLEX, .65, (0, 150, 255), 2)
        h, w = frame.shape[:2]
        # Keep the camera's dimensions. Pad only odd edges for H.264 4:2:0.
        h, w = h + h % 2, w + w % 2
        tiles = [cv2.copyMakeBorder(im, 0, h-im.shape[0], 0, w-im.shape[1], cv2.BORDER_REPLICATE)
                 for im in (frame, skeleton, surface)]
        canvas = np.hstack(tiles)
        path = self.frames / f'{self.count:08d}.jpg'
        if not cv2.imwrite(str(path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 95]):
            raise OSError(f'Could not write recording frame {path}')
        sim_ns = self.sample['stamp_ns']
        self.states.write(json.dumps(dict(frame=self.count, **self.sample)) + "\n")
        self.states.flush()
        if self.start is None:
            self.start = timestamp
        stamp = command.header.stamp
        diagnostic = dict(frame=self.count, time_s=timestamp-self.start, tracking=bool(tracking),
                          command_stamp_ns=stamp.sec * 10**9 + stamp.nanosec,
                          command_joint_names=list(command.name), command_position_rad=list(command.position), **trace)
        self.targets.write(json.dumps(diagnostic, default=lambda a: a.tolist(), separators=(',', ':'),
                                      allow_nan=False) + '\n')
        self.targets.flush()
        self.rows.writerow((self.count, f'{timestamp-self.start:.9f}', self.read_ns, sim_ns,
                            f'{(self.read_ns-sim_ns)/1e6:.3f}', int(detected), int(tracking), int(visible)))
        self.log.flush()
        self.count += 1


def render_session_frames(directory, rows, staging):
    import cv2
    import mujoco
    import numpy as np

    model = mujoco.MjModel.from_xml_path(str(directory / 'scene.xml'))
    data = mujoco.MjData(model)
    camera = mujoco.MjvCamera()
    options = mujoco.MjvOption()
    from gs_s1_mano_overlay import ManoOverlay
    overlay = None
    last_payload = None
    epoch = int(rows[0]['camera_read_ros_ns'])
    last_time = -1.
    with mujoco.Renderer(model, width=640, height=480) as renderer, (directory / 'sim_states.jsonl').open() as states:
        for i, row in enumerate(rows):
            t = float(row['time_s'])
            state = json.loads(states.readline())
            if (int(row['frame']) != i or state['frame'] != i or t <= last_time or
                    state['stamp_ns'] != int(row['sim_view_ros_ns'])):
                raise ValueError(f'Inconsistent recording at frame {i}')
            last_time = t
            data.qpos[:] = state['qpos']
            mujoco.mj_forward(model, data)
            for name, value in state['camera'].items():
                if name == 'lookat':
                    camera.lookat[:] = value
                else:
                    setattr(camera, name, value)
            for name, values in state['options'].items():
                getattr(options, name)[:] = values
            payload = state.get('mano_overlay_pose')
            if payload and overlay is None:
                overlay = ManoOverlay(model)
            if overlay is not None:
                overlay.set_visible(bool(payload))
            renderer.update_scene(data, camera=camera, scene_option=options)
            if payload:
                if payload != last_payload:
                    overlay.update_vertices(payload)
                    mujoco.mjr_uploadMesh(model, renderer._mjr_context, overlay.mesh_id)
                    last_payload = payload
                overlay.draw(renderer.scene, data)
            sim = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)
            if payload:
                cv2.putText(sim, 'Thumb: cyan MANO / orange GS S1', (10, sim.shape[0] - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 1, cv2.LINE_AA)
            name = f'{i:08d}.jpg'
            tiles = cv2.imread(str(directory / 'frames' / name))
            if tiles is None:
                raise ValueError(f'Could not read recorded frame {i}')
            raw, skeleton, surface = np.hsplit(tiles, 3)
            h, w = raw.shape[:2]
            grid = np.vstack((np.hstack((raw, skeleton)), np.hstack((surface, fit_view(sim, w, h)))))
            image = staging / name
            if not cv2.imwrite(str(image), grid, [cv2.IMWRITE_JPEG_QUALITY, 96]):
                raise OSError(f'Could not write export frame {i}')
            # Ubuntu 22.04 FFmpeg image2 reads nanosecond timestamps from mtime.
            stamp = epoch + round(t * 1e9)
            os.utime(image, ns=(stamp, stamp))


def export_videos(directory):
    """Keep captured frames recoverable until every H.264 output has been encoded."""
    directory = directory.resolve()
    with (directory / 'frames.csv').open(newline='') as log:
        rows = list(csv.DictReader(log))
    if not rows:
        raise ValueError('No camera frames were recorded')
    frames = directory / 'frames'
    names = ('human', 'skeleton', 'mano', 'mujoco', 'comparison')
    with TemporaryDirectory(dir=directory) as temp:
        staging = Path(temp)
        render_session_frames(directory, rows, staging)
        filters = ['[0:v]split=5[tl][tr][bl][br][grid]']
        for src, name, x, y in zip(('tl','tr','bl','br'), names, ('0','iw/2','0','iw/2'),
                                   ('0','0','ih/2','ih/2')):
            filters.append(f'[{src}]crop=iw/2:ih/2:{x}:{y}[{name}]')
        labels = [('Human','8','8'), ('MediaPipe','w/2+8','8'),
                  ('MANO','8','h/2+8'), ('MuJoCo','w/2+8','h/2+8')]
        text = [f"drawtext=text='{label}':x={x}:y={y}:fontsize=20:fontcolor=white:box=1:boxcolor=black@0.65"
                for label, x, y in labels]
        filters.append('[grid]' + ','.join(text) + '[comparison]')
        cmd = ['ffmpeg', '-nostdin', '-hide_banner', '-loglevel', 'error', '-f', 'image2',
               '-ts_from_file', '2', '-i', str(staging / '%08d.jpg'), '-filter_complex_threads', '2', '-filter_complex',
               ';'.join(filters), '-vsync', 'vfr']
        for name in names:
            cmd.extend(['-map', f'[{name}]', '-an', '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '18',
                        '-threads', '2', '-pix_fmt', 'yuv420p', '-enc_time_base', '1:1000',
                        '-movflags', '+faststart', str(staging / f'{name}.mp4')])
        subprocess.run(cmd, check=True)
        for name in names:
            (staging / f'{name}.mp4').replace(directory / f'{name}.mp4')
    for i in range(len(rows)):
        (frames / f'{i:08d}.jpg').unlink()
    if not any(frames.iterdir()):
        frames.rmdir()
    print(f'{len(rows)} synchronized frames saved in {directory}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Export a captured Botyard recording session to MP4')
    parser.add_argument('directory', type=Path)
    export_videos(parser.parse_args().directory)
