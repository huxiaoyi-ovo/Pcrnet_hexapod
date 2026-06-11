#!/usr/bin/env python3
"""Joystick-triggered rosbag recorder for PCR real-robot experiments."""

import argparse
import datetime as dt
import json
import os
import shutil
import signal
import subprocess
import threading
import time
from pathlib import Path
from typing import Optional


DEFAULT_TOPICS = [
    "/pcr_realplay/debug",
    "/pcr/target_state",
    "/pcr/local_map_2ch",
    "/pcr/raw_occ_map",
    "/pcr/memory_occ_map",
    "/usr/command_pcr",
    "/sita_des",
    "/left_sita_cur",
    "/right_sita_cur",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Toggle a minimal PCR experiment rosbag from a joystick button."
    )
    parser.add_argument("--button_index", type=int, default=12)
    parser.add_argument("--joy_topic", type=str, default="/joy")
    parser.add_argument("--output_root", type=str, default="~/pcr_records")
    parser.add_argument("--min_free_gb", type=float, default=2.0)
    parser.add_argument("--debounce_s", type=float, default=0.5)
    parser.add_argument(
        "--topic",
        action="append",
        default=[],
        help="Additional topic to record. May be repeated.",
    )
    args, _unknown = parser.parse_known_args()
    return args


class PcrSessionRecorder:
    def __init__(self, args, rospy, Joy, Bool, String) -> None:
        self.args = args
        self.rospy = rospy
        self.Bool = Bool
        self.String = String
        self.lock = threading.RLock()
        self.recording = False
        self.last_button = False
        self.last_toggle_wall = 0.0
        self.session_dir: Optional[Path] = None
        self.bag_process: Optional[subprocess.Popen] = None
        self.started_wall = 0.0
        self.started_ros = 0.0
        self.topics = list(dict.fromkeys(DEFAULT_TOPICS + list(args.topic)))

        self.status_pub = rospy.Publisher("/pcr/recording", Bool, queue_size=1, latch=True)
        self.session_pub = rospy.Publisher(
            "/pcr/recording_session", String, queue_size=1, latch=True
        )
        self.joy_sub = rospy.Subscriber(
            args.joy_topic, Joy, self._joy_callback, queue_size=1
        )
        self.health_timer = rospy.Timer(rospy.Duration(2.0), self._health_check)
        rospy.on_shutdown(self.shutdown)
        self._publish_status()
        rospy.loginfo(
            "[PCRRecorder] ready: button[%d] toggles recording, output=%s",
            int(args.button_index),
            os.path.expanduser(args.output_root),
        )

    def _publish_status(self) -> None:
        path = "" if self.session_dir is None else str(self.session_dir)
        # Publish the directory first so camera-side recording never starts
        # without knowing where the viewer files belong.
        self.session_pub.publish(self.String(data=path))
        self.status_pub.publish(self.Bool(data=bool(self.recording)))

    def _joy_callback(self, msg) -> None:
        index = int(self.args.button_index)
        pressed = index >= 0 and index < len(msg.buttons) and bool(msg.buttons[index])
        now = time.monotonic()
        rising = pressed and not self.last_button
        self.last_button = pressed
        if not rising or now - self.last_toggle_wall < float(self.args.debounce_s):
            return
        self.last_toggle_wall = now
        if self.recording:
            self.stop_recording("joystick")
        else:
            self.start_recording()

    def _new_session_dir(self) -> Path:
        root = Path(os.path.expanduser(self.args.output_root)).resolve()
        root.mkdir(parents=True, exist_ok=True)
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = root / stamp
        suffix = 1
        while session_dir.exists():
            session_dir = root / f"{stamp}_{suffix:02d}"
            suffix += 1
        session_dir.mkdir(parents=True)
        return session_dir

    @staticmethod
    def _free_gb(path: Path) -> float:
        return float(shutil.disk_usage(str(path)).free) / float(1024 ** 3)

    def _write_metadata(self, stopped: bool, reason: str) -> None:
        if self.session_dir is None:
            return
        payload = {
            "session_dir": str(self.session_dir),
            "recording_button_index": int(self.args.button_index),
            "joy_topic": self.args.joy_topic,
            "topics": self.topics,
            "bag_buffer_mb": 256,
            "started_ros_time": float(self.started_ros),
            "started_wall_time": float(self.started_wall),
            "stopped_ros_time": (
                float(self.rospy.Time.now().to_sec()) if stopped else None
            ),
            "stopped_wall_time": time.time() if stopped else None,
            "stop_reason": reason if stopped else None,
        }
        tmp_path = self.session_dir / "session.json.tmp"
        final_path = self.session_dir / "session.json"
        try:
            with open(tmp_path, "w") as handle:
                json.dump(payload, handle, indent=2, sort_keys=True)
            os.replace(str(tmp_path), str(final_path))
        except OSError as exc:
            self.rospy.logerr("[PCRRecorder] failed to write session metadata: %s", exc)

    @staticmethod
    def _stop_process(process: Optional[subprocess.Popen]) -> None:
        if process is None or process.poll() is not None:
            return
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGINT)
            process.wait(timeout=12.0)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                process.wait(timeout=3.0)
            except (OSError, subprocess.TimeoutExpired):
                pass
        except OSError:
            pass

    def start_recording(self) -> None:
        with self.lock:
            if self.recording:
                return
            session_dir = self._new_session_dir()
            free_gb = self._free_gb(session_dir)
            if free_gb < float(self.args.min_free_gb):
                self.rospy.logerr(
                    "[PCRRecorder] not started: %.2f GiB free, require %.2f GiB",
                    free_gb,
                    float(self.args.min_free_gb),
                )
                return

            bag_path = session_dir / "experiment.bag"
            command = [
                "rosbag",
                "record",
                "--buffsize=256",
                "-O",
                str(bag_path),
            ] + self.topics
            try:
                bag_process = subprocess.Popen(command, preexec_fn=os.setsid)
            except OSError as exc:
                self.rospy.logerr("[PCRRecorder] failed to start rosbag: %s", exc)
                return

            self.session_dir = session_dir
            self.bag_process = bag_process
            self.started_wall = time.time()
            self.started_ros = float(self.rospy.Time.now().to_sec())
            self.recording = True
            self._write_metadata(stopped=False, reason="")
            self._publish_status()
            self.rospy.logwarn("[PCRRecorder] RECORDING STARTED: %s", session_dir)

    def stop_recording(self, reason: str) -> None:
        with self.lock:
            if not self.recording:
                return
            self.recording = False
            process = self.bag_process
            self.bag_process = None
            session_dir = self.session_dir

            # Tell the camera writer to finish immediately, then wait for
            # rosbag to flush independently.
            self._publish_status()
            self._stop_process(process)
            self._write_metadata(stopped=True, reason=reason)
            self.rospy.logwarn(
                "[PCRRecorder] RECORDING STOPPED: %s, reason=%s",
                session_dir,
                reason,
            )

    def _health_check(self, _event) -> None:
        with self.lock:
            if not self.recording:
                return
            if self.bag_process is not None and self.bag_process.poll() is not None:
                self.rospy.logerr("[PCRRecorder] rosbag exited unexpectedly")
                self.stop_recording("rosbag_exited")
                return
            if self.session_dir is not None:
                free_gb = self._free_gb(self.session_dir)
                if free_gb < float(self.args.min_free_gb):
                    self.rospy.logerr(
                        "[PCRRecorder] stopping: disk free %.2f GiB below %.2f GiB",
                        free_gb,
                        float(self.args.min_free_gb),
                    )
                    self.stop_recording("low_disk")

    def shutdown(self) -> None:
        self.stop_recording("node_shutdown")


def main() -> None:
    args = parse_args()
    try:
        import rospy
        from sensor_msgs.msg import Joy
        from std_msgs.msg import Bool
        from std_msgs.msg import String
    except ImportError as exc:
        raise SystemExit("ROS1 Python packages are required; source the workspace first.") from exc

    rospy.init_node("pcr_session_recorder", anonymous=False)
    PcrSessionRecorder(args, rospy, Joy, Bool, String)
    rospy.spin()


if __name__ == "__main__":
    main()
