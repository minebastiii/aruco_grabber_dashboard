#!/usr/bin/env python3
"""
dashboard_node.py — runs on the LAPTOP
"""

import json
import yaml
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Bool, String, Float32MultiArray, Float32
import threading
import time
import os

from flask import Flask, Response, jsonify, request


class DashboardNode(Node):
    def __init__(self):
        super().__init__("dashboard_node")

        self.declare_parameter("port", 8080)
        self.port = self.get_parameter("port").value

        self._state_lock = threading.Lock()

        self.state = {
            "robot_state":      "UNKNOWN",
            "target_found":     False, "target_name":     "",
            "target_cx":        0.0,   "target_cy":       0.0,
            "target_area":      0.0,   "target_id":       -1,
            "target_cx_f":      0.0,   "target_area_f":   0.0,
            "target_distance":  -1.0,
            "trailer_found":    False, "trailer_name":    "",
            "trailer_cx":       0.0,   "trailer_cy":      0.0,
            "trailer_area":     0.0,   "trailer_id":      -1,
            "trailer_cx_f":     0.0,   "trailer_area_f":  0.0,
            "trailer_distance": -1.0,
            "roi_source":       "none",
            "fps":              0.0,
            "last_update":      time.time(),
            "last_state_msg":   0.0,
            # FSM control mode (filled from fsm/control_mode topic)
            "use_distance_target":  True,
            "use_distance_trailer": True,
            "grasp_distance_m":     0.20,
            "deploy_distance_m":    0.25,
            # Distance-mode approach controller
            "approach_dist_gain":   8.0,
            "approach_step_min":    0.3,
            "approach_step_max":    2.0,
            # Robot battery level (0-100, -1 = unknown)
            "battery":              -1.0,
        }
        self._frame_times   = []
        self.detector_params = {}
        self.frames = {"raw": None, "hsv": None, "mask": None, "zoomed": None}

        subs = [
            (Bool,              "aruco/target/found",          self._cb_target_found),
            (Float32MultiArray, "aruco/target/data",           self._cb_target_data),
            (String,            "aruco/target/name",           self._cb_target_name),
            (Float32MultiArray, "aruco/target/data_filtered",  self._cb_target_filt),
            (Float32,           "aruco/target/distance",       self._cb_target_dist),
            (Bool,              "aruco/trailer/found",         self._cb_trailer_found),
            (Float32MultiArray, "aruco/trailer/data",          self._cb_trailer_data),
            (String,            "aruco/trailer/name",          self._cb_trailer_name),
            (Float32MultiArray, "aruco/trailer/data_filtered", self._cb_trailer_filt),
            (Float32,           "aruco/trailer/distance",      self._cb_trailer_dist),
            (String,            "robot/state",                 self._cb_robot_state),
            (String,            "aruco/roi_count",             self._cb_roi_source),
            (String,            "fsm/control_mode",            self._cb_control_mode),
            (CompressedImage,   "aruco/debug/raw",             self._cb_raw),
            (CompressedImage,   "aruco/debug/hsv",             self._cb_hsv),
            (CompressedImage,   "aruco/debug/mask",            self._cb_mask),
            (CompressedImage,   "aruco/debug/zoomed",          self._cb_zoomed),
            (String,            "detector/params/current",     self._cb_params_current),
            (Float32,           "robot/battery",               self._cb_battery),
        ]
        for msg_type, topic, cb in subs:
            self.create_subscription(msg_type, topic, cb, 1)

        self.create_timer(1.0, self._check_state_staleness)

        self.params_pub  = self.create_publisher(String, "detector/params", 10)
        self.control_pub = self.create_publisher(String, "fsm/control",     10)

        self.app = Flask(__name__)
        self._register_routes()
        threading.Thread(
            target=lambda: self.app.run(
                host="0.0.0.0", port=self.port, debug=False, use_reloader=False),
            daemon=True
        ).start()
        self.get_logger().info(f"Dashboard at http://0.0.0.0:{self.port}")

    # ── Callbacks ────────────────────────────────────────────────────
    def _cb_target_found(self,  m):
        with self._state_lock: self.state["target_found"] = m.data
    def _cb_target_name(self,   m):
        with self._state_lock: self.state["target_name"]  = m.data
    def _cb_trailer_found(self, m):
        with self._state_lock: self.state["trailer_found"] = m.data
    def _cb_trailer_name(self,  m):
        with self._state_lock: self.state["trailer_name"]  = m.data
    def _cb_roi_source(self,    m):
        with self._state_lock: self.state["roi_source"] = m.data

    def _cb_robot_state(self, m):
        with self._state_lock:
            self.state["robot_state"]    = m.data
            self.state["last_state_msg"] = time.time()

    def _cb_control_mode(self, m):
        try:
            d = json.loads(m.data)
            with self._state_lock:
                for k in ("use_distance_target", "use_distance_trailer",
                          "grasp_distance_m", "deploy_distance_m",
                          "approach_dist_gain", "approach_step_min", "approach_step_max"):
                    if k in d:
                        self.state[k] = d[k]
        except Exception:
            pass

    def _cb_target_filt(self, m):
        d = list(m.data)
        if len(d) >= 3:
            with self._state_lock:
                self.state["target_cx_f"]   = float(d[0])
                self.state["target_area_f"] = float(d[2])

    def _cb_trailer_filt(self, m):
        d = list(m.data)
        if len(d) >= 3:
            with self._state_lock:
                self.state["trailer_cx_f"]   = float(d[0])
                self.state["trailer_area_f"] = float(d[2])

    def _cb_target_dist(self, m):
        with self._state_lock: self.state["target_distance"] = float(m.data)

    def _cb_trailer_dist(self, m):
        with self._state_lock: self.state["trailer_distance"] = float(m.data)

    def _cb_target_data(self, m):
        d = list(m.data)
        with self._state_lock:
            self.state["target_cx"]   = float(d[0])
            self.state["target_cy"]   = float(d[1])
            self.state["target_area"] = float(d[2])
            self.state["target_id"]   = int(d[3])

    def _cb_trailer_data(self, m):
        d = list(m.data)
        with self._state_lock:
            self.state["trailer_cx"]   = float(d[0])
            self.state["trailer_cy"]   = float(d[1])
            self.state["trailer_area"] = float(d[2])
            self.state["trailer_id"]   = int(d[3])

    def _cb_params_current(self, m):
        try: self.detector_params = json.loads(m.data)
        except Exception: pass

    def _cb_battery(self, m):
        with self._state_lock: self.state["battery"] = float(m.data)

    def _check_state_staleness(self):
        with self._state_lock:
            last = self.state["last_state_msg"]
            if last > 0.0 and (time.time() - last) > 5.0:
                self.state["robot_state"]    = "UNKNOWN"
                self.state["last_state_msg"] = 0.0

    def _tick_fps(self):
        now = time.time()
        self._frame_times.append(now)
        self._frame_times = [t for t in self._frame_times if now - t < 2.0]
        with self._state_lock:
            self.state["fps"]         = len(self._frame_times) / 2.0
            self.state["last_update"] = now

    def _cb_raw(self,    m): self.frames["raw"]    = bytes(m.data); self._tick_fps()
    def _cb_hsv(self,    m): self.frames["hsv"]    = bytes(m.data)
    def _cb_mask(self,   m): self.frames["mask"]   = bytes(m.data)
    def _cb_zoomed(self, m): self.frames["zoomed"] = bytes(m.data)

    # ── Flask routes ─────────────────────────────────────────────────
    def _register_routes(self):
        app = self.app

        @app.route("/")
        def index():
            try:
                from ament_index_python.packages import get_package_share_directory
                pkg_share = get_package_share_directory("aruco_laptop")
                tmpl_path = os.path.join(pkg_share, "templates", "dashboard.html")
            except Exception:
                tmpl_path = os.path.join(
                    os.path.dirname(__file__), "..", "templates", "dashboard.html")
            with open(tmpl_path) as f:
                return f.read(), 200, {"Content-Type": "text/html; charset=utf-8"}

        @app.route("/api/state")
        def api_state():
            with self._state_lock:
                s = dict(self.state)
            s["connected"] = (time.time() - s["last_update"]) < 5.0
            s.pop("last_state_msg", None)
            return jsonify(s)

        @app.route("/api/config")
        def api_config():
            try:
                from ament_index_python.packages import get_package_share_directory
                pkg_share = get_package_share_directory("aruco_laptop")
                cfg_path  = os.path.join(pkg_share, "config", "aruco_config.yaml")
            except Exception:
                cfg_path = os.path.join(
                    os.path.dirname(__file__), "..", "config", "aruco_config.yaml")
            try:
                with open(cfg_path) as f:
                    cfg = yaml.safe_load(f)
                result = {"targets": {}, "trailers": {}}
                for key in ("targets", "trailers"):
                    for name, val in (cfg.get(key) or {}).items():
                        if isinstance(val, dict):
                            entry = {
                                "ids":  [int(i) for i in val.get("ids",  [])],
                                "dict": str(val.get("dict", "4x4_50")),
                            }
                            if val.get("marker_size_m") is not None:
                                entry["marker_size_m"] = float(val["marker_size_m"])
                            if key == "trailers" and val.get("far_ids"):
                                entry["far_ids"]  = [int(i) for i in val["far_ids"]]
                                entry["far_dict"] = str(val.get("far_dict",
                                                              val.get("dict", "4x4_50")))
                                if val.get("far_marker_size_m") is not None:
                                    entry["far_marker_size_m"] = float(val["far_marker_size_m"])
                            result[key][name] = entry
                        else:
                            result[key][name] = {"ids": [int(i) for i in val],
                                                 "dict": "4x4_50"}
                return jsonify(result)
            except Exception as e:
                return jsonify({"error": str(e), "targets": {}, "trailers": {}}), 500

        @app.route("/api/params", methods=["GET"])
        def api_params_get():
            return jsonify(self.detector_params)

        @app.route("/api/params", methods=["POST"])
        def api_params_set():
            updates = request.get_json()
            if not updates:
                return jsonify({"error": "no data"}), 400
            msg = String(); msg.data = json.dumps(updates)
            self.params_pub.publish(msg)
            self.detector_params.update(updates)
            return jsonify({"ok": True})

        @app.route("/api/fsm_control", methods=["POST"])
        def api_fsm_control():
            cmd = request.get_json()
            if not cmd:
                return jsonify({"error": "no data"}), 400
            msg = String(); msg.data = json.dumps(cmd)
            self.control_pub.publish(msg)
            # Optimistic local update
            with self._state_lock:
                for k in ("use_distance_target", "use_distance_trailer",
                          "grasp_distance_m", "deploy_distance_m",
                          "approach_dist_gain", "approach_step_min", "approach_step_max"):
                    if k in cmd:
                        self.state[k] = cmd[k]
            return jsonify({"ok": True})

        for key in ["raw", "hsv", "mask", "zoomed"]:
            def make_stream(k):
                def stream():
                    return Response(self._mjpeg_gen(k),
                                    mimetype="multipart/x-mixed-replace; boundary=frame")
                stream.__name__ = f"stream_{k}"
                return stream
            app.add_url_rule(f"/stream/{key}", f"stream_{key}", make_stream(key))

    def _mjpeg_gen(self, key):
        last_sent = None
        while True:
            frame = self.frames[key]
            if frame is None or frame is last_sent:
                time.sleep(0.005)
                continue
            last_sent = frame
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")


def main(args=None):
    rclpy.init(args=args)
    node = DashboardNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
