#!/usr/bin/env python3
"""
dashboard_node.py — runs on the LAPTOP
Serves Flask dashboard with live video + tune tab.
Publishes to detector/params for live parameter updates.
"""

import json
import yaml
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Bool, String, Float32MultiArray
import threading
import time
import os

from flask import Flask, Response, render_template_string, jsonify, request


class DashboardNode(Node):
    def __init__(self):
        super().__init__("dashboard_node")

        self.declare_parameter("port", 8080)
        self.port = self.get_parameter("port").value

        # ── Detection state ──────────────────────────────────────────
        self.state = {
            "robot_state":   "UNKNOWN",
            "target_found":  False, "target_name":   "",
            "target_cx":     0.0,   "target_cy":     0.0,
            "target_area":   0.0,   "target_area_f": 0.0, "target_id":  -1,
            "trailer_found": False, "trailer_name":  "",
            "trailer_cx":    0.0,   "trailer_cy":    0.0,
            "trailer_area":  0.0,   "trailer_id":    -1,
            "roi_source":    "none",
            "fps":           0.0,
            "last_update":   time.time(),
        }
        self._frame_times = []

        # ── Current detector params (populated from detector/params/current) ──
        self.detector_params = {}

        # ── Video frames ─────────────────────────────────────────────
        self.frames = {"raw": None, "hsv": None, "mask": None, "zoomed": None}

        # ── Subscribers ──────────────────────────────────────────────
        subs = [
            (Bool,              "aruco/target/found",        self._cb_target_found),
            (Float32MultiArray, "aruco/target/data",         self._cb_target_data),
            (String,            "aruco/target/name",         self._cb_target_name),
            (Float32MultiArray, "aruco/target/data_filtered",self._cb_target_filt),
            (Bool,              "aruco/trailer/found",       self._cb_trailer_found),
            (Float32MultiArray, "aruco/trailer/data",        self._cb_trailer_data),
            (String,            "aruco/trailer/name",        self._cb_trailer_name),
            (Float32MultiArray, "aruco/trailer/data_filtered",self._cb_trailer_filt),
            (String,            "robot/state",               self._cb_robot_state),
            (String,            "aruco/roi_count",           self._cb_roi_source),
            (CompressedImage,   "aruco/debug/raw",           self._cb_raw),
            (CompressedImage,   "aruco/debug/hsv",           self._cb_hsv),
            (CompressedImage,   "aruco/debug/mask",          self._cb_mask),
            (CompressedImage,   "aruco/debug/zoomed",        self._cb_zoomed),
            (String,            "detector/params/current",   self._cb_params_current),
        ]
        for msg_type, topic, cb in subs:
            self.create_subscription(msg_type, topic, cb, 1)

        # ── Publisher: send param updates to detector ─────────────────
        self.params_pub = self.create_publisher(String, "detector/params", 10)

        # ── Flask ────────────────────────────────────────────────────
        self.app = Flask(__name__)
        self._register_routes()
        threading.Thread(
            target=lambda: self.app.run(
                host="0.0.0.0", port=self.port,
                debug=False, use_reloader=False
            ),
            daemon=True
        ).start()
        self.get_logger().info(f"Dashboard at http://0.0.0.0:{self.port}")

    # ── Callbacks ────────────────────────────────────────────────────
    def _cb_target_found(self,  m): self.state["target_found"]  = m.data
    def _cb_target_name(self,   m): self.state["target_name"]   = m.data
    def _cb_target_filt(self,   m): self.state["target_area_f"] = float(list(m.data)[2]) if len(m.data)>2 else 0.0
    def _cb_trailer_found(self, m): self.state["trailer_found"] = m.data
    def _cb_trailer_name(self,  m): self.state["trailer_name"]  = m.data
    def _cb_trailer_filt(self,  m): pass  # reserved
    def _cb_robot_state(self,   m): self.state["robot_state"]   = m.data
    def _cb_roi_source(self,    m): self.state["roi_source"]    = m.data

    def _cb_target_data(self, m):
        d = list(m.data)
        self.state.update({"target_cx": d[0], "target_cy": d[1],
                           "target_area": d[2], "target_id": int(d[3])})

    def _cb_trailer_data(self, m):
        d = list(m.data)
        self.state.update({"trailer_cx": d[0], "trailer_cy": d[1],
                           "trailer_area": d[2], "trailer_id": int(d[3])})

    def _cb_params_current(self, m):
        try:
            self.detector_params = json.loads(m.data)
        except Exception:
            pass

    def _tick_fps(self):
        now = time.time()
        self._frame_times.append(now)
        self._frame_times = [t for t in self._frame_times if now - t < 2.0]
        self.state["fps"] = len(self._frame_times) / 2.0
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
                    os.path.dirname(__file__), "..", "templates", "dashboard.html"
                )
            with open(tmpl_path) as f:
                return render_template_string(f.read())

        @app.route("/api/state")
        def api_state():
            s = dict(self.state)
            s["connected"] = (time.time() - s["last_update"]) < 3.0
            return jsonify(s)

        @app.route("/api/config")
        def api_config():
            """Read aruco_config.yaml directly — reliable at page load."""
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
                # Normalise to {name: {ids:[...], dict:"..."}}
                result = {"targets": {}, "trailers": {}}
                for key in ("targets", "trailers"):
                    for name, val in (cfg.get(key) or {}).items():
                        if isinstance(val, dict):
                            result[key][name] = {
                                "ids":  [int(i) for i in val.get("ids",  [])],
                                "dict": str(val.get("dict", "4x4_50")),
                            }
                        else:
                            result[key][name] = {"ids": [int(i) for i in val], "dict": "4x4_50"}
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
            msg = String()
            msg.data = json.dumps(updates)
            self.params_pub.publish(msg)
            # Optimistically update local cache
            self.detector_params.update(updates)
            return jsonify({"ok": True})

        for key in ["raw", "hsv", "mask", "zoomed"]:
            def make_stream(k):
                def stream():
                    return Response(
                        self._mjpeg_gen(k),
                        mimetype="multipart/x-mixed-replace; boundary=frame"
                    )
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
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" +
                frame + b"\r\n"
            )


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
