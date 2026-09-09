#!/usr/bin/env python3
"""
dashboard_node.py — runs on the LAPTOP/server

Receives the plain camera image (camera/compressed) from the robot and
draws the marker overlay (detected IDs, ROI box, tracked target/trailer
markers) on it for display. All colour-mask / zoomed-crop debug
visualization has been removed — it was never used by the FSM and only
existed for a debug view nobody needed any more.

Still supports settings pushed from the UI:
  - FSM-relevant tuning (filter_alpha, persist_ttl, camera intrinsics,
    marker sizes, target/trailer config) -> forwarded to camera/params
    on the robot.
  - FSM approach-mode / thresholds -> fsm/control on the robot.
"""

import json
import os
import threading
import time

import cv2
import numpy as np
import rclpy
from flask import Flask, Response, jsonify, request
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Bool, Float32, Float32MultiArray, String

# ── ArUco dictionary map (duplicated here so the dashboard can run its
#    own detection purely for the overlay) ──────────────────────────
DICT_MAP = {
    "4x4_50": cv2.aruco.DICT_4X4_50,
    "4x4_100": cv2.aruco.DICT_4X4_100,
    "4x4_250": cv2.aruco.DICT_4X4_250,
    "4x4_1000": cv2.aruco.DICT_4X4_1000,
    "5x5_50": cv2.aruco.DICT_5X5_50,
    "5x5_100": cv2.aruco.DICT_5X5_100,
    "5x5_250": cv2.aruco.DICT_5X5_250,
    "5x5_1000": cv2.aruco.DICT_5X5_1000,
    "6x6_50": cv2.aruco.DICT_6X6_50,
    "6x6_100": cv2.aruco.DICT_6X6_100,
    "6x6_250": cv2.aruco.DICT_6X6_250,
    "6x6_1000": cv2.aruco.DICT_6X6_1000,
    "7x7_50": cv2.aruco.DICT_7X7_50,
    "7x7_100": cv2.aruco.DICT_7X7_100,
    "7x7_250": cv2.aruco.DICT_7X7_250,
    "7x7_1000": cv2.aruco.DICT_7X7_1000,
    "original": cv2.aruco.DICT_ARUCO_ORIGINAL,
}


def _make_det_params():
    if hasattr(cv2.aruco, "DetectorParameters"):
        p = cv2.aruco.DetectorParameters()
    else:
        p = cv2.aruco.DetectorParameters_create()
    p.adaptiveThreshWinSizeMin = 3
    p.adaptiveThreshWinSizeMax = 53
    p.adaptiveThreshWinSizeStep = 10
    p.adaptiveThreshConstant = 7
    p.minMarkerPerimeterRate = 0.02
    p.maxMarkerPerimeterRate = 4.0
    p.polygonalApproxAccuracyRate = 0.05
    p.minCornerDistanceRate = 0.02
    p.minMarkerDistanceRate = 0.02
    p.errorCorrectionRate = 1.0
    p.cornerRefinementMethod = (
        cv2.aruco.CORNER_REFINE_SUBPIX if hasattr(cv2.aruco, "CORNER_REFINE_SUBPIX") else 1
    )
    p.cornerRefinementWinSize = 5
    p.cornerRefinementMaxIterations = 30
    p.cornerRefinementMinAccuracy = 0.1
    return p


def build_detector(dict_name: str):
    dict_id = DICT_MAP.get(dict_name, cv2.aruco.DICT_4X4_50)
    p = _make_det_params()
    if hasattr(cv2.aruco, "getPredefinedDictionary"):
        d = cv2.aruco.getPredefinedDictionary(dict_id)
        if hasattr(cv2.aruco, "ArucoDetector"):
            return ("new", cv2.aruco.ArucoDetector(d, p))
        return ("mid", (d, p))
    d = cv2.aruco.Dictionary_get(dict_id)
    return ("old", (d, p))


def run_detector(det_tuple, gray):
    mode, det = det_tuple
    if mode == "new":
        corners, ids, _ = det.detectMarkers(gray)
    else:
        d, p = det
        corners, ids, _ = cv2.aruco.detectMarkers(gray, d, parameters=p)
    return corners, ids


def entry_ids(e):
    return e.get("ids", []) if isinstance(e, dict) else list(e)


def entry_dict_name(e):
    return e.get("dict", "4x4_50") if isinstance(e, dict) else "4x4_50"


# Settings forwarded to the robot via camera/params.
ROBOT_PARAM_KEYS = {
    "filter_alpha", "persist_ttl", "targets", "trailers",
    "cam_fx", "cam_fy", "cam_cx", "cam_cy", "cam_k1", "cam_k2", "cam_p1", "cam_p2",
    "target_marker_size_m", "trailer_marker_size_m",
}


class DashboardNode(Node):
    def __init__(self):
        super().__init__("dashboard_node")

        self.declare_parameter("port", 8080)
        self.port = self.get_parameter("port").value
        self.domain_id = os.environ.get("ROS_DOMAIN_ID", "0")

        self._state_lock = threading.Lock()

        self.state = {
            "robot_state":      "UNKNOWN",
            "target_found":     False, "target_name":     "",
            "target_cx":        0.0,   "target_cy":       0.0,
            "target_area":      0.0,   "target_id":       -1,
            "target_cx_f":      0.0,   "target_cy_f":     0.0, "target_area_f": 0.0,
            "target_distance":  -1.0,
            "trailer_found":    False, "trailer_name":    "",
            "trailer_cx":       0.0,   "trailer_cy":      0.0,
            "trailer_area":     0.0,   "trailer_id":      -1,
            "trailer_cx_f":     0.0,   "trailer_cy_f":    0.0, "trailer_area_f": 0.0,
            "trailer_distance": -1.0,
            "roi_source":       "none",
            "fps":              0.0,
            "last_update":      time.time(),
            "last_state_msg":   0.0,
            # FSM control mode (from fsm/control_mode)
            "use_distance_target":  True,
            "use_distance_trailer": True,
            "grasp_distance_m":     0.20,
            "deploy_distance_m":    0.25,
            "approach_dist_gain":   8.0,
            "approach_step_min":    0.3,
            "approach_step_max":    2.0,
            "battery":              -1.0,
        }
        # Last "camera/params/current" payload from the robot
        # (filter_alpha, persist_ttl, targets, trailers, cam_*, marker
        # sizes, dict_names) — this is exactly what the Settings tab shows.
        self.camera_params = {}

        self._frame_times = []
        self._det_cache = {}
        self.frame = None  # latest encoded overlay JPEG

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
            (String,            "fsm/control_mode",            self._cb_control_mode),
            (String,            "camera/params/current",       self._cb_params_current),
            (Float32,           "robot/battery",               self._cb_battery),
            (CompressedImage,   "camera/compressed",           self._cb_camera_image),
        ]
        for msg_type, topic, cb in subs:
            self.create_subscription(msg_type, topic, cb, 1)

        self.create_timer(1.0, self._check_state_staleness)

        self.params_pub  = self.create_publisher(String, "camera/params", 10)
        self.control_pub = self.create_publisher(String, "fsm/control",   10)

        self.app = Flask(__name__)
        self._register_routes()
        threading.Thread(
            target=lambda: self.app.run(
                host="0.0.0.0", port=self.port, debug=False, use_reloader=False),
            daemon=True
        ).start()
        self.get_logger().info(f"Dashboard at http://0.0.0.0:{self.port}  (ROS_DOMAIN_ID={self.domain_id})")

    # ── ROS callbacks — robot-reported FSM state ──────────────────────
    def _cb_target_found(self,  m):
        with self._state_lock: self.state["target_found"] = m.data
    def _cb_target_name(self,   m):
        with self._state_lock: self.state["target_name"]  = m.data
    def _cb_trailer_found(self, m):
        with self._state_lock: self.state["trailer_found"] = m.data
    def _cb_trailer_name(self,  m):
        with self._state_lock: self.state["trailer_name"]  = m.data

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
                self.state["target_cy_f"]   = float(d[1])
                self.state["target_area_f"] = float(d[2])

    def _cb_trailer_filt(self, m):
        d = list(m.data)
        if len(d) >= 3:
            with self._state_lock:
                self.state["trailer_cx_f"]   = float(d[0])
                self.state["trailer_cy_f"]   = float(d[1])
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
        try:
            self.camera_params = json.loads(m.data)
        except Exception:
            pass

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

    # ── Main image callback — draws the marker overlay ────────────────
    def _cb_camera_image(self, msg: CompressedImage):
        buf = np.frombuffer(msg.data, dtype=np.uint8)
        frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if frame is None:
            return
        self._tick_fps()
        self._process_overlay(frame)

    def _get_det(self, dict_name: str):
        if dict_name not in self._det_cache:
            self._det_cache[dict_name] = build_detector(dict_name)
        return self._det_cache[dict_name]

    def _detect_all(self, gray, targets, trailers):
        needed = {entry_dict_name(e) for d in (targets, trailers) for e in d.values()}
        for e in trailers.values():
            fd = e.get("far_dict")
            if fd:
                needed.add(fd)
        result = {}
        for dn in needed:
            corners_list, ids = run_detector(self._get_det(dn), gray)
            dets = []
            if ids is not None:
                for corner, mid in zip(corners_list, ids.flatten()):
                    pts = corner[0]
                    cx = float(np.mean(pts[:, 0]))
                    cy = float(np.mean(pts[:, 1]))
                    side = float(np.linalg.norm(pts[0] - pts[1]))
                    dets.append((cx, cy, side * side, int(mid), pts))
            result[dn] = dets
        return result

    def _roi_from_detections(self, tracked_dets, target_ids, trailer_ids, frame_shape):
        for id_set in (target_ids, trailer_ids):
            cands = [d for d in tracked_dets if d[3] in id_set]
            if not cands:
                continue
            cx, cy, area, mid = min(cands, key=lambda d: d[3])
            side = area ** 0.5
            H, W = frame_shape[:2]
            pad = max(12, int(side * 0.5))
            x = max(0, int(cx - side) - pad)
            y = max(0, int(cy - side) - pad)
            w = min(W - x, int(side * 2) + 2 * pad)
            h = min(H - y, int(side * 2) + 2 * pad)
            return (x, y, w, h) if w >= 4 and h >= 4 else None
        return None

    def _process_overlay(self, frame):
        targets = self.camera_params.get("targets", {})
        trailers = self.camera_params.get("trailers", {})

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        det_by_dict = self._detect_all(gray, targets, trailers) if (targets or trailers) else {}

        all_target_ids = {i for e in targets.values() for i in entry_ids(e)}
        all_trailer_ids = {i for e in trailers.values() for i in entry_ids(e)}
        all_trailer_far_ids = {int(i) for e in trailers.values() for i in e.get("far_ids", [])}

        tracked_on_full = []
        for entry in list(targets.values()) + list(trailers.values()):
            dn = entry_dict_name(entry)
            for d in det_by_dict.get(dn, []):
                if d[3] in entry_ids(entry):
                    tracked_on_full.append(d)
            for far_id in entry.get("far_ids", []):
                far_dn = entry.get("far_dict", entry_dict_name(entry))
                for d in det_by_dict.get(far_dn, []):
                    if d[3] == int(far_id):
                        tracked_on_full.append(d)

        roi = None
        roi_source = "none"
        if tracked_on_full:
            roi = self._roi_from_detections(
                [(d[0], d[1], d[2], d[3]) for d in tracked_on_full],
                all_target_ids, all_trailer_ids | all_trailer_far_ids, frame.shape,
            )
            roi_source = "marker" if roi is not None else "none"

        vis = frame.copy()
        if roi is not None:
            rx, ry, rw, rh = roi
            cv2.rectangle(vis, (rx, ry), (rx + rw, ry + rh), (0, 220, 80), 2)
            cv2.putText(vis, roi_source, (rx + 4, ry + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 80), 2)

        for dn, dets in det_by_dict.items():
            for cx, cy, a, mid, *_ in dets:
                if mid in all_target_ids:
                    col, label = (0, 220, 80), f"ID:{mid}"
                elif mid in all_trailer_ids:
                    col, label = (0, 100, 255), f"ID:{mid}"
                elif mid in all_trailer_far_ids:
                    col, label = (255, 200, 0), f"FAR:{mid}"
                else:
                    col, label = (50, 200, 200), f"ID:{mid}"
                cv2.circle(vis, (int(cx), int(cy)), 6, col, -1)
                cv2.putText(vis, label, (int(cx) + 8, int(cy)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1)

        # FSM-effective target/trailer position (as actually used by
        # fsm_node, incl. persistence — taken from the robot's topics).
        with self._state_lock:
            s = dict(self.state)
        if s["target_found"]:
            cv2.circle(vis, (int(s["target_cx"]), int(s["target_cy"])), 14, (0, 255, 80), 2)
            if s["target_distance"] > 0:
                cv2.putText(vis, f'{s["target_distance"] * 100:.1f}cm',
                            (int(s["target_cx"]) + 16, int(s["target_cy"]) - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 80), 2)
            fx2, fy2, r = int(s["target_cx_f"]), int(s["target_cy_f"]), 10
            cv2.line(vis, (fx2 - r, fy2), (fx2 + r, fy2), (200, 80, 255), 2)
            cv2.line(vis, (fx2, fy2 - r), (fx2, fy2 + r), (200, 80, 255), 2)
        if s["trailer_found"]:
            tr_is_far = int(s["trailer_id"]) in all_trailer_far_ids
            tr_col = (255, 200, 0) if tr_is_far else (0, 100, 255)
            cv2.circle(vis, (int(s["trailer_cx"]), int(s["trailer_cy"])), 16, tr_col, 3)
            if s["trailer_distance"] > 0:
                cv2.putText(vis, f'{s["trailer_distance"] * 100:.1f}cm',
                            (int(s["trailer_cx"]) + 18, int(s["trailer_cy"]) - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, tr_col, 2)
            fx2, fy2, r = int(s["trailer_cx_f"]), int(s["trailer_cy_f"]), 10
            cv2.line(vis, (fx2 - r, fy2), (fx2 + r, fy2), (255, 160, 50), 2)
            cv2.line(vis, (fx2, fy2 - r), (fx2, fy2 + r), (255, 160, 50), 2)

        with self._state_lock:
            self.state["roi_source"] = roi_source

        self.frame = self._encode(vis, 75)

    @staticmethod
    def _encode(img, quality):
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return buf.tobytes() if ok else None

    # ── Flask routes ─────────────────────────────────────────────────
    def _register_routes(self):
        app = self.app

        @app.route("/")
        def index():
            try:
                from ament_index_python.packages import get_package_share_directory
                pkg_share = get_package_share_directory("aruco_laptop")
                tmpl_path = pkg_share + "/templates/dashboard.html"
            except Exception:
                import os
                tmpl_path = os.path.join(
                    os.path.dirname(__file__), "..", "templates", "dashboard.html")
            with open(tmpl_path) as f:
                return f.read(), 200, {"Content-Type": "text/html; charset=utf-8"}

        @app.route("/api/state")
        def api_state():
            with self._state_lock:
                s = dict(self.state)
            s["connected"] = (time.time() - s["last_update"]) < 5.0
            s["ros_domain_id"] = self.domain_id
            s["port"] = self.port
            s.pop("last_state_msg", None)
            return jsonify(s)

        @app.route("/api/config")
        def api_config():
            return jsonify({
                "targets":  self.camera_params.get("targets", {}),
                "trailers": self.camera_params.get("trailers", {}),
            })

        @app.route("/api/params", methods=["GET"])
        def api_params_get():
            return jsonify(self.camera_params)

        @app.route("/api/params", methods=["POST"])
        def api_params_set():
            updates = request.get_json()
            if not updates:
                return jsonify({"error": "no data"}), 400
            robot_updates = {k: v for k, v in updates.items() if k in ROBOT_PARAM_KEYS}
            if not robot_updates:
                return jsonify({"error": "no valid keys"}), 400
            msg = String(); msg.data = json.dumps(robot_updates)
            self.params_pub.publish(msg)
            self.camera_params.update(robot_updates)
            return jsonify({"ok": True})

        @app.route("/api/fsm_control", methods=["POST"])
        def api_fsm_control():
            cmd = request.get_json()
            if not cmd:
                return jsonify({"error": "no data"}), 400
            msg = String(); msg.data = json.dumps(cmd)
            self.control_pub.publish(msg)
            with self._state_lock:
                for k in ("use_distance_target", "use_distance_trailer",
                          "grasp_distance_m", "deploy_distance_m",
                          "approach_dist_gain", "approach_step_min", "approach_step_max"):
                    if k in cmd:
                        self.state[k] = cmd[k]
            return jsonify({"ok": True})

        @app.route("/stream/raw")
        def stream_raw():
            return Response(self._mjpeg_gen(),
                             mimetype="multipart/x-mixed-replace; boundary=frame")

    def _mjpeg_gen(self):
        last_sent = None
        while True:
            frame = self.frame
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
