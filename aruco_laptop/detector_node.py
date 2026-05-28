#!/usr/bin/env python3
"""
detector_node.py — runs on the LAPTOP

ArUco dict type is specified per config entry, e.g.:
  targets:
    cube_a:
      ids: [0, 1, 2]
      type: pick_up
      dict: 4x4_50
  trailers:
    trailer_main:
      ids: [10, 11]
      type: drop_off
      dict: 5x5_100

Multiple dict types are supported simultaneously.
Each unique dict is run once per frame; results are matched per-entry.
"""

import json
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Bool, String, Float32MultiArray
import numpy as np
import cv2
import yaml
import os
from ament_index_python.packages import get_package_share_directory

# ── ArUco dictionary map ─────────────────────────────────────────────
DICT_MAP = {
    "4x4_50":   cv2.aruco.DICT_4X4_50,   "4x4_100":  cv2.aruco.DICT_4X4_100,
    "4x4_250":  cv2.aruco.DICT_4X4_250,  "4x4_1000": cv2.aruco.DICT_4X4_1000,
    "5x5_50":   cv2.aruco.DICT_5X5_50,   "5x5_100":  cv2.aruco.DICT_5X5_100,
    "5x5_250":  cv2.aruco.DICT_5X5_250,  "5x5_1000": cv2.aruco.DICT_5X5_1000,
    "6x6_50":   cv2.aruco.DICT_6X6_50,   "6x6_100":  cv2.aruco.DICT_6X6_100,
    "6x6_250":  cv2.aruco.DICT_6X6_250,  "6x6_1000": cv2.aruco.DICT_6X6_1000,
    "7x7_50":   cv2.aruco.DICT_7X7_50,   "7x7_100":  cv2.aruco.DICT_7X7_100,
    "7x7_250":  cv2.aruco.DICT_7X7_250,  "7x7_1000": cv2.aruco.DICT_7X7_1000,
    "original": cv2.aruco.DICT_ARUCO_ORIGINAL,
}
DICT_NAMES = sorted(DICT_MAP.keys())

def build_detector(dict_name: str):
    """Build ArUco detector. Compatible with OpenCV 4.x, 4.5+ and 4.7+."""
    dict_id = DICT_MAP.get(dict_name, cv2.aruco.DICT_4X4_50)
    if hasattr(cv2.aruco, "getPredefinedDictionary"):
        d = cv2.aruco.getPredefinedDictionary(dict_id)
        p = cv2.aruco.DetectorParameters()
        if hasattr(cv2.aruco, "ArucoDetector"):
            return ("new", cv2.aruco.ArucoDetector(d, p))
        return ("mid", (d, p))
    d = cv2.aruco.Dictionary_get(dict_id)
    p = cv2.aruco.DetectorParameters_create()
    return ("old", (d, p))

def run_detector(det_tuple, gray):
    mode, det = det_tuple
    if mode == "new":
        corners, ids, _ = det.detectMarkers(gray)
    else:
        d, p = det
        corners, ids, _ = cv2.aruco.detectMarkers(gray, d, parameters=p)
    return corners, ids

# ── Config entry helpers ─────────────────────────────────────────────
def entry_ids(e):
    return e.get("ids", []) if isinstance(e, dict) else list(e)


def entry_dict_name(e):
    return e.get("dict", "4x4_50") if isinstance(e, dict) else "4x4_50"

def make_entry(ids, dict_name="4x4_50"):
    return {"ids": [int(i) for i in ids], "dict": str(dict_name)}

# ── Default tunable params ───────────────────────────────────────────
DEFAULT_PARAMS = {
    "roi_scale":      4.0,
    "min_cube_area":  200.0,
    "floor_h_low":    35,  "floor_h_high": 85,
    "floor_s_low":    60,  "floor_s_high": 255,
    "floor_v_low":    40,  "floor_v_high": 255,
    "white_v_thresh": 180, "white_s_thresh": 50,
    "dark_v_thresh":  100, "dark_fill_min":  0.08,
    "filter_alpha":   0.25,
}


class DetectorNode(Node):
    def __init__(self):
        super().__init__("detector_node")

        self.declare_parameter("config_path", "aruco_config.yaml")
        for k, v in DEFAULT_PARAMS.items():
            self.declare_parameter(k, v)

        self.p = {k: self.get_parameter(k).value for k in DEFAULT_PARAMS}

        # Detector cache: dict_name → detector_tuple
        self._det_cache: dict = {}

        config_path = self.get_parameter("config_path").value
        if not os.path.isabs(config_path):
            try:
                pkg_share = get_package_share_directory("aruco_laptop")
                config_path = os.path.join(pkg_share, "config", config_path)
            except Exception:
                config_path = os.path.join(
                    os.path.dirname(__file__), "..", "config", config_path)
        self._load_config(config_path)

        # EMA filter state
        self._filt = {"target": None, "trailer": None}

        # Subscribers
        self.create_subscription(CompressedImage, "camera/compressed", self._cb_image, 1)
        self.create_subscription(String, "detector/params", self._cb_params, 10)

        # Publishers
        self.pub = {}
        for kind in ("target", "trailer"):
            self.pub[f"{kind}_found"]     = self.create_publisher(Bool,              f"aruco/{kind}/found",          10)
            self.pub[f"{kind}_data"]      = self.create_publisher(Float32MultiArray, f"aruco/{kind}/data",           10)
            self.pub[f"{kind}_data_filt"] = self.create_publisher(Float32MultiArray, f"aruco/{kind}/data_filtered",  10)
            self.pub[f"{kind}_name"]      = self.create_publisher(String,            f"aruco/{kind}/name",           10)
            self.pub[f"{kind}_type"]      = self.create_publisher(String,            f"aruco/{kind}/type",           10)
        self.pub["roi_count"] = self.create_publisher(String, "aruco/roi_count", 10)

        self.dbg = {k: self.create_publisher(CompressedImage, f"aruco/debug/{k}", 1)
                    for k in ("raw", "hsv", "mask", "zoomed")}
        self.params_pub = self.create_publisher(String, "detector/params/current", 10)
        self.create_timer(2.0, self._pub_params)

        dicts_in_use = {entry_dict_name(e)
                        for d in (self.targets, self.trailers) for e in d.values()}
        self.get_logger().info(
            f"Detector ready. Dicts: {dicts_in_use}  "
            f"Targets: {list(self.targets.keys())}  Trailers: {list(self.trailers.keys())}"
        )

    # ── Detector cache ────────────────────────────────────────────────
    def _get_det(self, dict_name: str):
        if dict_name not in self._det_cache:
            self._det_cache[dict_name] = build_detector(dict_name)
        return self._det_cache[dict_name]

    # ── Config ───────────────────────────────────────────────────────
    def _load_config(self, path):
        self.targets = {}; self.trailers = {}
        try:
            with open(path) as f:
                cfg = yaml.safe_load(f)
            for name, val in cfg.get("targets", {}).items():
                if isinstance(val, dict):
                    self.targets[name] = make_entry(
                        val.get("ids", []), val.get("type", ""), val.get("dict", "4x4_50"))
                else:
                    self.targets[name] = make_entry(val)
            for name, val in cfg.get("trailers", {}).items():
                if isinstance(val, dict):
                    self.trailers[name] = make_entry(
                        val.get("ids", []), val.get("type", ""), val.get("dict", "4x4_50"))
                else:
                    self.trailers[name] = make_entry(val)
            self.get_logger().info(f"Config loaded from {path}")
        except Exception as e:
            self.get_logger().error(f"Config load failed: {e}")

    # ── Live param update ─────────────────────────────────────────────
    def _cb_params(self, msg: String):
        try:
            updates = json.loads(msg.data)
            for k, v in updates.items():
                if k == "targets":
                    self.targets = {
                        n: make_entry(
                            e.get("ids", []) if isinstance(e, dict) else e,
                            e.get("dict", "4x4_50") if isinstance(e, dict) else "4x4_50")
                        for n, e in v.items()}
                elif k == "trailers":
                    self.trailers = {
                        n: make_entry(
                            e.get("ids", []) if isinstance(e, dict) else e,
                            e.get("dict", "4x4_50") if isinstance(e, dict) else "4x4_50")
                        for n, e in v.items()}
                elif k in self.p:
                    self.p[k] = type(self.p[k])(v)
        except Exception as e:
            self.get_logger().error(f"Param update failed: {e}")

    def _pub_params(self):
        msg = String()
        payload = dict(self.p)
        payload["targets"]  = self.targets
        payload["trailers"] = self.trailers
        payload["dict_names"] = DICT_NAMES
        msg.data = json.dumps(payload)
        self.params_pub.publish(msg)

    # ── EMA filter ───────────────────────────────────────────────────
    def _update_filter(self, kind, cx, cy, area, found):
        alpha = self.p["filter_alpha"]
        if not found:
            return self._filt[kind]
        if self._filt[kind] is None:
            self._filt[kind] = [cx, cy, area]
        else:
            f = self._filt[kind]
            f[0] = alpha*cx   + (1-alpha)*f[0]
            f[1] = alpha*cy   + (1-alpha)*f[1]
            f[2] = alpha*area + (1-alpha)*f[2]
        return self._filt[kind]

    # ── Detect on image with all needed dicts ─────────────────────────
    def _detect_all(self, gray) -> dict:
        """
        Run detection for every unique dict used in config.
        Returns: {dict_name: [(cx, cy, area, id), ...]}
        """
        needed = {entry_dict_name(e)
                  for d in (self.targets, self.trailers) for e in d.values()}
        result = {}
        for dn in needed:
            corners, ids = run_detector(self._get_det(dn), gray)
            dets = []
            if ids is not None:
                for corner, mid in zip(corners, ids.flatten()):
                    pts  = corner[0]
                    cx   = float(np.mean(pts[:, 0]))
                    cy   = float(np.mean(pts[:, 1]))
                    side = float(np.linalg.norm(pts[0] - pts[1]))
                    dets.append((cx, cy, side*side, int(mid)))
            result[dn] = dets
        return result

    # ── Main callback ─────────────────────────────────────────────────
    def _cb_image(self, msg: CompressedImage):
        buf   = np.frombuffer(msg.data, dtype=np.uint8)
        frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if frame is None:
            return

        now = self.get_clock().now().to_msg()
        p   = self.p

        # HSV + masks
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        _, s_ch, v_ch = cv2.split(hsv)
        floor_mask = cv2.inRange(hsv,
            np.array([p["floor_h_low"],  p["floor_s_low"],  p["floor_v_low"]]),
            np.array([p["floor_h_high"], p["floor_s_high"], p["floor_v_high"]]))
        white_mask = cv2.bitwise_and(
            cv2.threshold(v_ch, p["white_v_thresh"], 255, cv2.THRESH_BINARY)[1],
            cv2.threshold(s_ch, p["white_s_thresh"], 255, cv2.THRESH_BINARY_INV)[1])
        dark_mask  = cv2.threshold(v_ch, p["dark_v_thresh"], 255, cv2.THRESH_BINARY_INV)[1]
        cube_mask  = cv2.bitwise_and(dark_mask,
                        cv2.bitwise_not(cv2.bitwise_or(floor_mask, white_mask)))
        k3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        k7 = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        cube_mask = cv2.morphologyEx(cv2.morphologyEx(cube_mask, cv2.MORPH_OPEN, k3),
                                     cv2.MORPH_CLOSE, k7)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Full-frame detection (all dicts)
        snap_by_dict = self._detect_all(gray)

        # Collect all tracked detections (any dict)
        tracked_on_full = []
        for entry in list(self.targets.values()) + list(self.trailers.values()):
            dn = entry_dict_name(entry)
            for d in snap_by_dict.get(dn, []):
                if d[3] in entry_ids(entry):
                    tracked_on_full.append(d)

        # All unique ids found (for display)
        all_target_ids  = {i for e in self.targets.values()  for i in entry_ids(e)}
        all_trailer_ids = {i for e in self.trailers.values() for i in entry_ids(e)}

        # ROI + zoom
        zoomed_img = None

        if tracked_on_full:
            det_by_dict = snap_by_dict
            roi         = self._roi_from_detections(tracked_on_full, all_target_ids,
                                                     all_trailer_ids, frame.shape)
            roi_source  = "marker"
            if roi is not None:
                rx, ry, rw, rh = roi
                scale      = p["roi_scale"]
                zoomed_col = cv2.resize(frame[ry:ry+rh, rx:rx+rw],
                                        (int(rw*scale), int(rh*scale)), interpolation=cv2.INTER_CUBIC)
                for cx, cy, area, mid in tracked_on_full:
                    cz, cyz = int((cx-rx)*scale), int((cy-ry)*scale)
                    cv2.circle(zoomed_col, (cz, cyz), 8, (0,255,100), -1)
                    cv2.putText(zoomed_col, f"ID:{mid}", (cz+10, cyz),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,100), 2)
                zoomed_img = zoomed_col
        else:
            roi        = self._find_roi(cube_mask, v_ch, p)
            roi_source = "mask" if roi is not None else "none"
            det_by_dict = {}

            if roi is not None:
                rx, ry, rw, rh = roi
                scale       = p["roi_scale"]
                roi_gray    = gray[ry:ry+rh, rx:rx+rw]
                zoomed_gray = cv2.resize(roi_gray, (int(rw*scale), int(rh*scale)),
                                         interpolation=cv2.INTER_CUBIC)
                zoomed_col  = cv2.resize(frame[ry:ry+rh, rx:rx+rw],
                                         (int(rw*scale), int(rh*scale)), interpolation=cv2.INTER_CUBIC)

                needed = {entry_dict_name(e)
                          for d in (self.targets, self.trailers) for e in d.values()}
                for dn in needed:
                    z_corners, z_ids = run_detector(self._get_det(dn), zoomed_gray)
                    dets = []
                    if z_ids is not None:
                        for corner, mid in zip(z_corners, z_ids.flatten()):
                            pts  = corner[0]
                            cx_z = float(np.mean(pts[:, 0]))
                            cy_z = float(np.mean(pts[:, 1]))
                            side = float(np.linalg.norm(pts[0]-pts[1])) / scale
                            dets.append((rx+cx_z/scale, ry+cy_z/scale, side*side, int(mid)))
                            cv2.circle(zoomed_col, (int(cx_z), int(cy_z)), 8, (0,255,100), -1)
                            cv2.putText(zoomed_col, f"ID:{int(mid)}", (int(cx_z)+10, int(cy_z)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,100), 2)
                    det_by_dict[dn] = dets
                zoomed_img = zoomed_col

            # Fallback: use snap results if zoom found nothing
            if not any(det_by_dict.values()):
                det_by_dict = snap_by_dict

        if zoomed_img is None:
            zoomed_img = np.zeros((120, 160, 3), dtype=np.uint8)
            cv2.putText(zoomed_img, "No ROI", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80,80,80), 2)

        # Match using per-entry dict
        t_found,  t_data,  t_name  = self._match(det_by_dict, self.targets,  roi)
        tr_found, tr_data, tr_name = self._match(det_by_dict, self.trailers, roi)

        # EMA filter
        t_filt  = self._update_filter("target",  t_data[0],  t_data[1],  t_data[2],  t_found)
        tr_filt = self._update_filter("trailer", tr_data[0], tr_data[1], tr_data[2], tr_found)

        # Debug image
        raw_vis = frame.copy()
        if roi is not None:
            rx, ry, rw, rh = roi
            c = (0,220,80) if roi_source=="marker" else (100,200,255)
            cv2.rectangle(raw_vis, (rx,ry), (rx+rw,ry+rh), c, 2)
            cv2.putText(raw_vis, roi_source, (rx+4,ry+18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 2)

        all_dets_flat = [(cx,cy,a,mid,dn)
                         for dn,dets in det_by_dict.items() for cx,cy,a,mid in dets]
        for cx,cy,a,mid,dn in all_dets_flat:
            col = (0,220,80) if mid in all_target_ids else \
                  (220,80,0) if mid in all_trailer_ids else (220,220,0)
            cv2.circle(raw_vis, (int(cx),int(cy)), 6, col, -1)
            cv2.putText(raw_vis, f"ID:{mid}", (int(cx)+8,int(cy)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1)
        if t_found:
            cv2.circle(raw_vis, (int(t_data[0]),int(t_data[1])), 12, (0,255,80), 2)
        if tr_found:
            cv2.circle(raw_vis, (int(tr_data[0]),int(tr_data[1])), 12, (255,100,0), 2)
        if t_filt is not None:
            fx, fy, r = int(t_filt[0]), int(t_filt[1]), 10
            cv2.line(raw_vis,(fx-r,fy),(fx+r,fy),(200,80,255),2)
            cv2.line(raw_vis,(fx,fy-r),(fx,fy+r),(200,80,255),2)
            cv2.circle(raw_vis,(fx,fy),r+4,(200,80,255),1)
            cv2.putText(raw_vis,"filt",(fx+r+4,fy-4),cv2.FONT_HERSHEY_SIMPLEX,0.4,(200,80,255),1)
        if tr_filt is not None:
            fx, fy, r = int(tr_filt[0]), int(tr_filt[1]), 10
            cv2.line(raw_vis,(fx-r,fy),(fx+r,fy),(255,160,50),2)
            cv2.line(raw_vis,(fx,fy-r),(fx,fy+r),(255,160,50),2)
            cv2.circle(raw_vis,(fx,fy),r+4,(255,160,50),1)

        mask_vis = self._make_mask_vis(frame, cube_mask, floor_mask, white_mask, roi)

        self._publish_result("target",  t_found,  t_data,  t_name,  t_filt)
        self._publish_result("trailer", tr_found, tr_data, tr_name, tr_filt)
        rc = String(); rc.data = roi_source; self.pub["roi_count"].publish(rc)
        self._pub_img(raw_vis,    "raw",    now, 70)
        self._pub_img(hsv,        "hsv",    now, 70)
        self._pub_img(mask_vis,   "mask",   now, 70)
        self._pub_img(zoomed_img, "zoomed", now, 80)

    # ── ROI: lowest target ID, then lowest trailer ID ─────────────────
    def _roi_from_detections(self, tracked_dets, target_ids, trailer_ids, frame_shape):
        for id_set in (target_ids, trailer_ids):
            cands = [d for d in tracked_dets if d[3] in id_set]
            if not cands: continue
            cx, cy, area, mid = min(cands, key=lambda d: d[3])
            side = area**0.5; H, W = frame_shape[:2]
            pad  = max(12, int(side*0.5))
            x = max(0, int(cx-side)-pad); y = max(0, int(cy-side)-pad)
            w = min(W-x, int(side*2)+2*pad); h = min(H-y, int(side*2)+2*pad)
            return (x,y,w,h) if w>=4 and h>=4 else None
        return None

    # ── ROI from mask ─────────────────────────────────────────────────
    def _find_roi(self, cube_mask, v_channel, p):
        contours, _ = cv2.findContours(cube_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best = None; best_score = 0.0
        for c in contours:
            area = cv2.contourArea(c)
            if area < p["min_cube_area"]: continue
            x, y, w, h = cv2.boundingRect(c)
            if w/max(h,1)<0.15 or w/max(h,1)>6.0: continue
            df = int(np.sum(v_channel[y:y+h,x:x+w] < p["dark_v_thresh"])) / max(w*h,1)
            if df < p["dark_fill_min"]: continue
            s = area*df
            if s > best_score: best_score=s; best=(x,y,w,h)
        if best is None: return None
        x,y,w,h = best; H,W = cube_mask.shape[:2]
        pad = max(10, int(max(w,h)*0.15))
        return (max(0,x-pad), max(0,y-pad),
                min(W-max(0,x-pad),w+2*pad), min(H-max(0,y-pad),h+2*pad))

    # ── Mask vis ─────────────────────────────────────────────────────
    def _make_mask_vis(self, frame, cube_mask, floor_mask, white_mask, roi):
        vis = (frame*0.35).astype(np.uint8)
        vis[floor_mask>0] = np.array([0,50,0],   dtype=np.uint8)
        vis[white_mask>0] = np.array([40,40,40],  dtype=np.uint8)
        vis[cube_mask >0] = np.array([180,220,0], dtype=np.uint8)
        if roi is not None:
            rx,ry,rw,rh = roi
            cv2.rectangle(vis,(rx,ry),(rx+rw,ry+rh),(0,230,255),2)
            fill = int(np.sum(cube_mask[ry:ry+rh,rx:rx+rw]>0)/max(rw*rh,1)*100)
            cv2.putText(vis,f"fill:{fill}%",(rx+4,ry+18),cv2.FONT_HERSHEY_SIMPLEX,0.45,(0,230,255),1)
        return vis

    # ── Match — uses per-entry dict ───────────────────────────────────
    def _match(self, det_by_dict: dict, group_dict: dict, roi):
        for name, entry in group_dict.items():
            dn   = entry_dict_name(entry)
            dets = det_by_dict.get(dn, [])
            for cx, cy, area, mid in dets:
                if mid in entry_ids(entry):
                    return True, [cx,cy,area,float(mid)], name
        if roi is not None:
            rx,ry,rw,rh = roi
            return False, [float(rx+rw/2),float(ry+rh/2),float(rw*rh),-1.0], ""
        return False, [0.0,0.0,0.0,-1.0], ""

    # ── Publish ───────────────────────────────────────────────────────
    def _publish_result(self, kind, found, data, name, filt):
        b = Bool(); b.data=found; self.pub[f"{kind}_found"].publish(b)
        fa = Float32MultiArray(); fa.data=[float(v) for v in data]; self.pub[f"{kind}_data"].publish(fa)
        if filt is not None:
            ff = Float32MultiArray()
            ff.data=[float(filt[0]),float(filt[1]),float(filt[2]),float(data[3])]
            self.pub[f"{kind}_data_filt"].publish(ff)
        s=String(); s.data=name; self.pub[f"{kind}_name"].publish(s)

    def _pub_img(self, img, key, stamp, quality=70):
        ok,buf = cv2.imencode(".jpg",img,[cv2.IMWRITE_JPEG_QUALITY,quality])
        if not ok: return
        msg=CompressedImage(); msg.header.stamp=stamp; msg.format="jpeg"; msg.data=buf.tobytes()
        self.dbg[key].publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = DetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node(); rclpy.shutdown()

if __name__ == "__main__":
    main()
