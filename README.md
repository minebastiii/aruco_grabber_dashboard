# ArUco Robot — Laptop Dashboard

Web dashboard for monitoring and live-tuning the ArUco pick-and-place robot from a laptop. It runs as a ROS2 node (`dashboard_node`) that bridges ROS2 topics to a small Flask web app, so you can watch the robot's state, camera feed and detections in a browser, and push parameter/config changes back to the robot.

This dashboard is a **pure viewer/controller** — all ArUco marker detection happens on the robot itself (`camera_node`, see the robot repo). The dashboard does not re-detect markers; it only displays what the robot reports and forwards config/tuning changes.

## What it does

- Subscribes to the robot's detection and state topics (`aruco/target/*`, `aruco/trailer/*`, `robot/state`, `fsm/control_mode`, `robot/battery`) and exposes them as JSON via `GET /api/state`.
- Serves `GET /api/config` — the currently loaded `targets`/`trailers` marker config (same YAML format as the robot side).
- Lets you push live parameter/tuning updates via `POST /api/params` and FSM control changes (distance thresholds, distance-vs-area mode, approach tuning) via `POST /api/fsm_control`.
- Serves MJPEG debug image streams at `/stream/raw`, `/stream/hsv`, `/stream/mask`, `/stream/zoomed` (see **Debug streams** below — these require an extra companion node).
- Serves the dashboard UI (`templates/dashboard.html`) at `/`.

## UI overview

The dashboard is a single-page dark-themed web UI (`dashboard.html`) with a 2×2 video grid on the left and a sidebar with two tabs on the right.

**Video grid** — four MJPEG panels: `HSV`, `Mask`, `Camera + ROI`, `Zoomed ROI`. All four come from the debug streams described above and require the companion `detector_node` to be running (see **Compatibility notes**).

**Status tab (sidebar):**
- Connection pill + fps badge, battery indicator (with a low-battery banner below 20%).
- Current FSM state as a colored badge (`SEARCH`, `ALIGN`, `APPROACH`, `GRASP`, `VERIFY`, `DEPLOY`, `DONE`, ...).
- An **alignment indicator** — a left/right arrow + track showing how far the currently-tracked marker's `cx` is from center, mirroring the FSM's own `cx_threshold`-based ALIGN logic (hardcoded in the UI to image width 1920 / threshold 0.15 — must be kept in sync manually if those change on the robot).
- **Approach mode control** — per marker (target/trailer), toggle distance-based vs. area-based triggering, set `grasp_distance_m`/`deploy_distance_m`, and tune the distance-approach controller (`approach_dist_gain`, `approach_step_min`, `approach_step_max`). "Apply" sends all of this via `POST /api/fsm_control`.
- Target/trailer cards — found/not-found indicator, name, `cx`/`cy`/`area`/`id`, filtered area bar, and a distance bar (color-coded against the current grasp/deploy threshold). The trailer card also shows a "far" badge when the currently-matched marker ID belongs to that trailer's `far_ids`.
- A system panel showing ROI source and last-update age (both legacy/inert — see **Compatibility notes**).

**Tune tab (sidebar):** a set of sliders and ArUco config editors, sent together via `POST /api/params` when you click "Apply All":
- **EMA Filter** (`filter_alpha`) and **Persistence** (`persist_ttl`) — these *do* correspond to real `camera_node` parameters.
- **ROI / Zoom, Green Floor Exclusion, White Wall Exclusion, Dark Cube Detection** — these are sliders for an older **color/HSV-based cube-detection algorithm** (floor/wall exclusion masks, dark-cube threshold, ROI zoom scale). The current ArUco-based `camera_node` has no such parameters and will silently ignore them.
- **ArUco Config — Targets / Trailers** — an editor for the `targets`/`trailers` marker config (name, IDs, dict, marker size, and for trailers the optional far-marker fields). This *does* map onto what `camera_node` expects.

## Package layout

```
aruco_laptop/
├── aruco_laptop/
│   └── dashboard_node.py
├── launch/
│   └── laptop.launch.py
├── config/
│   └── aruco_config.yaml       # same format as the robot's config
└── templates/
    └── dashboard.html          # the actual web UI — not included here
```

Node executable (as referenced by the launch file): `dashboard_node`.

## Running it

```bash
ros2 launch aruco_laptop laptop.launch.py
```

Then open `http://<laptop-ip>:8080/` in a browser (default port `8080`, see `dashboard_port` launch arg).

See `CHEATSHEET.md` for launch arguments, the HTTP API, and known compatibility caveats.

## Requirements

- ROS2, running on the same network/DDS domain as the robot
- Python: `flask`, `pyyaml`
