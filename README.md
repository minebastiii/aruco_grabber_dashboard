# XGO-mini2 Dashboard — Laptop Dashboard

Web dashboard for monitoring and live-tuning the ArUco pick-and-place robot from a laptop. It runs as a ROS2 node (`dashboard_node`) that bridges ROS2 topics to a small Flask web app, so you can watch the robot's state, live camera feed and detections in a browser, and push settings changes back to the robot.

This dashboard is a **pure viewer/controller** — all ArUco marker detection happens on the robot itself (`camera_node`, see the robot repo). The dashboard re-runs detection only locally, purely to draw the marker overlay on the live image; it does not feed anything back into the FSM from that. Everything the FSM actually acts on comes straight from the robot's topics.

## What it does

- Subscribes to the robot's detection and state topics (`aruco/target/*`, `aruco/trailer/*`, `robot/state`, `fsm/control_mode`, `robot/battery`) and exposes them as JSON via `GET /api/state`.
- Serves `GET /api/config` — the currently loaded `targets`/`trailers` marker config (same YAML format as the robot side).
- Lets you push live config updates via `POST /api/params` and FSM control changes (distance thresholds, distance-vs-area mode, approach tuning) via `POST /api/fsm_control`.
- Serves a single MJPEG live stream at `/stream/raw` — the raw camera feed with the marker overlay (detected IDs, ROI box, tracked target/trailer markers) drawn on top.
- Serves the dashboard UI (`templates/dashboard.html`) at `/`.

## UI overview

The dashboard is a single-page, high-contrast dark UI (`dashboard.html`) built for a large screen in a bright room: one large live camera panel on the left, and a sidebar with two tabs on the right.

**Header** — connection pill, `ROS_DOMAIN_ID` (read from the environment the node was launched in), fps, and battery indicator. Useful when several dashboards are open side by side, to see at a glance which one is talking to which domain.

**Video panel** — a single live feed (`/stream/raw`): the camera image with the marker overlay. The old HSV / colour-mask / zoomed-crop debug views are gone — they belonged to an older colour-based detection approach the robot no longer uses, so keeping them would only have added noise.

**Status tab (sidebar)** — everything you need at a glance, nothing else:
- Current FSM state as a colored badge (`SEARCH`, `ALIGN`, `APPROACH`, `GRASP`, `VERIFY`, `DEPLOY`, `DONE`, ...).
- An **alignment indicator** — a left/right arrow + track showing how far the currently-tracked marker's `cx` is from center, mirroring the FSM's own `cx_threshold`-based ALIGN logic (hardcoded in the UI to image width 1920 / threshold 0.15 — must be kept in sync manually if those change on the robot).
- Target/trailer cards — found/not-found indicator, name, `cx`/`cy`/`area`/`id`, filtered area bar, and a distance bar (color-coded against the current grasp/deploy threshold). The trailer card also shows a "far" badge when the currently-matched marker ID belongs to that trailer's `far_ids`.
- Connection state (header pill, always visible).

Everything that isn't robot state or detection — i.e. anything settable — now lives in **Settings**.

**Settings tab (sidebar)** — only settings the robot actually still uses:
- **Approach Behavior** — per marker (target/trailer), toggle distance-based vs. area-based triggering, set `grasp_distance_m`/`deploy_distance_m`, and tune the distance-approach controller (`approach_dist_gain`, `approach_step_min`, `approach_step_max`). "Apply" sends all of this via `POST /api/fsm_control`.
- **Detection Filtering** — `filter_alpha` (EMA smoothing) and `persist_ttl` (how long a lost marker's last position is held) — real `camera_node` parameters.
- **Marker Config — Targets / Trailers** — an editor for the `targets`/`trailers` marker config (name, IDs, dict, marker size, and for trailers the optional far-marker fields). Sent together with the sliders above via `POST /api/params` when you click "Apply All".

## Running multiple dashboards (one per robot)

`port` is a launch argument, so you can run one dashboard instance per robot on the same laptop, each on its own port:

```bash
ros2 launch aruco_laptop laptop.launch.py port:=8081   # robot 1
ros2 launch aruco_laptop laptop.launch.py port:=8082   # robot 2
```

Each instance is otherwise identical — just point your browser at the matching port. `ROS_DOMAIN_ID` is not a launch argument (it has to be set in the environment *before* `ros2 launch` runs, since DDS binds to it at process start), but the header shows whichever domain the running instance is actually on, so you can tell instances apart even if you forget which port maps to which robot.

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
    └── dashboard.html          # the actual web UI
```

Node executable (as referenced by the launch file): `dashboard_node`.

## Running it

```bash
ROS_DOMAIN_ID=<id> ros2 launch aruco_laptop laptop.launch.py port:=8080
```

Then open `http://<laptop-ip>:8080/` in a browser (default port `8080`, see the `port` launch arg above).

## Requirements

- ROS2, running on the same network/DDS domain as the robot (matching `ROS_DOMAIN_ID`)
- Python: `flask`, `pyyaml`, `opencv-python` (for the local marker overlay detection)
