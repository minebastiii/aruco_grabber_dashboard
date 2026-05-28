"""laptop.launch.py — launch on the laptop"""
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    pkg = get_package_share_directory("aruco_laptop")
    default_config = os.path.join(pkg, "config", "aruco_config.yaml")

    return LaunchDescription([
        DeclareLaunchArgument("config_path",    default_value=default_config),
        DeclareLaunchArgument("near_threshold", default_value="8000.0"),
        DeclareLaunchArgument("roi_scale",      default_value="4.0"),
        DeclareLaunchArgument("publish_debug",  default_value="true"),
        DeclareLaunchArgument("dashboard_port", default_value="8080"),

        # ── Detector node ────────────────────────────────────────────
        Node(
            package="aruco_laptop",
            executable="detector_node",
            name="detector_node",
            output="screen",
            parameters=[{
                "config_path":    LaunchConfiguration("config_path"),
                "near_threshold": LaunchConfiguration("near_threshold"),
                "roi_scale":      LaunchConfiguration("roi_scale"),
                "publish_debug":  LaunchConfiguration("publish_debug"),
            }],
        ),

        # ── Dashboard node ───────────────────────────────────────────
        Node(
            package="aruco_laptop",
            executable="dashboard_node",
            name="dashboard_node",
            output="screen",
            parameters=[{
                "port": LaunchConfiguration("dashboard_port"),
            }],
        ),
    ])
