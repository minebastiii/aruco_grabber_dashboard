"""laptop.launch.py — launch on the LAPTOP/server

Starts dashboard_node: observation-only (live video, robot/FSM state)
plus settings forwarding to camera/params and fsm/control on the robot.

"port" is a launch argument so multiple dashboards (one per robot) can
run side by side, e.g.:
    ros2 launch aruco_laptop laptop.launch.py port:=8081
"""
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            "port", default_value="8080",
            description="HTTP port for this dashboard instance — set a "
                         "different value per robot to run several at once.",
        ),

        Node(
            package="aruco_laptop",
            executable="dashboard_node",
            name="dashboard_node",
            output="screen",
            parameters=[{
                "port": LaunchConfiguration("port"),
            }],
        ),
    ])
