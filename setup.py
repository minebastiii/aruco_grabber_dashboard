from setuptools import setup
import os
from glob import glob

package_name = "aruco_laptop"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages",
            ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"),
            glob("launch/*.py")),
        (os.path.join("share", package_name, "config"),
            glob("config/*.yaml")),
        (os.path.join("share", package_name, "templates"),
            glob("templates/*.html")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    entry_points={
        "console_scripts": [
            "detector_node  = aruco_laptop.detector_node:main",
            "dashboard_node = aruco_laptop.dashboard_node:main",
        ],
    },
)
