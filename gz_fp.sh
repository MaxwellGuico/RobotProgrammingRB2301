#!/bin/bash
# VS Code (Snap) terminals leak Snap library paths that crash the Gazebo GUI.
unset GTK_PATH GTK_EXE_PREFIX GIO_MODULE_DIR GDK_PIXBUF_MODULE_FILE GDK_PIXBUF_MODULEDIR GTK_IM_MODULE_FILE LOCPATH GSETTINGS_SCHEMA_DIR
# Randomise the maze each launch? 0 = off (reuse the current world file), 1 = on
export RB2301_RANDOMIZE=${RB2301_RANDOMIZE:-0}
source install/setup.bash
ros2 launch rb2301_gz fp_gazebo.launch.py x:=0.0 y:=0.0
# ros2 launch rb2301_gz fp_gazebo.launch.py x:=3.8 y:=-2.6 # Start past the gate

pkill -9 ruby # kills any lingering Gazebo