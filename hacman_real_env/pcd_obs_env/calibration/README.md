## Procedure of Camera Calibration

### Prerequisite

Make sure that `hacman_real_env` is installed. 

```
# Under the hacman_real_robot project root
pip install -e .
``````

----

**Note about compatibility with different setups:**

1. We mostly just need the robot_controller class in the `hacman_real_env` package, which is a wrapper around the deoxys controller. However, note that this controller can be swapped with other controllers as long as the controller can move the eef to different poses and read the eef poses.

2. In our setup, we use Azure kinect to collect IR image and aruco markers as the calibration target. Both can be modified for different hardware setups.

3. This script is created for eye-to-hand calibration, please adjust the pose sampling accordingly for eye-in-hand calibration (see below).

----

### Run data collection

**What you need to complete:**

1. **For each camera**, manually adjust the robot arm such that the marker mounted on the gripper appears at the center of the camera. Record the joint positions to be the reset joint positions by running `hacman_real_env/robot_controller/robot_controller.py`.

2. Replace the default reset joint positions in the collect data [script](https://github.com/JiangBowen0008/hacman_real_robot/blob/7ed085abc75111fb5d683ba58a366e83ffffcf47/hacman_real_env/pcd_obs_env/calibration/collect_data.py#L77). We used 4 cameras in our setup, you can adjust the `cam_id` and `initial_joint_positions` based on the number of cameras in your setup.

3. **If you are using eye-in-hand calibration**, the eef pose sampling should be changed to such that the mounted camera can always see the fixed checkerboard/marker on the table. See [this part of the code](https://github.com/JiangBowen0008/hacman_real_robot/blob/7ed085abc75111fb5d683ba58a366e83ffffcf47/hacman_real_env/pcd_obs_env/calibration/collect_data.py#L37).


**Now run the data collection script:**

```
python hacman_real_env/pcd_obs_env/calibration/collect_data.py
```

This script moves the eef to a random pose near the initial joint positions, records down the eef pose in the robot frame and the detected aruco marker pose (**left top corner**) in the camera frame, and saves the data in `calibration/data/cam[selected_cam_id]_data.pkl`.


----

### Solve for calibration

**What you need to complete:**

1. Our script assumes the eef-to-marker transform is known. Adjust [this transform](https://github.com/JiangBowen0008/hacman_real_robot/blob/7ed085abc75111fb5d683ba58a366e83ffffcf47/hacman_real_env/pcd_obs_env/calibration/solve_calibration.py#L7) in the `calibration/solve_calibration.py` script based on your setup.

2. **If you are using eye-in-hand calibration**, the eef-to-marker transform can be set to uniform.


**Run the following script to solve for calibration using the data previously collected.**

```
python hacman_real_env/pcd_obs_env/calibration/solve_calibration.py
```

This script solves the correspondence between tag in robot base (world frame) and tag in camera frame. The result is the extrinsic matrix we need that transform pcd in the camera

----

### Multi-camera Alignment

If you have multiple cameras and you need to align them together, see `hacman_real_env/pcd_obs_env/debug_pcd_processing.ipynb` for detailed instructions.


---

### Notes from Lifan:

If using Ubuntu 20.04, follow this for Kinect SDK installation: [https://atlane.de/install-azure-kinect-sdk-1-4-on-ubuntu-22-04/](https://atlane.de/install-azure-kinect-sdk-1-4-on-ubuntu-22-04/)

If there are issues like the one below

```
[2019-11-08 11:45:59.193] [error] [t=6263] /__w/1/s/extern/Azure-Kinect-Sensor-SDK/src/usbcommand/usbcommand.c (30): TraceLibUsbError(). /__w/1/s/extern/Azure-Kinect-Sensor-SDK/src/usbcommand/usbcommand.c (447): libusb_claim_interface(usbcmd->libusb, usbcmd->interface) returned LIBUSB_ERROR_BUSY in usb_cmd_create 
[2019-11-08 11:45:59.193] [error] [t=6263] /__w/1/s/extern/Azure-Kinect-Sensor-SDK/src/usbcommand/usbcommand.c (30): TraceLibUsbError(). /__w/1/s/extern/Azure-Kinect-Sensor-SDK/src/usbcommand/usbcommand.c (489): libusb_release_interface(usbcmd->libusb, usbcmd->interface) returned LIBUSB_ERROR_NOT_FOUND in usb_cmd_destroy 
```

That means, by default, Linux limits image capturing to a Max_value, in my desktop it was limited to 16 MB.

To extend the USBFS limit, I manually moddified the grub
( /etc/default/grub ) chaning ( GRUB_CMDLINE_LINUX_DEFAULT="quiet splash" ) to ==> ( GRUB_CMDLINE_LINUX_DEFAULT="quiet splash usbcore.usbfs_memory_mb=1000" ),

Update the grup ( sudo update-grub ) and restart your PC ( sudo reboot ).

check if buffer size has successfully changed ( cat /sys/module/usbcore/parameters/usbfs_memory_mb )

source: [https://github.com/microsoft/Azure_Kinect_ROS_Driver/issues/97](https://github.com/microsoft/Azure_Kinect_ROS_Driver/issues/97)



