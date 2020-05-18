# fisheye-camera-calibration #
Fisheye camera calibration class in python.

Requirements:
-------------
- numpy
- opencv-python >= 3.0.0 (tested with 3.4.5.20)

Quick Start:
------------

FisheyeCameraCalibation class provides basic functionality for work with fisheye camera and 
fisheye images.
The class can be initialized with real camera parameters or with default values.

### Useful functions: ###

- **load_calibration_images(image_dir)** - Loads calibration images. Necessary to do before calibrating.
- **calibrate(calibratoin_flags, checkerboard)** - Perform calibration of the camera from calibration images.
The method save the parameters as a members *K* and *D* of the class.
- **undistort(image, result_crop, balance)** - Undistort the fisheye image. Result crop and balance specify whether the
undistorted image will be with invalid boundaries or not.
- **draw_indexes_of_corners(img, corners, show_image, save_path)** - Draw the indexes of detected corners into the
image. Helps with debugging.

For inspiration see *main.py*.