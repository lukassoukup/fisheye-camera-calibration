#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author Lukáš Soukup
"""
import numpy as np
import os

import cv2
assert cv2.__version__[0] == '3'  # The fisheye module requires opencv version >= 3.0.0


class FisheyeCameraCalibration(object):

    def __init__(self):

        self.parameters = None
        self.calibration_images = list()
        self._img_shape = None

    def calibrate(self, checkerboard: (int, int)=(6, 8)):

        assert len(self.calibration_images) == 0, "Calibration images must be loaded first."

        subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)

        objp = np.zeros((1, checkerboard[0] * checkerboard[1], 3), np.float32)
        objp[0, :, :2] = np.mgrid[0:checkerboard[0], 0:checkerboard[1]].T.reshape(-1, 2)
        objpoints = []  # 3d point in real world space
        imgpoints = []  # 2d points in image plane.

        for img in self.calibration_images:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray, checkerboard,
                                                     cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
            # If found, add object points, image points (after refining them)
            if ret is True:
                objpoints.append(objp)
                cv2.cornerSubPix(gray, corners, (3, 3), (-1, -1), subpix_criteria)

    def draw_indexes_of_corners(self, img: np.ndarray, corners: list, show_image: bool = False, save_path: str = None):
        # print corner indexes into image
        font = cv2.FONT_HERSHEY_SIMPLEX
        for idx, cor in enumerate(corners):
            cv2.putText(img, str(idx), tuple(cor[0]), font, 0.3, (0,0,255))
        if save_path:
            cv2.imwrite(save_path, img)
        if show_image:
            cv2.imshow("corners", img)

    def save_parameters(self, path: str, filename: str="P.npy"):
        pass

    def set_parameters(self, parameters: np.ndarray):
        pass

    def distort(self, image: np.ndarray):
        pass

    def undistort(self, image: np.ndarray):
        pass

    def load_calibration_images(self, image_dir: str):

        assert os.path.exists(image_dir), "Image dir does not exist."

        for idx, image_file in enumerate(os.listdir(image_dir)):

            img = cv2.imread(os.path.join(image_dir, image_file))

            if self._img_shape is None:
                self._img_shape = img.shape[:2]
            else:
                assert self._img_shape == img.shape[:2], "All images must share the same size."

            self.calibration_images.append(img)
