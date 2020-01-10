#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author Lukáš Soukup
"""
import numpy as np
import logging
import os
from typing import Union

import cv2
assert cv2.__version__[0] == '3'  # The fisheye module requires opencv version >= 3.0.0


class FisheyeCameraCalibration(object):

    def __init__(self, K=np.zeros((3, 3)), D=np.zeros((4, 1)), verbose=False):

        if verbose:
            logging.basicConfig(level=logging.DEBUG)
        else:
            logging.basicConfig(level=logging.WARNING)

        self.parameters = None
        self.calibration_images = list()
        self._img_shape = None
        self.imgpoints = list()
        self.K = K
        self.D = D

        self.objpoints = []  # 3d point in real world space
        self.imgpoints = []  # 2d points in image plane.

        logging.debug("Fisheye camera calibration class initialized.")

    def detect_corners(self, checkerboard: (int, int)=(6, 8)):

        logging.debug("Corner detection in calibration images started.")
        assert len(self.calibration_images) == 0, "Calibration images must be loaded first."

        subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)

        objp = np.zeros((1, checkerboard[0] * checkerboard[1], 3), np.float32)
        objp[0, :, :2] = np.mgrid[0:checkerboard[0], 0:checkerboard[1]].T.reshape(-1, 2)

        for img in self.calibration_images:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray, checkerboard,
                        cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
            # If found, add object points, image points (after refining them)
            if ret is True:
                self.objpoints.append(objp)
                cv2.cornerSubPix(gray, corners, (3, 3), (-1, -1), subpix_criteria)

                self.imgpoints.append(corners)

        logging.debug("Found " + str(len(self.objpoints)) + " valid images for calibration")
        logging.debug("Dimension of calibration images is: %s" % self._img_shape[::-1])

        return self.imgpoints

    def calibrate(self, calibration_flags=cv2.fisheye.CALIB_CHECK_COND + cv2.fisheye.CALIB_FIX_SKEW,
                  checkerboard: (int, int) = (6, 8)):

        logging.debug("Calibration started.")

        self.detect_corners(checkerboard)

        N_OK = len(self.objpoints)

        rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for _ in range(N_OK)]
        tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for _ in range(N_OK)]

        retval, self.K, self.D, rvecs, tvecs = cv2.fisheye.calibrate(
                                    self.objpoints,
                                    self.imgpoints,
                                    self._img_shape[::-1],
                                    self.K,
                                    self.D,
                                    rvecs,
                                    tvecs,
                                    calibration_flags,
                                    (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
            )

        logging.debug("Calibration successfully finished.")
        logging.info("Found calibration matrix K=np.array( %s )" % str(self.K.tolist()))
        logging.info("D=np.array( %s )" % str(self.D.tolist()))

    def draw_indexes_of_corners(self, img: np.ndarray, corners: list, show_image: bool = False, save_path: str = None):
        # print corner indexes into image
        logging.debug("Drawing indexes of corners to the image.")

        font = cv2.FONT_HERSHEY_SIMPLEX
        for idx, cor in enumerate(corners):
            cv2.putText(img, str(idx), tuple(cor[0]), font, 0.3, (0, 0, 255))
        if save_path:
            logging.info("Saving image with detected and numbered corners to  %s." % save_path)
            cv2.imwrite(save_path, img)
        if show_image:
            logging.info("Showing image with detected and numbered corners.")
            cv2.imshow("corners", img)

    def save_parameters(self, root_dir: str):

        os.makedirs(root_dir, exist_ok=True)  # create root dir of not exist

        # save parameter matrices
        np.save(os.path.join(root_dir, "K.npy"), self.K)
        np.save(os.path.join(root_dir, "D.npy"), self.D)

    def set_parameters(self, K: np.ndarray, D: np.ndarray):
        self.K = K
        self.D = D

    def undistort(self, image: Union[np.ndarray, str], result_crop: str="valid", save_path=None, show_image=False,
                  balance=0.0):
        """
        Undistort image using parameter matrices K and D
        :param image: image or path to the image
        :param result_crop - one of 'valid' or 'full', specify the resolution of the output image ('valid' means center
                             of the image without deformation of the borders)
        :param save_path - path to save undisorted image, not save if None
        :param show_image - show undistorted image
        :param balance - need to be specified if using 'full' result crop, means size of the crop (0.0 is like 'valid')
        :return: undistorted image
        """
        if isinstance(image, str):
            img = cv2.imread(image)
        elif isinstance(np.ndarray, image):
            img = image
        else:
            logging.error("Parameter must be an image or path to the image.")
            return None

        if result_crop == 'full':
            undistorted_img = self._undistort_full(img, balance)
        else:
            undistorted_img = self._undistort_valid(img)

        if save_path:
            logging.info("Saving undistorted image to  %s." % save_path)
            cv2.imwrite(save_path, undistorted_img)
        if show_image:
            logging.info("Showing undistorted image.")
            cv2.imshow("undistorted_image", undistorted_img)

        return undistorted_img

    def _undistort_valid(self, img: np.ndarray):

        map1, map2 = cv2.fisheye.initUndistortRectifyMap(self.K, self.D, np.eye(3),
                                                         self.K, self._img_shape, cv2.CV_16SC2)

        undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

        return undistorted_img

    def _undistort_full(self, img: np.ndarray, balance: float=0.0):

        dim1 = img.shape[:2][::-1]  # dim1 is the dimension of input image to un-distort
        assert dim1[0] / dim1[1] == self._img_shape[0] / self._img_shape[1],\
               "Image to un-distort needs to have same aspect ratio as the ones used in calibration"

        scaled_K = self.K * dim1[0] / self._img_shape[0]  # The values of K is to scale with image dimension.
        scaled_K[2][2] = 1.0  # Except that K[2][2] is always 1.0
        # This is how scaled_K, dim2 and balance are used to determine the final K used to un-distort image.
        # OpenCV document failed to make this clear!
        new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(scaled_K, self.D, dim1, np.eye(3), balance=balance)
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(scaled_K, self.D, np.eye(3), new_K, dim1, cv2.CV_16SC2)
        undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

        return undistorted_img

    def distort(self, image: np.ndarray):

        if isinstance(image, str):
            pass
        elif isinstance(np.ndarray, image):
            pass
        else:
            logging.error("Parameter must be an image or path to the image.")

    def load_calibration_images(self, image_dir: str):
        """
        Load calibration images from a directory
        :param image_dir: path to the directory with calibration images
        """

        logging.debug("Loading calibration images.")
        assert os.path.exists(image_dir), "Image dir does not exist."

        for idx, image_file in enumerate(os.listdir(image_dir)):

            try:
                img = cv2.imread(os.path.join(image_dir, image_file))

                if self._img_shape is None:
                    self._img_shape = img.shape[:2]
                else:
                    assert self._img_shape == img.shape[:2], "All images must share the same size."

            except:
                # file was not an image
                logging.warning("Could not read %s image file. Skipping this file."
                                % os.path.join(image_dir, image_file))
                continue

            self.calibration_images.append(img)

        logging.debug("Calibration images loaded.")
        logging.info("Found %s valid images for calibration." % len(self.calibration_images))
