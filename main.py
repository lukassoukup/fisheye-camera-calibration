from CameraCalibration import FisheyeCameraCalibration
import cv2

if __name__ == "__main__":

    image_dir = 'Kalibracni_snimky'

    calibrator = FisheyeCameraCalibration(verbose=True)

    calibrator.load_calibration_images(image_dir)
    calibrator.calibrate()

    image = calibrator.undistort(image_dir + "\\image_00000.png")
    cv2.imshow("Undistorted image", image)
    cv2.waitKey()
