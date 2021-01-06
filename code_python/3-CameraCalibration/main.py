# region Imports
# ------------------------------------------------------------------------------
#                                   Imports
# ------------------------------------------------------------------------------
import os
from typing import Union
from datetime import datetime

from pyrealsense2 import pyrealsense2 as rs
from cv2 import cv2 as cv
import numpy as np


# endregion

# region Real Sense Functions
# ------------------------------------------------------------------------------
#                               Real Sense Functions
# ------------------------------------------------------------------------------
def rs_config_color_pipeline(config_rs: rs.config,
                             stream_type_rs: rs.stream = rs.stream.color,
                             width: int = 848,
                             height: int = 480,
                             format_rs: rs.format = rs.format.rgb8,
                             fps: int = 15):
    """
    Configs the pipeline to enable the color stream

    example:
    pipeline = rs.pipeline(ctx=rs.context())
    config = rs.config()
    config = rs_config_color_pipeline(config_rs=config)
    pipeline.start(config=config)

    @param config_rs: real sense configuration
    @type config_rs: rs.config
    @param stream_type_rs: Sets the stream type, default is rs.stream.color
    @type stream_type_rs: rs.stream
    @param width: The width of the stream in pixels, default is 848
    @type width: int
    @param height: The height of the stream in pixels, default is 480
    @type height: int
    @param format_rs: real sense stream format, default is rs.format.rgb8
    @type format_rs: rs.format
    @param fps: The fps of the stream, default is 15
    @type fps: int
    @return: The configuration file already configured
    @rtype: rs.config
    """
    # Configure the pipeline to stream the color stream
    config_rs.enable_stream(stream_type_rs,
                            width,
                            height,
                            format_rs,
                            fps)
    return config_rs


def rs_config_IR_pipeline(config_rs: rs.config,
                          stream_type_rs: rs.stream = rs.stream.infrared,
                          width: int = 848,
                          height: int = 480,
                          format_rs: rs.format = rs.format.y8,
                          fps: int = 15):
    """
    Configs the pipeline to enable the infrared (IR) left and right stream

    example:
    pipeline = rs.pipeline(ctx=rs.context())
    config = rs.config()
    config = rs_config_IR_pipeline(config_rs=config)
    pipeline.start(config=config)

    @param config_rs: real sense configuration
    @type config_rs: rs.config
    @param stream_type_rs: Sets the stream type, default is rs.stream.infrared
    @type stream_type_rs: rs.stream
    @param width: The width of the stream in pixels, default is 848
    @type width: int
    @param height: The height of the stream in pixels, default is 480
    @type height: int
    @param format_rs: real sense stream format, default is rs.format.y8
    @type format_rs: rs.format
    @param fps: The fps of the stream, default is 15
    @type fps: int
    @return: The configuration file already configured
    @rtype: rs.config
    """
    # https://github.com/IntelRealSense/librealsense/issues/1140
    # Configure the pipeline to stream the IR stream. One config to each cameras
    config_rs.enable_stream(stream_type_rs,
                            1,
                            width,
                            height,
                            format_rs,
                            fps)

    config_rs.enable_stream(stream_type_rs,
                            2,
                            width,
                            height,
                            format_rs,
                            fps)

    return config_rs


def rs_config_depth_pipeline(config_rs: rs.config,
                             stream_type_rs: rs.stream = rs.stream.depth,
                             width: int = 848,
                             height: int = 480,
                             format_rs: rs.format = rs.format.z16,
                             fps: int = 15):
    """
    Configs the pipeline to enable the depth stream

    example:
    pipeline = rs.pipeline(ctx=rs.context())
    config = rs.config()
    config = rs_config_depth_pipeline(config_rs=config)
    pipeline.start(config=config)

    @param config_rs: real sense configuration
    @type config_rs: rs.config
    @param stream_type_rs: Sets the stream type, default is rs.stream.depth
    @type stream_type_rs: rs.stream
    @param width: The width of the stream in pixels, default is 848
    @type width: int
    @param height: The height of the stream in pixels, default is 480
    @type height: int
    @param format_rs: real sense stream format, default is rs.format.z16
    @type format_rs: rs.format
    @param fps: The fps of the stream, default is 15
    @type fps: int
    @return: The configuration file already configured
    @rtype: rs.config
    """
    # https://github.com/IntelRealSense/librealsense/issues/1140
    # Configure the pipeline to stream the IR stream. One config to each cameras
    config_rs.enable_stream(stream_type_rs,
                            width,
                            height,
                            format_rs,
                            fps)

    return config_rs


def get_extrinsics(src: rs.video_stream_profile, dst: rs.video_stream_profile):
    """
    Returns R, T transform from src to dst.

    @param src: The source
    @type src: rs.video_stream_profile
    @param dst: The destiny
    @type dst: rs.video_stream_profile
    @return: Rotation and Transform Matrix
    @rtype: np.ndarray, np.ndarray
    """
    _extrinsics = src.get_extrinsics_to(dst)
    R = np.reshape(_extrinsics.rotation, [3, 3]).T
    T = np.array(_extrinsics.translation)
    return R, T


def unpack_profile(pipeline_rs: rs.pipeline):
    """
    The keys are the names of the streams ie. Depth, Infrared 1, Infrared 2,
    Color

    example:
    unpacked_profiles = unpack_profile(pipeline_rs=pipeline)

    @param pipeline_rs: Pipeline where profiles need to be extracted. Extracted
    features are: Type, Video Profile, Stream Profile and Unique ID.
    @type pipeline_rs: rs.pipeline_profile
    @return: Dictionary in which the key is the profile name and the values
    are the extracted properties: Type, Video Profile, Stream Profile and
    Unique ID.
    @rtype: dict
    """
    profiles = pipeline_rs.get_active_profile()

    unpacked = {}
    for profile in profiles.get_streams():
        _key = profile.stream_name()
        values = {
            "Type": profile.stream_type(),
            "Video Profile": profile.as_video_stream_profile(),
            "Stream Profile": profile.as_stream_profile(),
            "UID": profile.unique_id(),
        }
        unpacked[_key] = values

    return unpacked


def get_intrinsics_extrinsics(pipeline_rs: rs.pipeline):
    """
    Gets the intrinsics and extrinsics parameters of the available streams

    Intrinsics parameters are from every profile available.
    Extrinsics parameters can only be from Color to Infrared 1 or Infrared 2
    to Infrared 1.

    example:
    intrinsics, extrinsics = get_intrinsics_extrinsics(pipeline_rs=pipeline)

    @param pipeline_rs: The pipeline to extract the streams/profiles
    @type pipeline_rs: rs.pipeline
    @return: A dictionary with intrinsics parameters and another dictionary
    with extrinsics parameters
    @rtype: dict, dict
    """
    unpacked_profiles = unpack_profile(pipeline_rs=pipeline_rs)

    _intrinsics = {}
    for _key, _value in unpacked_profiles.items():
        _intrinsics[_key] = _value.get("Video Profile").get_intrinsics()

    _extrinsics = {}
    if unpacked_profiles.__contains__("Infrared 1"):
        if unpacked_profiles.__contains__("Color"):
            _extrinsics["Color -> Infrared 1"] = get_extrinsics(
                unpacked_profiles.get("Color").get("Video Profile"),
                unpacked_profiles.get("Infrared 1").get("Video Profile"))
        if unpacked_profiles.__contains__("Infrared 2"):
            _extrinsics["Infrared 2 -> Infrared 1"] = get_extrinsics(
                unpacked_profiles.get("Infrared 2").get("Video Profile"),
                unpacked_profiles.get("Infrared 1").get("Video Profile"))

    return _intrinsics, _extrinsics


# endregion

# region Utility Functions
# ------------------------------------------------------------------------------
#                               Utility Functions
# ------------------------------------------------------------------------------
def reject_outliers_2(data: np.ndarray, m: float = 2.):
    """
    Sets the outliers to 0 on an numpy array. Based on:
    https://stackoverflow.com/questions/11686720/is-there-a-numpy-builtin-to-reject-outliers-from-a-list
    """
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d / (mdev if mdev else 1.)
    return data[s < m]


# endregion

# ------------------------------------------------------------------------------
#                                       Main
# ------------------------------------------------------------------------------

path = os.path.join("..", "..", "data",
                    "CV_D435_20201104_160738_RGB_calibration.bag")

timestamp = "_" + datetime.now().strftime('%Y%m%d_%H%M%S')
save_path = os.path.join(f"calib_params{timestamp}")

# Number of corners required to compute the calibration matrix
_MIN_CORNERS = 40

# Sets the length of the chessboard square
square_size = 2.5  # Length of the square (2.5)
units = 0.01  # Units of square_size (cm)

# Distance between left and right IR cameras in meters. Cameras are
# assumed to be parallel to each other. We are assuming no distortion for
# all cameras
baseline = 0.05  # m

# Number of Inner Corners of the chessboard pattern
chess_rows = 6
chess_cols = 9

# Creates a Real Sense Pipeline Object
pipeline = rs.pipeline(ctx=rs.context())

# Create a config object
config = rs.config()

# Tell config that we will use a recorded device from file to be used by
# the pipeline through playback (comment this line if you want to use a
# real camera).
config.enable_device_from_file(file_name=path, repeat_playback=True)
config = rs_config_color_pipeline(config_rs=config)

try:
    # Starts the pipeline with the configuration done previously
    pipeline.start(config=config)
except RuntimeError as err:
    print(err)
    raise RuntimeError("Make sure the config streams exists in the device!")

# Create colorizer object to apply to depth frames
colorizer = rs.colorizer()

# Get intrinsics and extrinsics parameters from the multiple profiles of
# the pipeline
intrinsics, extrinsics = get_intrinsics_extrinsics(pipeline_rs=pipeline)

# patternSize = (points_per_row, points_per_column)
chess_size = (chess_rows, chess_cols)

# Stores all corner points that where found on the image
image_points = []

# Creates a list with the real world object(chessboard pattern) coordinates
obj_point = np.zeros((chess_rows * chess_cols, 3), dtype=np.float32)
obj_point[:, :2] = np.mgrid[
                   0:chess_rows * square_size:square_size,
                   0:chess_cols * square_size:square_size
                   ].T.reshape(-1, 2)
obj_point = obj_point * units

# Used to store all the real world points of the chessboard pattern
obj_points = []

# Window to show the stream
cv.namedWindow("Color Stream", cv.WINDOW_AUTOSIZE)

# FLAG (Don't touch)
first_run = True
trigger_pressed = False
# Main cycle/loop
while True:
    # Read key and waits 1ms
    key = cv.waitKey(1)

    # Wait for new frames and grabs the frameset
    frameset = pipeline.wait_for_frames()

    # Get RGB Camera frame
    rs_color_rgb = cv.cvtColor(
        np.asanyarray(frameset.get_color_frame().get_data()),
        cv.COLOR_BGR2RGB
    )
    rs_color_gray = cv.cvtColor(
        np.asanyarray(frameset.get_color_frame().get_data()),
        cv.COLOR_BGR2GRAY
    )

    # Gather image information on the first run
    if first_run:
        # Image dimensions
        _h_, _w_, _c_ = rs_color_rgb.shape[:3]

        # Resized image dimensions
        _h_rsz_, _w_rsz_ = cv.resize(
            src=rs_color_rgb,
            dsize=None,
            fx=1 / _MIN_CORNERS,
            fy=1 / _MIN_CORNERS).shape[:2]

        # Creates the space to hold the images
        image_corners = np.zeros((_h_rsz_, _w_, _c_))
        image_corners[:, :, :] = [65, 65, 65]  # Gray
        first_run = False

    # Creates a division bar and stacks it to the images
    div = np.zeros((4, _w_, _c_))  # NOQA
    div[:, :, :] = [100, 100, 65]  # Dark Cyan Blue
    image_bar = np.vstack((div, image_corners))  # NOQA

    if not trigger_pressed:
        _show = np.copy(rs_color_rgb)
        _show = cv.putText(img=_show,
                           text="PRESS <SPACE> TO CAPTURE",
                           org=(3, 26),
                           fontFace=cv.FONT_HERSHEY_SIMPLEX,
                           fontScale=1,
                           color=(0, 0, 255),
                           thickness=3,
                           bottomLeftOrigin=False)
    else:
        _show = rs_color_rgb

    # Render image in opencv window
    cv.imshow("Color Stream", np.uint8(np.vstack((_show, image_bar))))

    # If SPACE is pressed
    if key == 32:
        # Find the chessboard inner corners
        ret_val, corners = cv.findChessboardCorners(image=rs_color_gray,
                                                    patternSize=chess_size)
        trigger_pressed = True

        if ret_val:
            # Adds the real world coordinates to the array that stores the real
            # world coordinates.
            obj_points.append(obj_point)

            """
            corners = cv.cornerSubPix(image=rs_color_gray,
                                      corners=corners,
                                      winSize=(11, 11),
                                      zeroZone=(-1, -1),
                                      criteria=(cv.TERM_CRITERIA_EPS +
                                                cv.TERM_CRITERIA_MAX_ITER,
                                                30, 0.001))
            # """

            # Adds the image point to the array.
            image_points.append(corners)  # NOQA - Supressed
            # warnings on the current line. Used to prevent "Name corners can
            # be undefined"

            # Resizes the image to display it.
            _img_resized = cv.resize(
                src=cv.drawChessboardCorners(image=rs_color_rgb,
                                             patternSize=chess_size,
                                             corners=corners,
                                             patternWasFound=ret_val),
                dsize=None,
                fx=1 / _MIN_CORNERS,
                fy=1 / _MIN_CORNERS)

            # Stacks the resized image of the corners to the previous images
            image_corners = np.uint8(np.hstack((image_corners, _img_resized)))
            # Removes the oldest image.
            image_corners = image_corners[:, _w_rsz_:]  # NOQA

            # If the array of the image points have more than the minimum
            # images required for computation...
            if len(image_points) >= _MIN_CORNERS:
                # Removes the first entry, meaning the oldest one, of the
                # image_points and object_points
                obj_points.pop(0)
                image_points.pop(0)

                _, cam_mat, dist_coef, rot_vec, trans_vec = cv.calibrateCamera(
                    objectPoints=obj_points,
                    imagePoints=image_points,
                    imageSize=(rs_color_rgb.shape[1], rs_color_rgb.shape[0]),
                    cameraMatrix=None,
                    distCoeffs=None)

                rot_vec = np.array(rot_vec)
                """
                rotX_avg = np.sum(rot_vec[:, 0, :]) / _MIN_CORNERS
                rotY_avg = np.sum(rot_vec[:, 1, :]) / _MIN_CORNERS
                rotZ_avg = np.sum(rot_vec[:, 2, :]) / _MIN_CORNERS
                # """
                _inliers = reject_outliers_2(rot_vec[:, 0, :], m=2)
                rotX_avg = np.sum(_inliers) / len(_inliers)
                _inliers = reject_outliers_2(rot_vec[:, 1, :], m=2)
                rotY_avg = np.sum(_inliers) / len(_inliers)
                _inliers = reject_outliers_2(rot_vec[:, 2, :], m=2)
                rotZ_avg = np.sum(_inliers) / len(_inliers)
                rot_vec_avg = np.vstack((rotX_avg, rotY_avg, rotZ_avg))
                rot_mat = cv.Rodrigues(rot_vec_avg)[0]

                trans_vec = np.array(trans_vec)
                """
                transX_avg = np.sum(trans_vec[:, 0, :]) / _MIN_CORNERS
                transY_avg = np.sum(trans_vec[:, 1, :]) / _MIN_CORNERS
                transZ_avg = np.sum(trans_vec[:, 2, :]) / _MIN_CORNERS
                # """
                _inliers = reject_outliers_2(trans_vec[:, 0, :], m=2)
                transX_avg = np.sum(_inliers) / len(_inliers)
                _inliers = reject_outliers_2(trans_vec[:, 1, :], m=2)
                transY_avg = np.sum(_inliers) / len(_inliers)
                _inliers = reject_outliers_2(trans_vec[:, 2, :], m=2)
                transZ_avg = np.sum(_inliers) / len(_inliers)
                trans_vec_avg = np.vstack((transX_avg, transY_avg, transZ_avg))

                print("\n----------------------------------------")
                print("\tIntrinsics Matrix")
                print(np.round(cam_mat, 2))
                print("----------------------------------------")
                print("\tExtrinsic Matrix")
                print(np.round(np.hstack((trans_vec_avg, rot_mat)), 2))
                print("----------------------------------------\n")

                error_sum = 0
                for i in range(len(obj_points)):
                    image_points_reprojected, _ = cv.projectPoints(
                        objectPoints=obj_points[i],
                        rvec=rot_vec[i],
                        tvec=trans_vec[i],
                        cameraMatrix=cam_mat,
                        distCoeffs=dist_coef
                    )
                    error = cv.norm(
                        src1=image_points[i],
                        src2=image_points_reprojected,  # NOQA
                        normType=cv.NORM_L2
                    ) / len(image_points_reprojected)
                    error_sum += error

                avg_error = error_sum / len(obj_points)
                print(f"Error: {avg_error}")

                new_cam_mat, ROI = cv.getOptimalNewCameraMatrix(
                    cameraMatrix=cam_mat,
                    distCoeffs=dist_coef,
                    imageSize=(_w_, _h_),  # NOQA
                    alpha=1)

                img_undistorted = cv.undistort(src=rs_color_rgb,
                                               cameraMatrix=cam_mat,
                                               distCoeffs=dist_coef,
                                               dst=None,
                                               newCameraMatrix=new_cam_mat)

                rs_color_rgb = cv.putText(img=rs_color_rgb,
                                          text="PRESS <ESC> TO EXIT",
                                          org=(3, 26),
                                          fontFace=cv.FONT_HERSHEY_SIMPLEX,
                                          fontScale=1,
                                          color=(0, 0, 255),
                                          thickness=3,
                                          bottomLeftOrigin=False)

                rs_color_rgb = cv.resize(src=rs_color_rgb,
                                         dsize=None,
                                         fx=0.5,
                                         fy=0.5)

                img_undistorted = cv.resize(src=img_undistorted,
                                            dsize=None,
                                            fx=0.5,
                                            fy=0.5)

                cv.namedWindow("Original - Undistorted", cv.WINDOW_AUTOSIZE)
                cv.imshow("Original - Undistorted",
                          np.hstack((rs_color_rgb, img_undistorted)))

    # if pressed ESCAPE exit program
    if key == 27:
        cv.destroyAllWindows()
        break

try:
    np.savez(save_path, cam_mat, dist_coef,  # NOQA
             rvec=rot_vec_avg, tvec=trans_vec_avg,  # NOQA
             avg_error=avg_error)  # NOQA


except NameError:
    pass
