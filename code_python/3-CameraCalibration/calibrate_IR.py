# %% ---------------------------------------------------------------------------
#                                   Imports
# ------------------------------------------------------------------------------
import os
from pyrealsense2 import pyrealsense2 as rs
from cv2 import cv2 as cv
import numpy as np


# %% ---------------------------------------------------------------------------
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


def get_depth_scale(pipeline_rs: rs.pipeline):
    """
    Returns the Depth Scale from the first depth sensor of the device on the
    active profile.

    example:
    depth_scale = get_depth_scale(pipeline_rs=pipeline)
    print("Depth Scale is [m/px_val]: ", depth_scale)

    @param pipeline_rs: A real sense pipeline
    @type pipeline_rs: rs.pipeline
    @return: Retrieves mapping between the units of the depth image and meters
    @rtype: float
    """
    # Gets the active profile from the pipeline; Gets the Device (ie. Intel
    # RealSense D435); Gets the 1st Depth Sensor from the Device; Gets the
    # Depth Scale and returns it.
    return pipeline_rs.get_active_profile().get_device(). \
        first_depth_sensor().get_depth_scale()


# ------------------------------------------------------------------------------
#                                       Main
# ------------------------------------------------------------------------------
FULL_CALIB = os.path.join("..", "..", "data",
                          "CV_D435_20201104_161043_Full_calibration.bag")
RGB_CALIB = os.path.join("..", "..", "data",
                         "CV_D435_20201104_160738_RGB_calibration.bag")
STREAM = os.path.join("..", "..", "data",
                      "CV_D435_20201104_162148.bag")

path = FULL_CALIB

# Creates a Real Sense Pipeline Object
pipeline = rs.pipeline(ctx=rs.context())

# Create a config object
config = rs.config()

# Tell config that we will use a recorded device from file to be used by
# the pipeline through playback (comment this line if you want to use a
# real camera).
config.enable_device_from_file(file_name=path, repeat_playback=True)
config = rs_config_color_pipeline(config_rs=config)
config = rs_config_IR_pipeline(config_rs=config)
config = rs_config_depth_pipeline(config_rs=config)

try:
    # Starts the pipeline with the configuration done previously
    pipeline.start(config=config)
except RuntimeError as err:
    print(err)
    raise RuntimeError("Make sure the config streams exists in the device!")

# Get intrinsics and extrinsics parameters from the multiple profiles of
# the pipeline
intrinsics_orig, extrinsics_orig = get_intrinsics_extrinsics(
    pipeline_rs=pipeline)

# Obtain the depth scale
depth_scale = get_depth_scale(pipeline_rs=pipeline)
print("Depth Scale is [m/px_val]: ", depth_scale)

# Distance between left and right IR cameras in meters. Cameras are
# assumed to be parallel to each other. We are assuming no distortion for
# all cameras
baseline = 0.05  # extrinsics["Infrared 2 -> Infrared 1"][1][0]

# Creates windows to display the frames
cv.namedWindow("Color Stream", cv.WINDOW_AUTOSIZE)

# Number of Inner Corners of the chessboard pattern
chess_size = (6, 9)

# Stores all corner points that where found on the image
image_l_points = []
image_r_points = []

# Creates a list with the real world object(chessboard pattern) coordinates
square_size = 2.5
obj_point = np.zeros(
    ((chess_size[0]) * (chess_size[1]),
     3), np.float32)
obj_point[:, :2] = np.mgrid[
                   0:(chess_size[0]) * square_size:square_size,
                   0:(chess_size[1]) * square_size:square_size] \
    .T.reshape(-1, 2)

# Used to store all the real world points of the chessboard pattern
obj_points = []

# Number of corners required to compute the calibration matrix
_MIN_CORNERS = 10

while True:
    # Wait for new frames and grabs the frameset
    frameset = pipeline.wait_for_frames()

    # Get IR Camera frame
    left_ir = np.asanyarray(frameset.get_infrared_frame(1).get_data())
    right_ir = np.asanyarray(frameset.get_infrared_frame(2).get_data())

    # Render image in opencv window
    cv.imshow("IR Stream", np.hstack((left_ir, right_ir)))

    # Find chessboard inner corners
    ret_val_l, corners_l = cv.findChessboardCorners(image=left_ir,
                                                    patternSize=chess_size)

    # If founded corners on left try to find on right
    if ret_val_l:
        ret_val_r, corners_r = cv.findChessboardCorners(image=left_ir,
                                                        patternSize=chess_size)
        # If found on left and right proceed to append points
        if ret_val_r:
            # Adds the real world coordinates to the array that stores the real
            # world coordinates.
            obj_points.append(obj_point)

            # Adds the image point to the array
            image_l_points.append(corners_l)
            image_r_points.append(corners_r)

            # Draws the corners on a image for visualization
            img_l_corners = cv.drawChessboardCorners(image=left_ir,
                                                     patternSize=chess_size,
                                                     corners=corners_l,
                                                     patternWasFound=ret_val_l)

            # Draws the corners on a image for visualization
            img_r_corners = cv.drawChessboardCorners(image=right_ir,
                                                     patternSize=chess_size,
                                                     corners=corners_r,
                                                     patternWasFound=ret_val_r)

            cv.imshow("ChessBoard Pattern", np.hstack((img_l_corners,
                                                       img_r_corners)))

    if len(image_r_points) >= _MIN_CORNERS:
        h, w = left_ir.shape[:2]
        _, cam_mat_l, dist_coefs_l, \
        cam_mat_r, dist_coefs_r, \
        rot_vec, trans_vec, \
        essent_mat, fund_mat = cv.stereoCalibrate(objectPoints=obj_points,
                                                  imagePoints1=image_l_points,
                                                  imagePoints2=image_r_points,
                                                  cameraMatrix1=None,
                                                  distCoeffs1=None,
                                                  cameraMatrix2=None,
                                                  distCoeffs2=None,
                                                  imageSize=(w, h))

        obj_points.pop(0)
        image_l_points.pop(0)
        image_r_points.pop(0)

        """
        mean_error = 0
        for i in range(len(obj_points)):
            imgpoints2, _ = cv.projectPoints(obj_points[i],
                                             rot_vec[i],
                                             trans_vec[i],
                                             cam_mat_l,
                                             dist_coefs_l)
            error = cv.norm(image_l_points[i], imgpoints2, cv.NORM_L2) / len(
                imgpoints2)
            mean_error += error

        print(f"error: {mean_error / len(obj_points)}\t")
        # """

    key = cv.waitKey(1)
    # if pressed ESCAPE exit program
    if key == 27:
        cv.destroyAllWindows()
        break
