# ------------------------------------------------------------------------------
#                                   Imports
# ------------------------------------------------------------------------------
import argparse
import os
import time
import numpy as np
from cv2 import cv2 as cv
from pyrealsense2 import pyrealsense2 as rs


# ------------------------------------------------------------------------------
#                               Real Sense Functions
# ------------------------------------------------------------------------------
def rs_config_color_pipeline(config_rs: rs.config,
                             width: int = 848,
                             height: int = 480,
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
    @param width: The width of the stream in pixels, default is 848
    @type width: int
    @param height: The height of the stream in pixels, default is 480
    @type height: int
    @param fps: The fps of the stream, default is 15
    @type fps: int
    @return: The configuration file already configured
    @rtype: rs.config
    """
    # Configure the pipeline to stream the color stream
    config_rs.enable_stream(rs.stream.color,
                            width,
                            height,
                            rs.format.rgb8,
                            fps)
    return config_rs


def rs_config_IR_pipeline(config_rs: rs.config,
                          width: int = 848,
                          height: int = 480,
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
    @param width: The width of the stream in pixels, default is 848
    @type width: int
    @param height: The height of the stream in pixels, default is 480
    @type height: int
    @param fps: The fps of the stream, default is 15
    @type fps: int
    @return: The configuration file already configured
    @rtype: rs.config
    """
    # https://github.com/IntelRealSense/librealsense/issues/1140
    # Configure the pipeline to stream the IR stream. One config to each cameras
    config_rs.enable_stream(rs.stream.infrared,
                            1,
                            width,
                            height,
                            rs.format.y8,
                            fps)

    config_rs.enable_stream(rs.stream.infrared,
                            2,
                            width,
                            height,
                            rs.format.y8,
                            fps)

    return config_rs


def rs_config_depth_pipeline(config_rs: rs.config,
                             width: int = 848,
                             height: int = 480,
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
    @param width: The width of the stream in pixels, default is 848
    @type width: int
    @param height: The height of the stream in pixels, default is 480
    @type height: int
    @param fps: The fps of the stream, default is 15
    @type fps: int
    @return: The configuration file already configured
    @rtype: rs.config
    """
    # https://github.com/IntelRealSense/librealsense/issues/1140
    # Configure the pipeline to stream the IR stream. One config to each cameras
    config_rs.enable_stream(rs.stream.depth,
                            width,
                            height,
                            rs.format.z16,
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
#                               Script Functions
# ------------------------------------------------------------------------------
# Wait for the input of the keyboard or by the delay to be reach
def trigger_function(_t_=None):
    if _t_ is None:
        time.sleep(_trig / 1000)
        return
    elif _t_ == _trig:
        return True


def find_points_and_append(_img: np.ndarray, _show: bool = False):
    # Number of inner corners per a chessboard row and column ( patternSize =
    # cv::Size(points_per_row,points_per_colum) = cv::Size(columns,rows)
    _img_gray = cv.cvtColor(_img, cv.COLOR_RGB2GRAY)
    _ret_val, _corners = cv.findChessboardCorners(image=_img_gray,
                                                  patternSize=_chess)

    if _ret_val:
        # Adds the real world coordinates to the array that stores the real
        # world coordinates.
        obj_points.append(obj_point)

        # Gets the corners with subpixel accuracy
        _corners_accurate = cv.cornerSubPix(image=_img_gray,
                                            corners=_corners,
                                            winSize=(11, 11),
                                            zeroZone=(-1, -1),
                                            criteria=criteria)

        # Adds the image point to the array
        image_points.append(_corners_accurate)

        # Draws the corners on a image for visualization
        img_corners = cv.drawChessboardCorners(image=_img,
                                               patternSize=_chess,
                                               corners=_corners_accurate,
                                               patternWasFound=_ret_val)
        cv.imshow("ChessBoard Pattern", img_corners)


def calibrate_and_return(_img: np.ndarray):
    if len(image_points) >= args.min:
        h, w = _img.shape[:2]
        return cv.calibrateCamera(
            objectPoints=obj_points,
            imagePoints=image_points,
            imageSize=(w, h),
            cameraMatrix=None,
            distCoeffs=None)
    else:
        return False


# ------------------------------------------------------------------------------
#                                   Main
# ------------------------------------------------------------------------------
# Creates Parser
parser = argparse.ArgumentParser(description="Camera Calibration from a ROS "
                                             "stream/file.")

# Add arguments
parser.add_argument("path", type=str, nargs='?', help="BAG File Path.")
parser.add_argument("stream_types", nargs='?',
                    choices=["C", "I", "D"],
                    help="The stream type(s)."
                         " C → Color; I → Infrared;"
                         " D → Depth.")
parser.add_argument("-c", "--chess", type=int, nargs=2,
                    default=[7, 7],
                    help="The number of inner corners of the chess pattern. "
                         "The size must be a tuple with the number of rows "
                         "and the number of columns of the chess pattern. "
                         "example: (rows,cols). Default is (7,7).")
parser.add_argument("-cs", "--chess_square", type=float, nargs='?',
                    default=1,
                    help="The size of the chess squares. Default is 1, "
                         "so the units are relative or, in other words in "
                         "\"chess squares\".")
parser.add_argument("-m", "--min", type=int, nargs='?', default=10,
                    help="The minimum number of images to compute the "
                         "extrinsics and intrinsics parameters. Default is 10.")
parser.add_argument("-sw", "--stream_width", type=int, nargs='?', default=848,
                    help="The stream width in pixels. Default is 848.")
parser.add_argument("-sh", "--stream_height", type=int, nargs='?', default=480,
                    help="The stream height in pixels. Default is 480.")
parser.add_argument("-sfps", "--stream_fps", type=int, nargs='?', default=15,
                    help="The stream frames per second (FPS). Default is 15.")
parser.add_argument("-t", "--trigger", type=str, nargs='?',
                    default=10000,
                    help="The key which is pressed to grab that frame into "
                         "the calibration. This makes the calibration process "
                         "manual, meaning that you should watch the stream "
                         "and press the key when you see a good frame for "
                         "calibration. If a integer is passed automatic "
                         "calibration is set, meaning that a frame will be "
                         "grabbed in the desired interval in ms. Default is "
                         "10000.")

# Parse Arguments
args = parser.parse_args()

# Retrieve Arguments
f_path = args.path

if not os.path.exists(f_path):
    raise FileNotFoundError(f"{f_path} not found!")

if os.path.splitext(f_path)[-1].lower() != ".bag":
    raise FileExistsError(f"{f_path} is not a .bag file!")

try:
    _trig = int(args.trigger)
except ValueError:
    if len(args.trigger) != 1:
        raise ValueError(f"Trigger ({args.trigger}) must be a int or a str with"
                         f" length equal to 1!")
    else:
        _trig = args.trigger

_chess = tuple(args.chess)

# Creates a Real Sense Pipeline Object
pipeline = rs.pipeline(ctx=rs.context())

# Create a config object
config = rs.config()

# Tell config that we will use a recorded device from file to be used by
# the pipeline through playback (comment this line if you want to use a
# real camera).
config.enable_device_from_file(file_name=f_path,
                               repeat_playback=False)

if "C" in args.stream_types:
    config = rs_config_color_pipeline(config_rs=config,
                                      width=args.stream_width,
                                      height=args.stream_height,
                                      fps=args.stream_fps)
elif "I" in args.stream_types:
    config = rs_config_IR_pipeline(config_rs=config,
                                   width=args.stream_width,
                                   height=args.stream_height,
                                   fps=args.stream_fps)
elif "D" in args.stream_types:
    config = rs_config_depth_pipeline(config_rs=config,
                                      width=args.stream_width,
                                      height=args.stream_height,
                                      fps=args.stream_fps)

    # Creates a colorizer for depth streams
    colorizer = rs.colorizer()

try:
    # Starts the pipeline with the configuration done previously
    pipeline.start(config=config)
except RuntimeError as err:
    print(err)
    raise RuntimeError("Make sure the config streams exists in the device!")

# Termination criteria set the desired accuracy to 0.001 and  and maximum
# iterations to 30
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Stores all corner points that where found on the image
image_points = []

obj_point = np.zeros(
    ((_chess[0]) * (_chess[1]),
     3), np.float32)
obj_point[:, :2] = np.mgrid[
                   0:(_chess[0]) * args.chess_square:args.chess_square,
                   0:(_chess[1]) * args.chess_square:args.chess_square] \
    .T.reshape(-1, 2)

# Used to store all the real world points of the chessboard pattern
obj_points = []

# Friendly message
print("Press ESC to abort.")
if type(_trig) is str:
    print(f"Press {_trig} to snapshot.")

while True:
    _key = cv.waitKey(1)

    frameset = pipeline.wait_for_frames()

    if "C" in args.stream_types:

        _color = cv.cvtColor(np.asanyarray(frameset
                                           .get_color_frame()
                                           .get_data()), cv.COLOR_BGR2RGB)

    elif "I" in args.stream_types:
        _ir_1 = np.asanyarray(frameset.get_infrared_frame(1).get_data())
        _ir_2 = np.asanyarray(frameset.get_infrared_frame(2).get_data())

    elif "D" in args.stream_types:
        _depth = np.asanyarray(colorizer
                               .colorize(frameset.get_depth_frame()).get_data())
        ret_val, corners = cv.findChessboardCorners(image=_depth,
                                                    patternSize=_chess)

    # if pressed ESCAPE exit program
    if _key == 27:
        cv.destroyAllWindows()
        break
