# region Imports
# ------------------------------------------------------------------------------
#                                   Imports
# ------------------------------------------------------------------------------
import os
from enum import IntEnum
from datetime import datetime
import csv

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


# endregion

# region People Counter Functions & Classes
# ------------------------------------------------------------------------------
#                       People Counter Functions & Classes
# ------------------------------------------------------------------------------
def create_csv(f_path):
    _now = datetime.now()
    with open(f_path, mode='w', newline='') as csv_f:
        csv_file = csv.writer(csv_f)
        csv_file.writerow(
            ['# Group X', '2190383', 'Jose Rosa', '2192447', 'Ricardo Silva']
        )
        csv_file.writerow([_now.strftime("%H:%M:%S"), 'none', 0])


def write2csv(f_path, number, in_out):
    _now = datetime.now()
    with open(f_path, mode='a', newline='') as csv_f:
        csv_file = csv.writer(csv_f)
        csv_file.writerow([_now.strftime("%H:%M:%S"), in_out, number])


class MaskTypes(IntEnum):
    SUM = 1
    ANY = 1
    SUMMATION = 1
    AVG = 2
    AVERAGE = 2


class BackGroundTypes(IntEnum):
    BG = 0
    BACKGROUND = 0
    AVG = 1
    AVERAGE = 1


# endregion

# ------------------------------------------------------------------------------
#                                       Main
# ------------------------------------------------------------------------------
path = os.path.join("..", "..", "data", "CV_D435_20201104_162148.bag")

# timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
timestamp = ""
csv_path = os.path.join(f"dense_ppl_counter_{timestamp}.csv")
create_csv(csv_path)

# Creates a Real Sense Pipeline Object
pipeline = rs.pipeline(ctx=rs.context())

# Create a config object
config = rs.config()

# Tell config that we will use a recorded device from file to be used by
# the pipeline through playback (comment this line if you want to use a
# real camera).
config.enable_device_from_file(file_name=path, repeat_playback=True)
config = rs_config_color_pipeline(config_rs=config)
config = rs_config_depth_pipeline(config_rs=config)

try:
    # Starts the pipeline with the configuration done previously
    pipeline.start(config=config)
except RuntimeError as err:
    print(err)
    raise RuntimeError("Make sure the config streams exists in the device!")

# Create colorizer object to apply to depth frames
colorizer_qntz = rs.colorizer(7)  # Quantized
colorizer_gray = rs.colorizer(2)  # WhiteToBlack

# Creates a window to show color and depth stream
cv.namedWindow("Color - Depth Stream", cv.WINDOW_KEEPRATIO)

# Don't Touch! first_run is a flag!
_first_run = True

# Main cycle/loop
while True:
    # Read key and waits 1ms
    key = cv.waitKey(1)

    # Wait for new frames and grabs the frameset
    frameset = pipeline.wait_for_frames()

    # RS435 Depth Frame Object
    depth_frame = frameset.get_depth_frame()
    # Get Depth Frames with Color (Quantized Color Map)
    rs_depth_color = np.asanyarray(
        colorizer_qntz.colorize(depth_frame).get_data()
    )

    # Gray scale depth map based on the depth map with Quantized Color Map
    rs_depth_gray = cv.cvtColor(rs_depth_color, cv.COLOR_RGB2GRAY)

    # Applies a filtering process to enhance the tracking
    rs_depth_gray = cv.GaussianBlur(rs_depth_gray, (5, 5), sigmaX=0, sigmaY=0)

    # If is the first run...
    if _first_run:
        # The shape of the lists to hold the last image and draw N lines
        _shape = np.shape(rs_depth_color)

        # Disable first_run flag
        _first_run = False

    # if pressed ESCAPE exit program
    if key == 27:
        cv.destroyAllWindows()
        break
