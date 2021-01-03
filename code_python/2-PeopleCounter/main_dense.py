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

# timestamp = "_" + datetime.now().strftime('%Y%m%d_%H%M%S')
timestamp = ""
csv_path = os.path.join(f"dense_ppl_counter{timestamp}.csv")
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

# ROI: Region Of Interest. The region of interest where dense optical flow is
# computed using Gunnar Farneback's algorithm.
# ROI = [[Minimum Height, Maximum Height],
#        [Minimum Width, Maximum Width]]
# Example: ROI = [[0, None], [212, 636]]
# If ROI[0][1] or ROI[1][1] is None then that element is set to the maximum
# height and width respectively.
ROI = [[180, 420], [212, 706]]

NUM_IMG = 1
MIN_ANG = 45
MIN_MAG = 35
MAG_N_AVG = 4
ANG_N_AVG = 4

# Don't Touch! first_run is a flag!
_first_run = True
# Don't Touch! _people is a counter!
_people = 0

cv.namedWindow("Last - N Avg", cv.WINDOW_KEEPRATIO + cv.WINDOW_AUTOSIZE)
cv.namedWindow("Dense Optical Flow", cv.WINDOW_KEEPRATIO + cv.WINDOW_AUTOSIZE)

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

    # If is the first run...
    if _first_run:
        # The shape of the lists to hold the last image and draw N lines
        _shape = np.shape(rs_depth_color)

        # Initializes the Region of Interest of tracking
        if np.shape(ROI) == (2, 2):
            if ROI[0][1] is None:
                ROI[0][1] = _shape[0]
            if ROI[1][1] is None:
                ROI[1][1] = _shape[1]

            if any(ROI) is None:
                raise ValueError("_ROI_FEATURE_TRACKING can only contain None "
                                 "elements at index [0][1] and [1][1]. "
                                 "example: _ROI_FEATURE_TRACKING = [[0, "
                                 "None], [0, None]].")
        else:
            ROI = [[0, _shape[0]], [0, _shape[1]]]

        # Array to hold the last N previous images. Removes the last
        # dimensions because this array will hold gray scale images
        prev_imgs_gray = np.zeros((NUM_IMG,
                                   ROI[0][1] - ROI[0][0],
                                   ROI[1][1] - ROI[1][0]),
                                  dtype=np.uint8)

        # The previous frame is equal to the current frame
        prev_imgs_gray[-1] = np.copy(
            rs_depth_gray[ROI[0][0]:ROI[0][1], ROI[1][0]:ROI[1][1]]
        )

        # Array to hold the last overlay (To insert on top of the depth image
        # so we can visualize the dense optical flow)
        overlay = np.zeros((ROI[0][1] - ROI[0][0],
                            ROI[1][1] - ROI[1][0],
                            _shape[2]), dtype=np.uint8)
        # Sets image overlay saturation to maximum (Using HSV)
        overlay[:, :, 1] = 255

        # Creates the mask to display the optical flow on top of the full
        # size image
        mask = np.zeros(_shape)
        mask[ROI[0][0]:ROI[0][1], ROI[1][0]:ROI[1][1], :] = 1

        # Array to hold the last N previous images.
        imgs_color = np.zeros((NUM_IMG,
                               _shape[0],
                               _shape[1],
                               _shape[2]), dtype=np.uint8)

        # Adds current image to the array
        imgs_color[-1] = np.copy(rs_depth_color)

        """
        # Initializes the magnitudes array. This is needed so that the image
        # displayed corresponds to the N previous magnitudes
        mags = np.zeros((NUM_IMG,
                         ROI[0][1] - ROI[0][0],
                         ROI[1][1] - ROI[1][0]), dtype=np.float)

        # Initializes the angles array. This is needed so that the image
        #         # displayed corresponds to the N previous magnitudes
        angs = np.zeros((NUM_IMG,
                         ROI[0][1] - ROI[0][0],
                         ROI[1][1] - ROI[1][0]), dtype=np.float)
        """

        # Initilizes the bests arrays, that hold the best 4 magnitudes and
        # angles.
        _n_best = 4
        best_mags = np.zeros((MAG_N_AVG, _n_best), dtype=np.float)
        best_angs = np.zeros((ANG_N_AVG, _n_best), dtype=np.float)

        # Disable first_run flag
        _first_run = False

    # Computes the last N images average
    prev_N_gray_avg = np.rint(np.average(prev_imgs_gray, axis=0)).astype(
        np.uint8)

    # Crops the next image
    next_img_gray = rs_depth_gray[ROI[0][0]:ROI[0][1], ROI[1][0]:ROI[1][1]]

    # Opens a new window and displays the output frame
    cv.imshow("Last - N Avg", np.hstack((prev_imgs_gray[-1], prev_N_gray_avg)))

    flow = cv.calcOpticalFlowFarneback(prev=prev_N_gray_avg,
                                       next=next_img_gray,
                                       flow=None,
                                       pyr_scale=0.5,
                                       levels=3,
                                       winsize=9,
                                       iterations=3,
                                       poly_n=7,
                                       poly_sigma=1.5,
                                       flags=0)

    # Computes the magnitude and angle of the 2D vectors
    # mags[-1], angs[-1] = cv.cartToPolar(flow[..., 0], flow[..., 1])
    mags, angs = cv.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees=True)

    # Flattens the arrays to get the best 4 magnitudes
    mags_flat = np.ravel(mags)
    angs_flat = np.ravel(angs)

    # Sorts magnitudes on ascending order and grab the best 4 magnitudes!
    best_mags[-1] = np.sort(mags_flat)[-_n_best:]
    # Sets the maximum magnitude to a large value to enable the computation
    # of STD, otherwise the result is NaN.
    best_mags[-1][best_mags[-1] == np.Inf] = 1024

    # Sorts magnitudes on ascending order and grab the best 4 angles!
    best_angs[-1] = angs_flat[np.argsort(mags_flat)[-_n_best:]]

    # Computes the Standard Deviation of the best N values
    x, y = cv.polarToCart(best_mags.ravel(), best_angs.ravel())
    std = np.std(np.array([x, y]))

    print(f"MAG: {np.average(best_mags, axis=1)}\t"
          f"ANG: {np.average(best_angs)}\t"
          f"P: {_people}\t"
          f"std: {std}")

    # If the average magnitude (of the best N magnitudes) is high then some
    # big displacement happened
    if np.all(best_mags >= np.Inf):
        best_angs_avg = np.average(best_angs)
        # Check if the angle to determine the direction of the displacement
        if 90 - MIN_ANG < best_angs_avg < 90 + MIN_ANG:
            _people -= 1
            # Writes data to CSV
            write2csv(f_path=csv_path, number=_people, in_out="out")
            # Zeros the N best magnitudes and angles arrays so that a high
            # peek need to be reached again to count the people.
            best_mags = np.zeros((MAG_N_AVG, _n_best))
            best_angs = np.zeros((ANG_N_AVG, _n_best))
        elif 270 - MIN_ANG < best_angs_avg < 270 + MIN_ANG:
            _people += 1
            # Writes data to CSV
            write2csv(f_path=csv_path, number=_people, in_out="in")
            # Zeros the N best magnitudes and angles arrays so that a high
            # peek need to be reached again to count the people.
            best_mags = np.zeros((MAG_N_AVG, _n_best))
            best_angs = np.zeros((ANG_N_AVG, _n_best))

    # Sets the Hue (Color) according to the angle of the flow
    overlay[:, :, 0] = np.rint(angs / 2).astype(np.uint8)

    # Sets image Value (Black to Color) according to the optical flow
    # magnitude (normalized)
    overlay[:, :, 2] = cv.normalize(mags, None, 0, 255, cv.NORM_MINMAX)

    # Converts HSV to RGB (BGR) color representation
    overlay = cv.cvtColor(overlay, cv.COLOR_HSV2BGR)

    # Computes the render image to correspond to the average of the N last
    # images
    render = np.rint(
        np.average(imgs_color, axis=0)
    ).astype(np.uint8)
    # Sets the overlay (Dense Flow) to the render image
    render[ROI[0][0]:ROI[0][1], ROI[1][0]:ROI[1][1], :] = overlay
    # Opens a new window and displays the output frame
    cv.imshow("Dense Optical Flow", render)

    # Updates the last frame from the array
    prev_imgs_gray = np.roll(a=prev_imgs_gray, shift=-1, axis=0)
    prev_imgs_gray[-1] = np.copy(next_img_gray)

    # Updates the last frame from the array
    imgs_color = np.roll(a=imgs_color, shift=-1, axis=0)
    imgs_color[-1] = np.copy(rs_depth_color)

    # Sifts the arrays to left.
    mags = np.roll(a=mags, shift=-1, axis=0)
    angs = np.roll(a=angs, shift=-1, axis=0)
    best_mags = np.roll(a=best_mags, shift=-1, axis=0)
    best_angs = np.roll(a=best_angs, shift=-1, axis=0)

    # if pressed ESCAPE exit program
    if key == 27:
        cv.destroyAllWindows()
        break
