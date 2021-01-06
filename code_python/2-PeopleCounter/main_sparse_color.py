# region Imports
# ------------------------------------------------------------------------------
#                                   Imports
# ------------------------------------------------------------------------------
import os
from enum import IntEnum
from datetime import datetime
import csv
import time

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

# region User Parameters
# ------------------------------------------------------------------------------
#                                   User Parameters
# ------------------------------------------------------------------------------
# Defines the path of the BAG file containing the streams
path = os.path.join("..", "..", "data", "CV_D435_20201104_162148.bag")

# Defines the path where the CSV file is saved. The CSV file will store the
# number of people that entered and exited the room with a timestamp.
# timestamp = "_" + datetime.now().strftime('%Y%m%d_%H%M%S')
timestamp = ""
csv_path = os.path.join(f"sparse_color_ppl_counter{timestamp}.csv")

# Maximum features that cv.goodFeaturesToTrack retrieve. (Shi-Tomasi Corner
# Detector)
MAX_FEATURES = 60

# If True the already tracked features have priority meaning that if new
# features are found these will not replace the already tracked features to a
# maximum of MAX_FEATURES.
# If set to False then the new features will always be preserved to a maximum
# of MAX_FEATURES.
PRIORITY_TRACKED = True

# Minimum frames/iterations to search for new features. If 0 then always search
# for new features on new frames.
MIN_ITER = 0

# SHI TOMASI Feature Detector Quality Level
QUALITY_LVL_SHITOMASI = 0.2

# Number of lines to retain in the image presented to the user for each keypoint
NUM_LINES = 4

# The type of the background to compute the absolute differences.
#   - If BackGroundTypes.BG: The first frame is always compared with the rest
#   of the frames and the difference is computed from there.
#   - If BackGroundTypes.AVG: The N last images are averaged and the absolute
#   difference is computed. The N last frames is setted by _NUM_IMG.
# The background type is used to compute the mask to try to remove unintended
# points from the shi-tomasi computation.
BACKGROUND_TYPE = BackGroundTypes.AVG
NUM_IMG = 4

# The type of processing to compute the mask to find new feature points on
# each _MIN_ITER. The number of masks that are taken into account is given by
# the value set by _NUM_MASKS.
#   - If MaskTypes.AVG: The N last masks are averaged. If average of mask is
#   greater than 0.5 of its maximum then the mask is set to 1, meaning that
#   the shi-tomasi corner detector will try to find a keypoint there,
#   otherwise is set to 0 and shi-tomasi corner detector will ignore the area.
#   - If MaskTypes.ANY: The N last masks are summed. If the any of these
#   masks is 1 then the shi-tomasi corner detector will try to find keypoints
#   on that region.
# Everytime a mask is processed are took into account the last points that
# were found so that the same points are not found again.
# The mask is computed with the background type defined above.
MASK_TYPE = MaskTypes.AVG
NUM_MASKS = 2

# ROI: Region Of Interest. The region of interest where the new keypoints are
# generated/searched using cv.goodFeaturesToTrack.
# ROI_NEW_FEATURES = [[Minimum Height, Maximum Height],
#                     [Minimum Width, Maximum Width]]
# Example: ROI_NEW_FEATURES = [[120, 360], [212, 636]]
# If ROI_NEW_FEATURES = None then its set to be all the image.
ROI_NEW_FEATURES = [[300, 400], [260, 640]]

# ROI: Region Of Interest. The region of interest where the keypoints that
# are tracked via cv.calcOpticalFlowPyrLK are maintained. The keypoints
# outside this region are discarded.
# ROI_FEATURE_TRACKING = [[Minimum Height, Maximum Height],
#                         [Minimum Width, Maximum Width]]
# Example: ROI_FEATURE_TRACKING = [[0, 480], [212, 636]]
# If ROI_FEATURE_TRACKING[0][1] or ROI_FEATURE_TRACKING[1][1] is None then
# that element is set to the maximum height and width respectively.
ROI_FEATURE_TRACKING = [[230, 460], [220, 680]]

# Minimum magnitude needed to check if angle and computes the number of persons
MIN_MAG = 60

# Minimum angle displacement around 90º and -90º to check the direction of
# the movement. Note that magnitude requirement is also needed.
# 90 - _MIN_ANG < angle < 90 + _MIN_ANG -> person -1
# -90 - _MIN_ANG < angle < -90 + _MIN_ANG -> person +1
MIN_ANG = 45

# The number of magnitudes values to store x4. The value is multiplied by 4
# because the 4 best magnitudes per iteration are stored. This values are used
# to compute the average and compare this average with _MIN_MAG.
MAG_N_AVG = 8

# The number of angles values to store x4. This values is multiplied by 4
# because we store the 4 angles of the best 4 magnitudes. These are used to
# compute the average and compare it to the range defined by _MIN_ANG.
ANG_N_AVG = 4

"""
@note Quantization showed worst results. Is preferred to use the quantized 
color map and set the window size for median filter.
"""
# Sets the median filter to apply to the image.
KSize_MEDIAN_FILTER = 1

# Set frame rate to constant value to avoid different behaviors on different
# devices or CPUs.
FPS = 10
# endregion

# region Main
# ------------------------------------------------------------------------------
#                                   Main
# ------------------------------------------------------------------------------
# Creates the CSV File
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

# Creates a window to show color and depth stream
cv.namedWindow("Sparse Optical Flow (on Depth Image)",
               cv.WINDOW_KEEPRATIO + cv.WINDOW_AUTOSIZE)

# Don't Touch! INTERNAL VARIABLES!
_iter_count = 0
_people = 0
_first_run = True
_fps_t = 1 / FPS
_t = time.time()
# Main cycle/loop
while True:
    _t += _fps_t

    # Read key and waits 1ms
    key = cv.waitKey(1)

    # Wait for new frames and grabs the frameset
    frameset = pipeline.wait_for_frames()

    # RS435 Depth Frame Object
    color_frame = frameset.get_color_frame()

    # Get Depth Frames with Color (Quantized Color Map)
    rs_color = cv.cvtColor(np.asanyarray(color_frame.get_data()),
                           cv.COLOR_BGR2RGB)

    # Remove some noise
    rs_color = cv.medianBlur(rs_color, KSize_MEDIAN_FILTER)

    # Gray scale depth map based on the depth map with Quantized Color Map
    rs_gray = cv.cvtColor(rs_color, cv.COLOR_RGB2GRAY)

    # If is the first run...
    if _first_run:
        # The shape of the lists to hold the last image and draw N lines
        _shape = np.shape(rs_color)

        # Array to hold the last N previous images. Removes the last
        # dimensions because this array will hold gray scale images
        previous_images_gray = np.zeros((NUM_IMG, _shape[0], _shape[1]),
                                        dtype=np.uint8)

        # The previous frame is equal to the current frame
        previous_images_gray[-1] = np.copy(rs_gray)

        # Sets the background as the first image (Assumption)
        background = np.copy(rs_gray)

        # Array to hold the last N overlays (To insert on top of the depth
        # image so we can visualize the tracking)
        overlay = np.zeros((NUM_LINES, _shape[0], _shape[1], _shape[2]),
                           dtype=np.uint8)

        # Initializes the mask so that it corresponds to the Region of
        # Interest for creation of new keypoints
        if np.shape(ROI_NEW_FEATURES) == (2, 2):
            mask = np.zeros((_shape[0], _shape[1]), dtype=np.uint8)
            mask[ROI_NEW_FEATURES[0][0]:ROI_NEW_FEATURES[0][1],
            ROI_NEW_FEATURES[1][0]:ROI_NEW_FEATURES[1][1]] = 1
        else:
            ROI_NEW_FEATURES = [[0, _shape[0]], [0, _shape[1]]]
            mask = None

        # Initializes the Region of Interest of tracking
        if np.shape(ROI_FEATURE_TRACKING) == (2, 2):
            if ROI_FEATURE_TRACKING[0][1] is None:
                ROI_FEATURE_TRACKING[0][1] = _shape[0]
            if ROI_FEATURE_TRACKING[1][1] is None:
                ROI_FEATURE_TRACKING[1][1] = _shape[1]

            if any(ROI_FEATURE_TRACKING) is None:
                raise ValueError("_ROI_FEATURE_TRACKING can only contain None "
                                 "elements at index [0][1] and [1][1]. "
                                 "example: _ROI_FEATURE_TRACKING = [[0, "
                                 "None], [0, None]].")
        else:
            ROI_FEATURE_TRACKING = [[0, _shape[0]], [0, _shape[1]]]

        # Sets the previous features as the new ones
        features_prev = cv.goodFeaturesToTrack(
            image=previous_images_gray[-1],  # NOQA
            maxCorners=MAX_FEATURES,
            qualityLevel=QUALITY_LVL_SHITOMASI,
            minDistance=2,
            blockSize=7,
            mask=mask
        )

        # Initializes the mask array to hold the N previous masks
        masks = np.zeros((NUM_MASKS, _shape[0], _shape[1]), dtype=np.uint8)

        # Initializes the best magnitudes and angles array to hold the N
        # (_MAG_N_AVG and _ANG_N_AVG) previous best 4 magnitudes and angle
        # values. Used to compute the average of the angle and magnitude.
        best_mags = np.zeros((MAG_N_AVG, 4))
        best_angs = np.zeros((ANG_N_AVG, 4))

        # Initializes Fixed Overlay that hold the background object that are
        # constant
        fixed_overlay = np.zeros(_shape, dtype=np.uint8)

        # Draws a rectangle around the Region of Interest of new Features
        fixed_overlay = cv.rectangle(img=fixed_overlay,
                                     pt1=tuple([ROI_NEW_FEATURES[1][0],
                                                ROI_NEW_FEATURES[0][0]]),
                                     pt2=tuple([ROI_NEW_FEATURES[1][1],
                                                ROI_NEW_FEATURES[0][1]]),
                                     color=(0, 255, 0),
                                     thickness=3)

        # Draws a rectangle around the Region of Interest Tracking
        fixed_overlay = cv.rectangle(img=fixed_overlay,
                                     pt1=tuple([ROI_FEATURE_TRACKING[1][0],
                                                ROI_FEATURE_TRACKING[0][0]]),
                                     pt2=tuple([ROI_FEATURE_TRACKING[1][1],
                                                ROI_FEATURE_TRACKING[0][1]]),
                                     color=(0, 0, 255),
                                     thickness=3)

        # Draws text showing the number of people inside the room
        fixed_overlay = cv.putText(img=fixed_overlay,
                                   text="New Features",
                                   org=tuple(
                                       [ROI_NEW_FEATURES[1][0] + 15,
                                        ROI_NEW_FEATURES[0][0] + 15]
                                   ),
                                   fontFace=cv.FONT_HERSHEY_SIMPLEX,
                                   fontScale=0.5,
                                   color=(0, 255, 0),
                                   thickness=2)

        # Draws text showing the number of people inside the room
        fixed_overlay = cv.putText(img=fixed_overlay,
                                   text="Tracking",
                                   org=tuple(
                                       [ROI_FEATURE_TRACKING[1][0] + 15,
                                        ROI_FEATURE_TRACKING[0][0] + 15]
                                   ),
                                   fontFace=cv.FONT_HERSHEY_SIMPLEX,
                                   fontScale=0.5,
                                   color=(0, 0, 255),
                                   thickness=2)

        # Disable first_run flag
        _first_run = False

    # Calculates sparse optical flow by Lucas-Kanade method
    # https://docs.opencv.org/3.0-beta/modules/video/doc/motion_analysis_and_object_tracking.html#calcopticalflowpyrlk
    features_next, status, error = cv.calcOpticalFlowPyrLK(
        prevImg=previous_images_gray[-1],  # NOQA
        nextImg=rs_gray,
        prevPts=features_prev,  # NOQA
        nextPts=None,
        winSize=(15, 15),
        maxLevel=4,
        criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03)
    )

    try:
        # _ROI_FEATURE_TRACKING = [[Minimum Height, Maximum Height],
        #                          [Minimum Width, Maximum Width]]
        status[features_next[:, 0, 1] < ROI_FEATURE_TRACKING[0][0]] = 0
        status[features_next[:, 0, 1] > ROI_FEATURE_TRACKING[0][1]] = 0
        status[features_next[:, 0, 0] < ROI_FEATURE_TRACKING[1][0]] = 0
        status[features_next[:, 0, 0] > ROI_FEATURE_TRACKING[1][1]] = 0

        # Selects good feature points for previous position
        features_prev_good = features_prev[status == 1]

        # Selects good feature points for next position
        features_next_good = features_next[status == 1]

        """
        # Shows the previous and next image
        cv.imshow("Prev - Next",
                  np.hstack((previous_images_gray[-1], rs_depth_gray))
                  )
        # """

        # A random line color per iteration
        color = tuple(np.random.randint(0, 256, size=3).tolist())  # NOQA

        # Draws the optical flow tracks
        for i, (new, old) in enumerate(
                zip(features_next_good, features_prev_good)):
            # Returns a contiguous flattened array as (x, y) coordinates for new
            # point
            a, b = new.ravel()
            # Returns a contiguous flattened array as (x, y) coordinates for old
            # point
            c, d = old.ravel()

            # Draws line between new and old position with green color and 2
            # thickness. The new image (with lines) is drawn on a blank array and
            # sets the last element of the overlays list. This is done so that we
            # only view the last N lines.
            overlay[-1] = cv.line(img=overlay[-1],  # NOQA
                                  pt1=(a, b),
                                  pt2=(c, d),
                                  color=color,
                                  thickness=2)
            # Draws filled circle (thickness of -1) at new position with green
            # color and radius of 3
            rs_color = cv.circle(rs_color, (a, b), 3, color, -1)

        # Adds the fixed overlay to background
        rs_color = cv.add(rs_color, fixed_overlay)  # NOQA

        # Draws text showing the number of people inside the room
        rs_color = cv.putText(img=rs_color,
                              text=str(_people),
                              org=(5, _shape[0] - 5),  # NOQA
                              fontFace=cv.FONT_HERSHEY_SIMPLEX,
                              fontScale=1,
                              color=(255, 0, 255),
                              thickness=3)

        # Adds all the overlays and then adds it to the image to show
        output = cv.add(rs_color, np.sum(overlay, axis=0, dtype=np.uint8))

        # Opens a new window and displays the output frame
        cv.imshow("Sparse Optical Flow (on Depth Image)", output)

        # Sift the overlay array to left and sets the last element to zeros
        overlay = np.roll(a=overlay, shift=-1, axis=0)
        overlay[-1] = np.zeros_like(overlay[-1])

        # Computes the magnitude for all group of points (Line between previous and
        # next points)
        mags = np.sqrt(
            np.square(features_next_good[:, 0] - features_prev_good[:, 0]) +
            np.square(features_next_good[:, 1] - features_prev_good[:, 1])
        )

        # Computes the angle for all group of points (Line between previous and
        # next points)
        angs = np.arctan2(
            features_next_good[:, 1] - features_prev_good[:, 1],
            features_next_good[:, 0] - features_prev_good[:, 0]
        ) * 180 / np.pi

        # If the array have not the 4 "best" required elements then just add
        # nulls to avoid ValueError exception
        if len(mags) < 4:
            mags = np.hstack((mags, np.zeros((4 - len(mags)))))
            angs = np.hstack((angs, np.zeros((4 - len(angs)))))

        # Sorts the magnitudes in ascending order and grabs the best 4 magnitudes.
        # Thee best magnitudes are added to an array to compute the average of
        # the N sets of best magnitudes.
        best_mags[-1] = np.sort(mags)[-4:]  # NOQA

        # Sorts the magnitudes in ascending order, gets the index of best 4
        # magnitudes and gets the angles corresponding to those best magnitudes.
        best_angs[-1] = angs[np.argsort(mags)[-4:]]  # NOQA

        # Computes the average of the angle and the magnitude
        best_mags_avg = np.average(best_mags)
        best_angs_avg = np.average(best_angs)

        # If the average magnitude (of the best N magnitudes) is high then some
        # big displacement happened
        if best_mags_avg > MIN_MAG:
            # Check if the angle to determine the direction of the displacement
            if 90 - MIN_ANG < best_angs_avg < 90 + MIN_ANG:
                _people -= 1
                # Writes data to CSV
                write2csv(f_path=csv_path, number=_people, in_out="out")
                # Zeros the N best magnitudes and angles arrays so that a high
                # peek need to be reached again to count the people.
                best_mags = np.zeros((MAG_N_AVG, 4))
                best_angs = np.zeros((ANG_N_AVG, 4))
            elif -90 - MIN_ANG < best_angs_avg < -90 + MIN_ANG:
                _people += 1
                # Writes data to CSV
                write2csv(f_path=csv_path, number=_people, in_out="in")
                # Zeros the N best magnitudes and angles arrays so that a high
                # peek need to be reached again to count the people.
                best_mags = np.zeros((MAG_N_AVG, 4))
                best_angs = np.zeros((ANG_N_AVG, 4))

        # Shifts the best magnitudes and angles array so that the last
        # element is the oldest one and zeros that element.
        best_mags = np.roll(a=best_mags, shift=-1, axis=0)
        best_angs = np.roll(a=best_angs, shift=-1, axis=0)
        best_mags[-1] = np.zeros_like(best_mags[-1])
        best_angs[-1] = np.zeros_like(best_angs[-1])

        # All processing is done here! If Processing in all iterations set
        # _MIN_ITER to 0.
        if _iter_count > MIN_ITER:
            if BACKGROUND_TYPE == BackGroundTypes.AVERAGE:
                # Average of N last images
                previous_N_avg = np.rint(np.average(
                    previous_images_gray, axis=0
                )).astype(np.uint8)
            else:
                previous_N_avg = background  # NOQA

            # Absolute difference to compute the last mask (1 = compute;
            # 0 = ignore). The mask takes into account the region of interest!
            masks[-1][ROI_NEW_FEATURES[0][0]:ROI_NEW_FEATURES[0][1],  # NOQA
            ROI_NEW_FEATURES[1][0]:ROI_NEW_FEATURES[1][1]] = \
                abs(rs_gray[ROI_NEW_FEATURES[0][0]:ROI_NEW_FEATURES[0][1],
                    ROI_NEW_FEATURES[1][0]:ROI_NEW_FEATURES[1][1]] -
                    previous_N_avg[
                    ROI_NEW_FEATURES[0][0]:ROI_NEW_FEATURES[0][1],
                    ROI_NEW_FEATURES[1][0]:ROI_NEW_FEATURES[1][1]])

            # Converts the mask to binary.
            masks[-1] = cv.threshold(src=masks[-1],
                                     thresh=0.5 * np.max(masks[-1]),
                                     maxval=1,
                                     type=cv.THRESH_BINARY)[1]

            # Sets the last mask to 0 when this corresponds to an point that
            # already exists
            _k_size = 3
            for (w, h) in features_next_good:
                w = np.rint(w).astype(np.uint)
                h = np.rint(h).astype(np.uint)
                try:
                    masks[-1][h - _k_size:h + _k_size,
                    w - _k_size:w + _k_size] = 0
                except IndexError:
                    masks[-1] = 0

            if MASK_TYPE == MaskTypes.AVERAGE:
                mask_avg = np.rint(np.average(masks, axis=0)).astype(np.uint8)
                mask_avg = cv.threshold(src=mask_avg,
                                        thresh=0.5 * np.max(masks),
                                        maxval=1,
                                        type=cv.THRESH_BINARY)[1]
            else:
                mask_avg = np.sum(masks, axis=0).astype(np.uint8)
                mask_avg[mask_avg >= 1] = 1

            """
            # Shows the Last N Images
            cv.imshow("N LAST IMAGES", previous_N_avg)
    
            # Remaps the 1 to 255 so we can visualize the mask
            _show = mask_avg
            _show[mask_avg == 1] = 255
            # Shows the Mask
            cv.imshow("MASK", _show)
            # """

            # Get features to track using Shi-Tomasi Corner Detector
            features_current = cv.goodFeaturesToTrack(
                image=rs_gray,
                maxCorners=MAX_FEATURES,
                qualityLevel=QUALITY_LVL_SHITOMASI,
                minDistance=2,
                mask=mask_avg,
                blockSize=7
            )

            # Shifts the mask array to left and set the oldest element (now on
            # right) to zeros.
            masks = np.roll(a=masks, shift=-1, axis=0)
            masks[-1] = np.zeros_like(masks[-1])

            # If there is some features found (not None)
            if features_current is not None:
                # Lets add them to the features to use on next iteration
                if PRIORITY_TRACKED:
                    # Always add the new features
                    features_prev = np.vstack((
                        features_current,
                        features_next_good.reshape(-1, 1, 2)
                    ))
                else:
                    # Always add the already being tracked features
                    features_prev = np.vstack((
                        features_next_good.reshape(-1, 1, 2),
                        features_current
                    ))
            # If its None then just do the usual... The features to use on next
            # iteration are the ones that were tracked.
            else:
                features_prev = features_next_good.reshape(-1, 1, 2)

            # Retains only N features (Don't forget the features are ordered by
            # ascending quality!) This may lead
            if len(features_prev) > MAX_FEATURES:
                features_prev = features_prev[:MAX_FEATURES, :, :]

            # Resets Counter
            _iter_count = 0

        # _iter_count <= MIN_ITER: don't search new features to track
        else:
            # Usual
            features_prev = features_next_good.reshape(-1, 1, 2)

    # Sometimes the calcOpticalFlowPyrLK will return an empty object (
    # NoneType) because it failed to track the features. In this situation we
    # just try to track new ones on the next iteration searching for new good
    # feature to track using shi-tomasi. This rarely happens and a good
    # solution to this problem is to reduce the mask array size.
    except TypeError:
        print("\033[91m"
              "Failed to track the previous features. Generating new ones!"
              "\033[0m")
        # Initializes the mask so that it corresponds to the Region of
        # Interest for creation of new keypoints
        if np.shape(ROI_NEW_FEATURES) == (2, 2):
            mask = np.zeros((_shape[0], _shape[1]), dtype=np.uint8)
            mask[ROI_NEW_FEATURES[0][0]:ROI_NEW_FEATURES[0][1],
            ROI_NEW_FEATURES[1][0]:ROI_NEW_FEATURES[1][1]] = 1
        else:
            ROI_NEW_FEATURES = [[0, _shape[0]], [0, _shape[1]]]
            mask = None

        # Sets the previous features as the new ones
        features_prev = cv.goodFeaturesToTrack(
            image=previous_images_gray[-1],  # NOQA
            maxCorners=MAX_FEATURES,
            qualityLevel=QUALITY_LVL_SHITOMASI,
            minDistance=2,
            blockSize=7,
            mask=mask
        )

    # Updates the last frame from the array
    previous_images_gray = np.roll(a=previous_images_gray, shift=-1, axis=0)
    previous_images_gray[-1] = np.copy(rs_gray)

    # Increase Counter
    _iter_count += 1

    # if pressed ESCAPE exit program
    if key == 27:
        cv.destroyAllWindows()
        break

    # Waits till next frame
    time.sleep(max(0, _t - time.time()))  # NOQA
# endregion
