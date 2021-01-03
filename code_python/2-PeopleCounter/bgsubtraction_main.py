# https://www.pyimagesearch.com/2015/05/25/basic-motion-detection-and-tracking-with-python-and-opencv/
# region Imports
# ------------------------------------------------------------------------------
#                                   Imports
# ------------------------------------------------------------------------------
import datetime
import os
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

# Create CSV file
def create_csv(f_path: str):
    with open(f_path, mode='w', newline='') as csv_f:
        current_time = datetime.now().strftime("%H:%M:%S")
        csv_file = csv.writer(csv_f)
        csv_file.writerow(
            ['# Group X', '2190383', 'Jose Rosa', '2192447', 'Ricardo Silva']
        )
        csv_file.writerow([current_time, 'none', 0])


# Write/Append Data to CSV
def append_csv(f_path: str, people_count: int, in_out: str):
    current_time = datetime.now().strftime("%H:%M:%S")
    with open(f_path, mode='a', newline='') as csv_f:
        csf_file = csv.writer(csv_f)
        csf_file.writerow([current_time, in_out, people_count])
    return current_time


def frame_handler(first_frame, current_frame):
    frameDelta = cv.absdiff(first_frame, current_frame)

    thresh = cv.threshold(frameDelta, 100, 255, cv.THRESH_BINARY)[1]

    contours, hierarchy = cv.findContours(
        thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
    )

    flag = 0
    # loop over the contours
    for c in contours:
        # if the contour is too small, ignore it
        if cv.contourArea(c) < min_area:
            continue
        # compute the bounding box for the contour, draw it on the frame,
        # and update the text
        (x, y, w, h) = cv.boundingRect(c)
        cv.rectangle(current_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        flag = 1

    return flag


# endregion


# ------------------------------------------------------------------------------
#                                       Main
# ------------------------------------------------------------------------------
path = os.path.join("..", "..", "data", "CV_D435_20201104_162148.bag")

# timestamp = "_" + datetime.now().strftime('%Y%m%d_%H%M%S')
timestamp = ""
csv_path = os.path.join(f"bgsubtraction_ppl_counter{timestamp}.csv")
create_csv(csv_path)

# Creates a Real Sense Pipeline Object
pipeline = rs.pipeline(ctx=rs.context())

# Create a config object
config = rs.config()

# Tell config that we will use a recorded device from file to be used by
# the pipeline through playback (comment this line if you want to use a
# real camera).
config.enable_device_from_file(file_name=path, repeat_playback=False)
config = rs_config_color_pipeline(config_rs=config)

try:
    # Starts the pipeline with the configuration done previously
    pipeline.start(config=config)
    profiles = pipeline.get_active_profile()
    streams = {
        "color": profiles.get_stream(rs.stream.color).as_video_stream_profile()
    }
except RuntimeError as err:
    print(err)
    raise RuntimeError("Make sure the config streams exists in the device!")

# initialize the first frame in the video stream
roi1_firstFrame = None

# minimum size (in pixels) for a region of an image to be considered actual
# “motion”
min_area = 5000

_counter = 0
time = datetime.now().strftime("%H:%M:%S")
_dummy1 = 0
_dummy2 = 0
bot = 0
top = 0

cv.namedWindow("Color Stream", cv.WINDOW_KEEPRATIO + cv.WINDOW_AUTOSIZE)
cv.namedWindow("ROI 1", cv.WINDOW_KEEPRATIO + cv.WINDOW_AUTOSIZE)
cv.namedWindow("ROI 2", cv.WINDOW_KEEPRATIO + cv.WINDOW_AUTOSIZE)

# loop over the frames of the video
while True:

    # Get frameset of depth
    frames = pipeline.wait_for_frames()

    # Color frame
    color_frame = frames.get_color_frame()
    color_image = cv.cvtColor(np.asanyarray(color_frame.get_data()),
                              cv.COLOR_RGB2BGR)

    # if the frame could not be grabbed, then we have reached the end of the
    # video
    if color_image is None:
        break

    # Render image in opencv window
    # cv.imshow("Color Stream", color_image)

    gray = cv.cvtColor(color_image, cv.COLOR_BGR2GRAY)
    gray = cv.GaussianBlur(gray, (31, 31), 0)

    # grab the current frame and initialize the occupied/unoccupied text
    roi1_frame = gray[300:400, 330:630]
    roi2_frame = gray[100:200, 330:630]

    if roi1_firstFrame is None:
        roi1_firstFrame = roi1_frame
        roi2_firstFrame = roi2_frame

    _dummy1 = frame_handler(roi1_firstFrame, roi1_frame)
    _dummy2 = frame_handler(roi2_firstFrame, roi2_frame)  # NOQA

    if _dummy1 == 1:
        bot = 1
        if top == 1:
            _counter -= 1
            top = 0
            time = append_csv(csv_path, _counter, 'out')
    else:
        if _dummy2 == 1:
            top = 1
            if bot == 1:
                _counter += 1
                bot = 0
                time = append_csv(csv_path, _counter, 'in')
        else:
            top = 0
        bot = 0
    text = str(_counter)

    # draw the text and timestamp on the frame
    cv.putText(img=color_image,
               text=f"Persons inside: {_counter}",
               org=(10, 20),
               fontFace=cv.FONT_HERSHEY_SIMPLEX,
               fontScale=0.5,
               color=(0, 0, 255),
               thickness=2)
    cv.putText(img=color_image,
               text=f"Latest regisstry: {time}",
               org=(10, color_image.shape[0] - 10),
               fontFace=cv.FONT_HERSHEY_SIMPLEX,
               fontScale=0.5,
               color=(0, 0, 255),
               thickness=1)

    # Render image in opencv window
    cv.imshow("Color Stream", color_image)
    cv.imshow("ROI 1", roi1_frame)
    cv.imshow("ROI 2", roi2_frame)

    key = cv.waitKey(1) & 0xFF
    # if the 'Esc' key is pressed, break from the lop
    if key == 27:
        break
# cleanup the camera and close any open windows
cv.destroyAllWindows()
