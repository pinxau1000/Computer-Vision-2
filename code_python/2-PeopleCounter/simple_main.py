# Import Intel Real Sense SDK
from pyrealsense2 import pyrealsense2 as rs
# Import OpenCV for easy image rendering
from cv2 import cv2 as cv
# Import Numpy for easy array manipulation
import numpy as np
# Import CSV
import csv
# Import datetime
from datetime import datetime
# Import OS
import os


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


path = os.path.join("..", "..", "data", "CV_D435_20201104_162148.bag")

# timestamp = "_" + datetime.now().strftime('%Y%m%d_%H%M%S')
timestamp = ""
csv_path = os.path.join(f"simple_ppl_counter{timestamp}.csv")
create_csv(csv_path)

# Variables Initialization
_start = 0
n_persons = 0
count_up = 0
bot = 0
top = 0

# Creates a Real Sense Pipeline Object
pipeline = rs.pipeline(ctx=rs.context())

# Create a config object
config = rs.config()

# Tell config that we will use a recorded device from file to be used by
# the pipeline through playback (comment this line if you want to use a
# real camera).
rs.config.enable_device_from_file(config, path)

# Stream parameters
width = 848
height = 480
fps = 15

# Configure the pipeline to stream the depth image
config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
# Configure the pipeline to stream the color image
config.enable_stream(rs.stream.color, width, height, rs.format.rgb8, fps)

# Tries to start the streaming
try:
    # Start streaming from file
    profile = pipeline.start(config)
except RuntimeError as err:
    print(err)
    raise RuntimeError("Make sure the config streams exists in the device!")

colorizer = rs.colorizer(2)

cv.namedWindow("TOP - BOTTOM", cv.WINDOW_KEEPRATIO + cv.WINDOW_AUTOSIZE)
cv.namedWindow("Depth Stream", cv.WINDOW_KEEPRATIO + cv.WINDOW_AUTOSIZE)
cv.namedWindow("Color Stream", cv.WINDOW_KEEPRATIO + cv.WINDOW_AUTOSIZE)

# Streaming loop
while True:
    # Get frameset of depth
    frames = pipeline.wait_for_frames()

    # Get depth frame
    depth_frame = frames.get_depth_frame()
    depth_array = np.asanyarray(depth_frame.get_data())

    # Colorize depth frame to jet colormap
    depth_color_frame = colorizer.colorize(depth_frame)

    # Convert depth_frame to numpy array to render image in opencv
    depth_color_image = np.asanyarray(depth_color_frame.get_data())
    # depth_bw = depth_color_image[:, :, 0]
    # [480: 848]

    # Crop / Define a Region Of Interest (ROI)
    bot_depth_bw = depth_color_image[300: 350, 330: 650, 0]
    top_depth_bw = depth_color_image[150: 200, 330: 650, 0]
    bot_depth_bw = np.array(bot_depth_bw)
    top_depth_bw = np.array(top_depth_bw)

    b_depth_bw = np.ma.masked_equal(bot_depth_bw, 0)  # remove zeros
    t_depth_bw = np.ma.masked_equal(top_depth_bw, 0)  # remove zeros

    bot_mean = b_depth_bw.mean()
    top_mean = t_depth_bw.mean()

    if bot_mean > 115:
        bot = 1
        if top == 1:
            n_persons = n_persons - 1
            top = 0
            append_csv(csv_path, n_persons, 'out')
            # print(n_persons)
    else:
        if top_mean > 115:
            top = 1
            if bot == 1:
                n_persons = n_persons + 1
                bot = 0
                append_csv(csv_path, n_persons, 'in')
                # print(n_persons)
        else:
            top = 0
        bot = 0

    div = np.ones((3, top_depth_bw.shape[1]), dtype=np.uint8) * 128
    cv.imshow("TOP - BOTTOM", np.vstack((top_depth_bw,
                                         div,
                                         bot_depth_bw)))

    # Render image in opencv window
    cv.imshow("Depth Stream", depth_color_image)

    # Color frame
    color_frame = frames.get_color_frame()
    color_image = cv.cvtColor(np.asanyarray(color_frame.get_data()),
                              cv.COLOR_RGB2BGR)
    color_image = cv.putText(img=color_image,
                             text=str(n_persons),
                             org=(5, height - 5),
                             fontFace=cv.FONT_HERSHEY_SIMPLEX,
                             fontScale=1,
                             color=(255, 0, 255),
                             thickness=3)

    # Render image in opencv window
    cv.imshow("Color Stream", color_image)

    key = cv.waitKey(1)
    # if pressed escape exit program
    if key == 27:
        cv.destroyAllWindows()
        break
