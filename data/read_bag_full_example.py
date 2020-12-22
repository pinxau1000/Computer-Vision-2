#####################################################
##               Read bag from file                ##
#####################################################


# First import library
import pyrealsense2 as rs
# Import Numpy for easy array manipulation
import numpy as np
# Import OpenCV for easy image rendering
import cv2
# Import argparse for command-line options
import argparse
# Import os.path for file path manipulation
import os.path

"""
Returns R, T transform from src to dst
"""
def get_extrinsics(src, dst):
    extrinsics = src.get_extrinsics_to(dst)
    R = np.reshape(extrinsics.rotation, [3,3]).T
    T = np.array(extrinsics.translation)
    return (R, T)

# Create object for parsing command-line options
parser = argparse.ArgumentParser(description="Read recorded bag file and display depth stream in jet colormap.\
                                Remember to change the stream resolution, fps and format to match the recorded.")
# Add argument which takes path to a bag file as an input
parser.add_argument("-i", "--input", type=str, help="Path to the bag file")
# Parse the command line arguments to an object
args = parser.parse_args()
# Safety if no parameter have been given
if not args.input:
    print("No input paramater have been given.")
    print("For help type --help")
    exit()
# Check if the given file have bag extension
if os.path.splitext(args.input)[1] != ".bag":
    print("The given file is not of correct file format.")
    print("Only .bag files are accepted")
    exit()
try:
    # Create pipeline
    pipeline = rs.pipeline()

    # Create a config object
    config = rs.config()
    # Tell config that we will use a recorded device from filem to be used by
    # the pipeline through playback (comment this line if you want to use a
    # real camera).
    rs.config.enable_device_from_file(config, args.input)
    width = 848
    height = 480
    fps = 15
    # Configure the pipeline to stream the depth image
    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
    # Configure the pipeline to stream both infrared images
    config.enable_stream(rs.stream.infrared, 1, width, height, rs.format.y8, fps)
    config.enable_stream(rs.stream.infrared, 2, width, height, rs.format.y8, fps)
    # Configure the pipeline to stream the color image
    config.enable_stream(rs.stream.color, width, height, rs.format.rgb8, fps)

    # Start streaming from file
    profile = pipeline.start(config)

    # Create opencv window to render image in
    # Uncomment if you want to be able to resize the window. The KEEPRATIO
    #parameters might not work.
    # cv2.namedWindow("Depth Stream", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    # cv2.namedWindow("Left IR Stream", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    # cv2.namedWindow("Right IR Stream", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    # cv2.namedWindow("Color Stream", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO )
    
    # Create colorizer object
    colorizer = rs.colorizer()

    # Retreive the stream and intrinsic properties for all cameras
    profiles = pipeline.get_active_profile()
    streams = {"color" : profiles.get_stream(rs.stream.color).as_video_stream_profile(),
               "left"  : profiles.get_stream(rs.stream.infrared, 1).as_video_stream_profile(),
               "right" : profiles.get_stream(rs.stream.infrared, 2).as_video_stream_profile(),
               "depth" : profiles.get_stream(rs.stream.depth).as_video_stream_profile()
              }
    intrinsics = {"color" : streams["color"].get_intrinsics(),
                  "left"  : streams["left"].get_intrinsics(),
                  "right" : streams["right"].get_intrinsics(),
                  "depth" : streams["depth"].get_intrinsics(),
                 }
    extrinsics = {"color2left" : get_extrinsics(streams["color"], streams["left"]),
                  "right2left" : get_extrinsics(streams["right"], streams["left"])}
    # We can retrieve the extrinsic parameters from the camera itself, but
    # since we are using a bagfile, force the baseline. We are assuming the cameras
    # to be parallel and simply displaced along the X coordinate by the baseine
    # distance (in meters). 
    baseline = 0.005  # extrinsics["right2left"][1][0]

    # Obtain the depth scale
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print("Depth Scale is [m/px_val]: " , depth_scale)

    # Streaming loop
    while True:
        # Get frameset of depth
        frames = pipeline.wait_for_frames()

        # Get depth frame
        depth_frame = frames.get_depth_frame()

        # Colorize depth frame to jet colormap
        depth_color_frame = colorizer.colorize(depth_frame)

        # Convert depth_frame to numpy array to render image in opencv
        depth_color_image = np.asanyarray(depth_color_frame.get_data())

        # Render image in opencv window
        cv2.imshow("Depth Stream", depth_color_image)

        # Left IR frame
        ir_left_frame = frames.get_infrared_frame(1)
        ir_left_image = np.asanyarray(ir_left_frame.get_data())
        # Render image in opencv window
        cv2.imshow("Left IR Stream", ir_left_image)

        # Right IR frame
        ir_right_frame = frames.get_infrared_frame(2)
        ir_right_image = np.asanyarray(ir_right_frame.get_data())
        # Render image in opencv window
        cv2.imshow("Right IR Stream", ir_right_image)

        # Color frame
        color_frame = frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())
        # Render image in opencv window
        cv2.imshow("Color Stream", cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR))

        key = cv2.waitKey(1)
        # if pressed escape exit program
        if key == 27:
            cv2.destroyAllWindows()
            break

finally:
    pass
