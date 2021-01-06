# region Imports
# ------------------------------------------------------------------------------
#                                   Imports
# ------------------------------------------------------------------------------
import os
from typing import Union
import csv
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


# endregion


# region Image Processing Functions
# ------------------------------------------------------------------------------
#                           Image Processing Functions
# ------------------------------------------------------------------------------
def get_keypoints_and_descriptors(imageL: np.ndarray, imageR: np.ndarray,
                                  feature_desc: cv.Feature2D = None):
    """
    Computes the keypoints and descriptors of 2 images.

    example:
    kp_l, desc_l, kp_r, desc_r = get_keypoints_and_descriptors(image_l,
                                                               image_r[,
                                                               descriptor])

    @param imageL: Left image
    @type imageL: np.ndarray
    @param imageR: Right image
    @type imageR: np.ndarray
    @param feature_desc: The feature descriptor. Can be ORB, SIFT, SURF,
    etc. The default is SIFT is None is passed
    @type feature_desc: cv.Feature2D
    @return: Keypoints and descriptor of left image and keypoints and
    descriptor of right image
    @rtype: list, np.ndarray, list, np.ndarray
    """
    if feature_desc is None:
        # Use SIFT Feature Descriptor to detect interest points as default
        feature_desc = cv.SIFT_create()

    assert isinstance(feature_desc, cv.Feature2D), \
        "A object that inherits cv.Feature2D must be passed!"

    # find the keypoints and descriptors
    _kp1, _desc1 = feature_desc.detectAndCompute(image=imageL, mask=None)
    _kp2, _desc2 = feature_desc.detectAndCompute(image=imageR, mask=None)

    return _kp1, _desc1, _kp2, _desc2


def get_matching_points(descriptorL: np.ndarray,
                        descriptorR: np.ndarray,
                        desc_matcher: cv.DescriptorMatcher = None,
                        k: int = 2):
    """
    Computes the matches between keypoints. The matches are returned in the
    distance increasing order.

    example:
    matches = get_matching_points(desc_l, desc_r[, matcher])

    The matcher object must be FLANN or BFMatcher Type! See the examples below:
    ----------------------------------------------------------------------------
    Example of FLANN:
    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)   # or pass empty dictionary

    flann = cv2.FlannBasedMatcher(index_params,search_params)
    ----------------------------------------------------------------------------

    ----------------------------------------------------------------------------
    Example of Brute Force:
    cv.BFMatcher_create(normType=cv.NORM_L2)
    ----------------------------------------------------------------------------

    @param descriptorL: The left image keypoints descriptors
    @type descriptorL: np.ndarray
    @param descriptorR: The right image keypoints descriptors
    @type descriptorR: np.ndarray
    @param desc_matcher: A DescriptorMatcher object for fine tuning. This
    can be either FLANN or BFMatcher.
    @type desc_matcher: cv.DescriptorMatcher
    @param k: Number of matches to return
    @type k: int
    @return: A list of matches
    @rtype: list
    """
    if desc_matcher is None:
        # Brute Force Matcher with default params
        desc_matcher = cv.BFMatcher_create(normType=cv.NORM_L2)

    assert isinstance(desc_matcher, cv.DescriptorMatcher), \
        "A object that inherits cv.DescriptorMatcher must be passed!"

    # Finds the k best matches for each descriptor from a query set.
    return desc_matcher.knnMatch(queryDescriptors=descriptorL,
                                 trainDescriptors=descriptorR,
                                 k=k)


def lowe_ratio_test(matches_list: list, keypointsL: list, keypointsR: list,
                    K: float = 0.8, best_N: Union[int, float] = None):
    """
    Implements David Lowe ratio test.

    example:
    matches_l, matches_r, matches_l2r, matches_r2l = lowe_ratio_test(
                                                    matches_list=matches_all,
                                                    keypointsL=kp1,
                                                    keypointsR=kp2,
                                                    K=0.2,
                                                    best_N=8)

    @param matches_list: The matches list
    @type matches_list: list
    @param keypointsL: keypoints of the left image
    @type keypointsL: list
    @param keypointsR: keypoints of the right image
    @type keypointsR: list
    @param K: Lowe's ratio.
    @type K: float
    @param best_N: Number of best matches to return. If int returns the N
    best matches, if float returns the N * matches results, if None returns
    all matches.
    @type best_N: Union[int, float]
    @return: matchesL, matchesR, matches1to2, matches2to1
    @rtype: list, list, list, list
    """
    assert type(best_N) is int or type(best_N) is float or best_N is None, \
        "Best N should be int, float or None!"

    # Initiate the array to store the values that mean the ratio test
    # requirement
    _matchesL = []
    _matchesR = []
    _matches1to2 = []
    _matches2to1 = []

    # Ratio test as described Lowe's paper
    for _match1, _match2 in matches_list:
        # If the (distance of the closest match)/(distance of the 2nd closest
        # match) is below K then we consider the closest point as an
        # unambiguous good (good = close) match.
        if _match1.distance < K * _match2.distance:
            # Access to the index of the match of the train set, which
            # corresponds to right image, and add that point to the match
            # points.
            _matchesL.append(keypointsR[_match1.trainIdx].pt)

            # Access to the index of the match of the query set, which
            # corresponds to image L, and add that point to the match points.
            _matchesR.append(keypointsL[_match1.queryIdx].pt)

            # Same thing but instead of appending the individual keypoints of
            # the images we append all the structure. Useful when plotting.
            _matches1to2.append(_match1)
            _matches2to1.append(_match2)

            # Collect only the best N matches. This part is fundamental. If
            # is not present we may encounter errors!
            if (type(best_N) is int) and (len(_matchesL) >= best_N):
                break

    if type(best_N) is float:
        best_N = int(np.rint(best_N * len(_matchesL)))

        return _matchesL[:best_N], _matchesR[:best_N], \
               _matches1to2[:best_N], _matches2to1[:best_N]

    if type(best_N) is int:
        assert len(_matchesL) >= best_N, \
            f"Unable to find {best_N} matches! Try decreasing this value or " \
            f"increase the Lowe Ratio."

    return _matchesL, _matchesR, _matches1to2, _matches2to1


def get_fundamental_matrix(matchesL: list, matchesR: list,
                           search_method: int = cv.FM_RANSAC):
    """
    Gets the fundamental matrix from a given matches points.

    example:
    fund_mat, inliers_l, inliers_r = get_fundamental_matrix(matchesL=matches_l,
                                                            matchesR=matches_r)

    @param matchesL: The matches from the left image
    @type matchesL: list
    @param matchesR: The matches form the right image
    @type matchesR: list
    @param search_method: The search method to compute the fundamental matrix
    @type search_method: int
    @return: fundamental_mat, inliersL, inliersR
    @rtype: np.ndarray, np.ndarray, np.ndarray
    """
    # RANSAC is the type of the algorithm for finding fundamental matrix.
    # Can be cv.FM_7POINT, cv.FM_8POINT, cv.FM_LMEDS and cv.FM_RANSAC
    # https://docs.opencv.org/master/d9/d0c/group__calib3d.html
    # #gae850fad056e407befb9e2db04dd9e509
    # https://docs.opencv.org/3.4/d9/d0c/group__calib3d.html
    # #ga04167220173d36be64ec6ddfca21bf18
    fundamental_mat, mask = cv.findFundamentalMat(points1=np.float64(matchesL),
                                                  points2=np.float64(matchesR),
                                                  method=search_method,
                                                  ransacReprojThreshold=3,
                                                  confidence=0.99,
                                                  maxIters=2000)  # Default

    # Ravel flattens the mask array. We select only the inliers (where the
    # mask is 1) as matching points!
    inliersL = np.float64(matchesL)[mask.ravel() == 1]
    inliersR = np.float64(matchesR)[mask.ravel() == 1]

    return fundamental_mat, inliersL, inliersR


def get_homography(match_pts1: list, match_pts2: list,
                   fundamental_mat: np.ndarray, size: tuple = None,
                   img: np.ndarray = None):
    """
    Computes the homography matrix for image 1 and for image 2. Size or Image
    should be passed.

    example:
    H1, H2 = get_homography(match_pts1, match_pts2, fundamental_mat, img)

    @param match_pts1: The match point of the first image
    @type match_pts1: list
    @param match_pts2: The match point of the second image
    @type match_pts2: list
    @param fundamental_mat: The fundamental matrix
    @type fundamental_mat: np.ndarray
    @param size: A tuple with size like (width, height)
    @type size: tuple
    @param img: An image to extract the size
    @type img: np.ndarray
    @return: Homography matrix for image 1 and Homography matrix for image 2
    @rtype: np.ndarray, np.ndarray
    """
    # HOMOGRAPHY: Correspondences between points in two different images from
    # the same scene.
    assert (size is not None) or (img is not None), \
        "Size or Image must be passed!"

    # Prevents warnings
    _size = (np.NaN, np.NaN)

    # size parameter always overwrites image size!
    if img is not None:
        # Size(double width, double height)
        height, width = np.shape(img)
        _size = width, height

    # size parameter always overwrites image size!
    if size is not None:
        _size = size

    # Compute output rectification homography matrix for the first image and
    # second image and return H1 and H2.
    return cv.stereoRectifyUncalibrated(points1=np.float32(match_pts1),
                                        points2=np.float32(match_pts2),
                                        F=fundamental_mat,
                                        imgSize=_size,
                                        threshold=3)[1:3]


def get_rectified(img: np.ndarray, M_mat: np.ndarray,
                  size: Union[tuple, int, float] = 1) -> np.ndarray:
    """
    Applies a perspective transformation to an image. The function
    warpPerspective transforms the source image using the specified matrix.

    example:
    rectified = get_rectified(img=image, M_mat=H2to1)

    @param img: Image to apply the perspective transformation
    @type img: np.ndarray
    @param M_mat: The transform matrix to apply to the image
    @type M_mat: np.ndarray
    @param size: The size of the output image. If not passed is equal to the
    image size. Can be set to int or float to be a factor of the image size
    @type size: tuple, int or float
    @return: Transformed/rectified image
    @rtype: np.ndarray
    """

    if type(size) is int or type(size) is float:
        # Size(double width, double height)
        height, width = np.shape(img)
        size = np.rint(size * width).astype(np.int), np.round(
            size * height).astype(np.int)

    return cv.warpPerspective(src=img, M=M_mat, dsize=size)


def get_disparity_map(imageL: np.ndarray, imageR: np.ndarray,
                      disparity_matcher: cv.StereoMatcher = None,
                      disparity_filter: cv.ximgproc_DisparityFilter = None):
    """
    Returns the disparity map based on image 1 and image 2.

    example:
    disparity = get_disparity_map(imageL=left_ir, imageR=right_rect)

    Based on https://docs.opencv.org/master/dd/d53/tutorial_py_depthmap.html

    @param imageL: first stereo image to calculate the depth
    @type imageL: np.ndarray
    @param imageR: second stereo image to calculate the depth
    @type imageR: np.ndarray
    @param disparity_matcher: A StereoMatcher object to implement the stereo
    correspondence algorithm. Available algorithms are: cv.StereoBM and
    cv.StereoSGBM. Default is cv.StereoBM_create(numDisparities=64, blockSize=9)
    @type disparity_matcher: cv.StereoMatcher
    @param disparity_filter: An instance of a disparity filter
    @type disparity_filter: cv.ximgproc.DisparityFilter
    @return: the disparity map of the stereo image pair provided by
    cv.StereoMatcher.compute method.
    @rtype: np.ndarray
    """
    assert isinstance(disparity_filter, cv.ximgproc_DisparityFilter) or \
           disparity_filter is None, \
        "Disparity Filter must be an instance of " \
        "cv.ximgproc_DisparityFilter or None!"

    if disparity_matcher is None:
        # Default disparity matcher if None is passed
        disparity_matcher = cv.StereoBM_create(numDisparities=64, blockSize=9)

    assert isinstance(disparity_matcher, cv.StereoMatcher), \
        "Matcher object must be an instance of cv.StereoMatcher!"

    # computes the disparity based on left image and right image
    disparity_imgL = disparity_matcher.compute(left=imageL, right=imageR)

    if isinstance(disparity_filter, cv.ximgproc_DisparityFilter):
        # Creates the right matcher from the left matcher
        disparity_matcher_R = \
            cv.ximgproc.createRightMatcher(disparity_matcher)

        # Computes disparity from right to left
        disparity_imgR = disparity_matcher_R.compute(left=imageR,
                                                     right=imageL)

        # Filters and returns disparity
        return disparity_filter.filter(disparity_map_left=disparity_imgL,
                                       left_view=imageL,
                                       disparity_map_right=disparity_imgR,
                                       right_view=imageR)

    return disparity_imgL


# endregion


# region Utility Functions
# ------------------------------------------------------------------------------
#                               Utility Functions
# ------------------------------------------------------------------------------
def create_csv(f_path):
    _now = datetime.now()
    with open(f_path, mode='w', newline='') as csv_f:
        csv_file = csv.writer(csv_f)
        csv_file.writerow(
            ['# Group X', '2190383', 'Jose Rosa', '2192447', 'Ricardo Silva']
        )


def write2csv(f_path, intrinsic, rotation, translation):
    _now = datetime.now()
    with open(f_path, mode='a', newline='') as csv_f:
        csv_file = csv.writer(csv_f)
        csv_file.writerow([_now.strftime("%H:%M:%S"),
                           intrinsic, rotation, translation])


def reject_outliers_1(data: np.ndarray, m: float = 2.):
    """
    Sets the outliers to 0 on an numpy array. Based on:
    https://stackoverflow.com/questions/11686720/is-there-a-numpy-builtin-to-reject-outliers-from-a-list
    """
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d / mdev if mdev else 0.
    return data[s >= m]


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
csv_path = os.path.join(f"intrinsics_extrinsics{timestamp}.csv")
create_csv(csv_path)

# Number of corners required to compute the calibration matrix
_MIN_CORNERS = 10


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

# Obtain the depth scale
rs_depth_scale = get_depth_scale(pipeline_rs=pipeline)
print("Depth Scale is [m/px_val]: ", rs_depth_scale)

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

# Tells user how to proceed
for _ in range(10):
    print("PRESS SPACE TO CAPTURE A FRAME FOR CALIBRATION!")

# FLAG (Don't touch)
first_run = True

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

    # Render image in opencv window
    cv.imshow("Color Stream", np.uint8(np.vstack((rs_color_rgb, image_bar))))

    # If SPACE is pressed
    if key == 32:
        # Find the chessboard inner corners
        ret_val, corners = cv.findChessboardCorners(image=rs_color_gray,
                                                    patternSize=chess_size)

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
            if len(image_points) > _MIN_CORNERS:
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

                write2csv(csv_path, cam_mat, rot_mat, trans_vec_avg)

                mean_error = 0
                for i in range(len(obj_points)):
                    imgpoints2, _ = cv.projectPoints(obj_points[i],
                                                     rot_vec[i],
                                                     trans_vec[i],
                                                     cam_mat,
                                                     dist_coef)
                    error = cv.norm(image_points[i],
                                    imgpoints2,
                                    cv.NORM_L2) / \
                                    len(imgpoints2)
                    mean_error += error

                print(mean_error/len(obj_points))



    # if pressed ESCAPE exit program
    if key == 27:
        cv.destroyAllWindows()
        break
