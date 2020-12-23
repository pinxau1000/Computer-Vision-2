# %% ---------------------------------------------------------------------------
#                                   Imports
# ------------------------------------------------------------------------------
import os
from typing import Union

from pyrealsense2 import pyrealsense2 as rs
from cv2 import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


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


# %% ---------------------------------------------------------------------------
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
                           search_method: int = cv.RANSAC):
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
                                                  confidence=0.99)

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

    # return H1 and H2
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
                      disparity_filter: cv.ximgproc_DisparityFilter = None,
                      enhance_filtering: bool = False):
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
    @param enhance_filtering: Enhance the filtering process by taking into
    account the second image.
    @type enhance_filtering: bool
    @return: the disparity map of the stereo image pair provided by
    cv.StereoMatcher.compute method.
    @rtype: np.ndarray
    """
    assert isinstance(disparity_filter, cv.ximgproc_DisparityFilter) or \
           disparity_filter is None, \
        "Disparity Filter must be an instance of cv.ximgproc_DisparityFilter!"

    if disparity_matcher is None:
        disparity_matcher = cv.StereoBM_create(numDisparities=64, blockSize=9)

    assert isinstance(disparity_matcher, cv.StereoMatcher), \
        "Matcher object must be an instance of cv.StereoMatcher!"

    matcher_img2 = cv.ximgproc.createRightMatcher(disparity_matcher)

    disparity_img1 = disparity_matcher.compute(left=imageL, right=imageR)

    if type(disparity_filter) is cv.ximgproc_DisparityFilter:
        if enhance_filtering:
            disparity_img2 = matcher_img2.compute(left=imageR, right=imageL)
            return disparity_filter.filter(disparity_map_left=disparity_img1,
                                           left_view=imageL,
                                           disparity_map_right=disparity_img2,
                                           right_view=imageR)
        else:
            return disparity_filter.filter(disparity_map_left=disparity_img1,
                                           left_view=imageL)

    return disparity_img1


# %% ---------------------------------------------------------------------------
#                               Utility Functions
# ------------------------------------------------------------------------------
def load_image(file_path: str = None, colorConvCode: int = None):
    """
    Loads an image with the path fpath. If color code is provided a color
    conversion is applied.

    example:
    image = load_image(file_path, cv.COLOR_RGB2BGR)

    @param file_path: Path of image to load
    @type file_path: str
    @param colorConvCode: Type of conversion of code to apply to image. None
    if no conversion is applied
    @type colorConvCode: int (based on enum cv.ColorConversionCodes)
    @return: image as numpy array
    @rtype: np.ndarray
    """

    assert file_path is not None, "path must not be None"
    assert os.path.exists(file_path), f"{file_path} didn't exist"

    _img = cv.imread(file_path)

    if colorConvCode is None:
        return _img
    else:
        return _img, cv.cvtColor(_img, colorConvCode)


def avg_std_welford(sample: np.ndarray, _N: int,
                    last_avg: np.ndarray, last_M2: np.ndarray):
    """
    Computes the average and standard deviation N samples, including the
    actual sample.

    @param sample: The current sample to be evaluated
    @type sample: np.ndarray
    @param _N: The number of current sample
    @type _N: int
    @param last_avg: last average
    @type last_avg: np.ndarray
    @param last_M2: last M2 param
    @type last_M2: np.ndarray
    @return: average, standard deviation, M2 and N
    @rtype: np.ndarray, np.ndarray, np.ndarray, int
    """

    _avg = last_avg + (sample - last_avg) / _N
    _M2 = last_M2 + (sample - last_avg) * (sample - _avg)
    _std = _M2 / _N

    return _avg, _std, _M2, _N


def compute_error(orig: np.ndarray, test: np.ndarray, _N: int,
                  _avg: np.ndarray, _M2: np.ndarray):
    """
    Computes the error between the original and test array using the absolute
    difference between pixels. Welfoard algoritm is used.

    @param orig: Original samples
    @type orig: np.ndarray
    @param test: Test samples
    @type test: np.ndarray
    @param _N: N iteration
    @type _N: int
    @param _avg: last average
    @type _avg: np.ndarray
    @param _M2: last M2
    @type _M2: np.ndarray
    @return: np.ndarray, np.ndarray, np.ndarray
    @rtype: average, standard deviation, M2
    """
    # Creates a mark with the same shape of original image/array
    _mask = np.zeros_like(orig)

    # Sets the mask to 1 only when orig and test are bigger than 0
    _abs_dif = np.zeros_like(orig)
    np.putmask(_abs_dif, (orig > 0) & (test > 0), np.abs(orig - test))

    # Applies Welford algoritm to compute avg and std (M2 is also returned)
    return avg_std_welford(sample=_abs_dif, _N=_N,
                           last_avg=_avg,
                           last_M2=_M2)[:3]


def zero_outliers(data: np.ndarray, m: float = 2.):
    """
    Sets the outliers to 0 on an numpy array. Based on:
    https://stackoverflow.com/questions/11686720/is-there-a-numpy-builtin-to-reject-outliers-from-a-list
    """
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d / mdev if mdev else 0.
    data[s >= m] = 0
    return data


def draw_error(current: np.ndarray, data: list,
               wh: int, sc: float, _data_max: float, _data_min: float):
    """
    Draws an "oscilloscope" style graph with the data. Because the graph is
    active the current "image" needs to be passed so the next one can be
    generated.

    example:
    # Draws a graph showing the average and std
    error_graph, _error_graph_max, error_graph_min = draw_error(
        current=error_graph,
        data=[global_avg, global_std],
        _data_max=_error_graph_max,
        _data_min=_error_graph_min,
        wh=win_h,
        sc=scale
    )

    @param current: Current image. initialize it as a null array and then
    update it with the return of this function.
    @type current: np.ndarray
    @param data: The data to be plotted. Always pass a list, even with only
    one element.
    @type data: list
    @param wh: The graph window height or the height of the current image.
    @type wh: int
    @param sc: A scale factor to apply when the data to be shows is too
    small. Keep in mind that the data will be rounded and assigned to a row
    index based on it's value.
    @type sc: float
    @param _data_max: Maximum value of the data. Needed for updating the window
    if the data goes above the window height. This value should be
    initialized as 0 and updated on every iteration.
    @type _data_max: float
    @param _data_min: Minimum value of the data. Needed for updating the window
    if the data goes below the window height. This value should be
    initialized as 0 and updated on every iteration.
    @type _data_min: float
    @return: The image to be show (cv.imshow), the new data_max and data_min
    @rtype: np.ndarray, float, float
    """
    COLORS = [
        (255, 0, 255),
        (0, 255, 255),
        (0, 0, 255),
        (0, 255, 0),
        (255, 0, 0),
        (255, 0, 255),
        (0, 255, 255),
        (255, 255, 0),
    ]

    # Shift the data on the error window
    _graph = np.roll(current, -1, axis=1)

    if np.max(data) > _data_max:
        _data_max = np.max(data)
        _shift = np.ceil(np.abs(_data_max - wh)).astype(np.int)
        _graph = np.roll(_graph, _shift, axis=0)
    if np.min(data) < _data_min:
        _data_min = np.min(data)
        _shift = np.ceil(np.abs(_data_min - wh)).astype(np.int)
        _graph = np.roll(_graph, -_shift, axis=0)

    # Sets the last column to zero to store new data
    _graph[:, -1] = 0

    # Stores new data. The index of the row is proportional to the value of
    # the data. Color is BGR so: AVG is Pink and STD is Yellow
    for i, d in enumerate(data):
        _graph[wh - 1 - np.rint(d * sc).astype(np.int), -1] = COLORS[i]

    return _graph, _error_graph_max, _error_graph_min


# ------------------------------------------------------------------------------
#                                       Main
# ------------------------------------------------------------------------------
FULL_CALIB = os.path.join("..", "..", "data",
                          "CV_D435_20201104_161043_Full_calibration.bag")
RGB_CALIB = os.path.join("..", "..", "data",
                         "CV_D435_20201104_160738_RGB_calibration.bag")
STREAM = os.path.join("..", "..", "data",
                      "CV_D435_20201104_162148.bag")

path = STREAM

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

# Create colorizer object to apply to depth frames
colorizer = rs.colorizer()

# Get intrinsics and extrinsics parameters from the multiple profiles of
# the pipeline
intrinsics, extrinsics = get_intrinsics_extrinsics(pipeline_rs=pipeline)

# Obtain the depth scale
rs_depth_scale = get_depth_scale(pipeline_rs=pipeline)
print("Depth Scale is [m/px_val]: ", rs_depth_scale)

# Distance between left and right IR cameras in meters. Cameras are
# assumed to be parallel to each other. We are assuming no distortion for
# all cameras
baseline = 0.05  # m

# Creates windows to display the frames
cv.namedWindow("D435 Depth Stream", cv.WINDOW_AUTOSIZE)
cv.namedWindow("My Depth Map", cv.WINDOW_AUTOSIZE)

# Instantiation of a stereo matcher object. We try to refine the parameters
# to achieve best results.
_min_disp = -8
_max_disp = 8
# Number of different disparities (Like Quantization)
_num_disp = np.int32(16 * np.round(_max_disp - _min_disp))
# Block size 9 because we don't need the details
matcher = cv.StereoSGBM_create(minDisparity=_min_disp,
                               numDisparities=_num_disp,
                               blockSize=5)

# Instantiation of a Disparity Filter Object
disp_filter = cv.ximgproc.createDisparityWLSFilter(matcher)

# FLAG to enable the calculation of transform matrix on the first run
first_run = True
H1 = np.ones((3, 3))  # Prevent warning
H2 = np.ones((3, 3))  # Prevent warning

# Error metrics variables (Welford)
count = 1
avg = 0
M2 = 0

# Creates the array to displays the avg error and std
win_w = 200
win_h = 200
scale = 200.
error_graph = np.zeros((win_h, win_w, 3))

# Creates the window to display the error and std
cv.namedWindow("ERROR - AVG & STD", cv.WINDOW_AUTOSIZE + cv.WINDOW_KEEPRATIO)

# Don't touch! These vars will be updated as the program run we are just
# initialing it. These defines the current maximum and minimum of the data to
# be show.
_error_graph_max = 0
_error_graph_min = 0

# Main cycle/loop
while True:
    # Wait for new frames and grabs the frameset
    frameset = pipeline.wait_for_frames()

    # https://intelrealsense.github.io/librealsense/doxygen
    # /classrs2_1_1frameset.html
    # Get IR Camera frames
    left_ir = np.asanyarray(frameset.get_infrared_frame(1).get_data())
    right_ir = np.asanyarray(frameset.get_infrared_frame(2).get_data())

    # Render image in opencv window
    # cv.imshow("IR Stream", np.hstack((left_ir, right_ir)))

    # Get Depth Frames with Color
    rs_depth_color = np.asanyarray(
        colorizer.colorize(frameset.get_depth_frame()).get_data())
    # Get Depth Frames without Color (Used for distance calculation)
    rs_depth = np.asanyarray(frameset.get_depth_frame().get_data())

    # Render image in opencv window
    cv.imshow("D435 Depth Stream", rs_depth_color)

    if first_run:
        # Get the descriptors and the keypoints using SIFT
        kp1, desc1, \
        kp2, desc2 = get_keypoints_and_descriptors(imageL=left_ir,
                                                   imageR=right_ir,
                                                   feature_desc=cv.SIFT_create())

        # Compute the matching keypoints based on their descriptors
        matches_all = get_matching_points(descriptorL=desc1,
                                          descriptorR=desc2)

        # Apply Lowe's test to ensure valid matches only
        matches_l, matches_r, matches_l2r, matches_r2l = lowe_ratio_test(
            matches_list=matches_all,
            keypointsL=kp1,
            keypointsR=kp2,
            K=0.6)
            # , best_N=0.95)

        """
        # Generate and show the matches
        matching_image = np.hstack((left_ir, right_ir))
        matching_image = cv.drawMatches(img1=left_ir, keypoints1=kp1,
         img2=right_ir,
                                        keypoints2=kp2, matches1to2=matches_l2r,
                                        outImg=matching_image, flags=2)
    
        cv.imshow("Matching", matching_image)
        # """

        # Compute the fundamental matrix
        fund_mat, inliers_l, inliers_r = get_fundamental_matrix(
            matchesL=matches_l,
            matchesR=matches_r)

        """
        # Find homography matrix/transform matrix with image left as reference
        H_2to1, _ = cv.findHomography(np.float64(inliers_r),
                                      np.float64(inliers_l),
                                      method=cv.RANSAC,
                                      ransacReprojThreshold=3, confidence=0.99)
    
        # Rectify the right image
        right_rect = get_rectified(img=right_ir, M_mat=H_2to1)
        # """

        # Transform matrix to virtual common plane
        H1, H2 = get_homography(match_pts1=inliers_l,
                                match_pts2=inliers_r,
                                fundamental_mat=fund_mat,
                                img=left_ir)

        # Keep the H1 and H2 from the first run for the next frames.
        first_run = False

    # Rectify the left image
    left_rect = get_rectified(img=left_ir, M_mat=H1)

    # Rectify the right image
    right_rect = get_rectified(img=right_ir, M_mat=H2)

    # Show rectified images
    # cv.imshow("Rectified", np.hstack((left_rect, right_rect)))

    # Get the disparity map from the left and rectified right image
    disparity = get_disparity_map(imageL=left_rect,
                                  imageR=right_rect,
                                  disparity_matcher=matcher,
                                  disparity_filter=disp_filter,
                                  enhance_filtering=True)

    # Compute the depth from disparity, focal length(pixels) and baseline(m)
    depth = np.zeros_like(disparity).astype(np.float64)
    depth[disparity > 0] = (intrinsics.get('Infrared 1').fx * baseline) / \
                           (0.1 * disparity[disparity > 0])

    """
    # DEBUG (USED TO ESTIMATE A _THRESHOLD_)
    _MAX = 1.5  # in m
    depth[depth > _MAX] = _MAX

    plt.boxplot(depth.ravel())
    plt.show()
    
    # Implemented an alternative, see below, zero_outliers()
    _THRESHOLD_ = 0.25
    depth[depth > np.max(depth) * _THRESHOLD_] = np.max(depth) * _THRESHOLD_
    # """

    # Remove outliers
    depth = zero_outliers(data=depth, m=6)

    # Show Depth Map (GRAY SCALE)
    # cv.imshow("My Depth Original", depth)

    # Remaps the depth values to match a 255 color image.
    depth_remap = np.interp(x=depth,
                            xp=(0, np.max(depth)),
                            fp=(255, 0)).astype(np.uint8)

    # Use median blur filtering to smooth the image (For display proposes only)
    depth_filtered = cv.medianBlur(depth_remap, 5)

    # Apply color map to the depth image
    depth_color = cv.applyColorMap(src=np.uint8(depth_filtered),
                                   colormap=cv.COLORMAP_JET)

    # Show the depth image
    cv.imshow("My Depth Map", depth_color)

    # Computes AVG and STD for quality metrics
    rs_depth_scaled = rs_depth * rs_depth_scale
    avg, std, M2 = compute_error(
        orig=rs_depth_scaled,
        test=depth,
        _N=count,
        _avg=avg,
        _M2=M2
    )

    # Gets average of all pixels
    global_avg = np.average(avg)

    # Gets the std of all pixels
    global_std = np.average(std)

    # Prints the AVG of the absolute difference of all pixels and the STD
    print(f"AVG: {str(global_avg)};\tSTD: {str(global_std)}")

    # Draws a graph showing the average and std
    error_graph, _error_graph_max, error_graph_min = draw_error(
        current=error_graph,
        data=[global_avg, global_std],
        _data_max=_error_graph_max,
        _data_min=_error_graph_min,
        wh=win_h,
        sc=scale
    )
    cv.imshow("ERROR - AVG & STD", error_graph)

    # Increment (Used for Welford metrics)
    count += 1

    # Read key and waits 1ms
    key = cv.waitKey(1)
    # if pressed ESCAPE exit program
    if key == 27:
        cv.destroyAllWindows()
        break
