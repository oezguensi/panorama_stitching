from cv2 import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine
from tqdm import tqdm


def prepare_img(img, roi=None, downscale_factor=4, kernel_size=3):
    """
    Prepare image for keypoint matching
    :param img: Image in BGR format
    :param roi: Region of interest to crop image to in format x0, y0, x1, y1
    :param downscale_factor: Factor to downscale image
    :param kernel_size: Size of kernel for blurring
    :return: Grayed, downscaled and blurred image
    """
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img[roi[1]:roi[3], roi[0]:roi[2]] if roi is not None else img
    img = cv2.resize(img, (int(img.shape[1] // downscale_factor), int(img.shape[0] // downscale_factor)))
    img = cv2.medianBlur(img, kernel_size)
    
    return img


def get_good_matches(ref_descs, descs, ratio_thresh=0.15):
    # matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    matcher = cv2.FlannBasedMatcher({'algorithm': 0, 'trees': 5}, {'checks': 50})
    matches = matcher.knnMatch(ref_descs, descs, k=2)
    
    good_matches = []
    for match in matches:
        if len(match) > 1:
            if match[0].distance < ratio_thresh * match[1].distance:
                good_matches.append(match)
    
    return good_matches


def plot_matches(img1, img2, pts1, pts2, matches):
    ref_kps = [cv2.KeyPoint(*pt, 50) for pt in pts1]
    kps = [cv2.KeyPoint(*pt, 50) for pt in pts2]
    
    vis = cv2.drawMatchesKnn(img1, ref_kps, img2, kps, matches, None, flags=2)
    plt.imshow(vis)
    plt.show()


def get_matching_points(img1, img2, roi1=None, roi2=None, downscale_factor=4, kernel_size=3, ratio_thresh=0.15, plot=False):
    """
    Using points and descriptors retrieved by a keypoint matcher (here: AKAZE) generate bounding boxes on a new image
    For robustness and accuracy, for each labeled object only nearby keypoints are considered and outliers are removed
    :param img1: Reference image
    :param img2: New image to compare
    :param roi1: Region of interest to crop image to in format x0, y0, x1, y1
    :param roi2: Region of interest to crop image to in format x0, y0, x1, y1
    :param downscale_factor: Scaling factor to resize image for better feature extraction
    :param kernel_size: Size of kernel for blurring
    :param ratio_thresh: Threshold to filter out bad matches [0, 1] (the higher the more matches will be considered)
    :param plot: Whether to plot the matched pairs of keypoints
    :return: Average shift in keypoints
    """
    
    # detector = cv2.AKAZE_create()
    detector = cv2.xfeatures2d.SIFT_create()
    
    # Get keypoints and descriptor for compared image
    kpss, descss = [], []
    for img, roi in zip([img1, img2], [roi1, roi2]):
        img_prepd = prepare_img(img, roi=roi, downscale_factor=downscale_factor, kernel_size=kernel_size)
        kps, descs = detector.detectAndCompute(img_prepd, None)
        
        kpss.append(kps)
        descss.append(descs)
    
    # Get actual points for original image by transforming back the keypoints
    pss = [np.array([kp.pt for kp in kps]) * downscale_factor + (roi[:2] if roi is not None else (0, 0)) for kps, roi in zip(kpss, [roi1, roi2])]
    
    # Filter keypoints and descriptors of current image based on matches with reference image
    tries = 0
    good_matches = get_good_matches(*descss, ratio_thresh=ratio_thresh)
    while len(good_matches) == 0 and tries < 3:  # Succesively increaser ratio_thresh to get more matches
        good_matches = get_good_matches(*descss, ratio_thresh=min((tries + 2) * ratio_thresh, 0.99))
        tries += 1
    
    matched_pts1 = pss[0][[match[0].queryIdx for match in good_matches]]
    matched_pts2 = pss[1][[match[0].trainIdx for match in good_matches]]
    
    if plot:
        plot_matches(img1, img2, *pss, good_matches)
    
    return pss[0], pss[1], good_matches, matched_pts1, matched_pts2


def filter_matches(matched_pts1, matched_pts2):
    dists = [cosine(matched_pt1, matched_pt2) for matched_pt1, matched_pt2 in zip(matched_pts1, matched_pts2)]
    q75, q25 = np.percentile(dists, [75, 25])
    iqr = q75 - q25
    mask = (dists > q25 - 1.5 * iqr) & (dists < q75 + 1.5 * iqr)
    
    return matched_pts1[mask], matched_pts2[mask]


def stitch_images(img1, img2, pts1, pts2):
    homography, _ = cv2.findHomography(pts1, pts2, cv2.RANSAC)
    if homography is None:
        raise ValueError('Homography could not be calculated')
    
    result = cv2.warpPerspective(img1, homography, (2 * img2.shape[1], img1.shape[0]))
    result[:img2.shape[0], :img2.shape[1]] = img2
    
    return result


def create_panorama(img_paths, rois, downscale_factor=4, kernel_size=3, ratio_thresh=0.15):
    stitched_img = cv2.imread(img_paths[0])
    
    gen = zip(*[[lst[i:i + 2] for i in range(len(lst) - 1)] for lst in [img_paths, rois]])
    
    for img_path_pair, roi_pair in tqdm(gen, total=len(img_paths) - 1):
        img_pair = [cv2.imread(path) for path in img_path_pair]
        pts1, pts2, good_matches, matched_pts1, matched_pts2 = get_matching_points(*img_pair, *roi_pair, downscale_factor=downscale_factor,
                                                                                   kernel_size=kernel_size, ratio_thresh=ratio_thresh)
        matched_pts1, matched_pts2 = filter_matches(matched_pts1, matched_pts2)
        stitched_img = stitch_images(stitched_img, img_pair[1], matched_pts1, matched_pts2)
    
    return stitched_img


def main():
    # the first image must be on the right side of the second image
    
    img1 = cv2.imread("../assets/test_images/2Hill.JPG")
    img2 = cv2.imread("../assets/test_images/1Hill.JPG")
    
    pts1, pts2, good_matches, matched_pts1, matched_pts2 = get_matching_points(img1, img2, ratio_thresh=0.6)
    print(len(good_matches))
    res = stitch_images(img1, img2, matched_pts1, matched_pts2)
    
    plt.imshow(cv2.cvtColor(res, cv2.COLOR_BGR2RGB))
    plt.show()


def main2():
    # the first image must be on the right side of the second image
    img_paths = [f"../assets/frames/image_sequence{f'{i:06d}'}.png" for i in range(2, 110, 4)]
    rois = [(0, 373, 1920, 710) for _ in img_paths]
    
    stitched_img = create_panorama(img_paths, rois, ratio_thresh=0.7)
    
    plt.imshow(cv2.cvtColor(stitched_img, cv2.COLOR_BGR2RGB))
    plt.show()


def main3():
    # the first image must be on the right side of the second image
    img_paths = [f"../assets/test_images/S{i}.jpg" for i in [1, 2, 3, 5, 6]][::-1]
    rois = [None for _ in img_paths]
    
    stitched_img = create_panorama(img_paths, rois, ratio_thresh=0.7)
    
    plt.imshow(cv2.cvtColor(stitched_img, cv2.COLOR_BGR2RGB))
    plt.show()


def main4():
    # the first image must be on the right side of the second image
    img_paths = [f"../assets/drone_images/DSC{f'{i:05d}'}.JPG" for i in range(798, 803)][::-1]
    # img_paths = [f"../assets/drone_images/DSC{f'{i:05d}'}.JPG" for i in range(798, 810)][::-1]
    rois = [None for _ in img_paths]
    
    stitched_img = create_panorama(img_paths, rois, ratio_thresh=0.7, downscale_factor=6)
    
    plt.imshow(cv2.cvtColor(stitched_img, cv2.COLOR_BGR2RGB))
    plt.show()


if __name__ == '__main__':
    main4()
