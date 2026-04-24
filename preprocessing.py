import cv2
import numpy as np
from PIL import Image

def deskew(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = cv2.findNonZero(binary)
    _, _, angle = cv2.minAreaRect(coords)
    if angle < -45:
        angle += 90
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def binarize(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary_image = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    if np.mean(binary_image) < 127:
        binary_image = cv2.bitwise_not(binary_image)
    return binary_image


def remove_noise(image):
    # TODO: clean up spots and artifacts

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    cleaned_image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

    return cleaned_image


def segment_lines(image, padding=5):
    # sum pixel values row by row — text rows have low sum (black pixels), gaps have high sum (white pixels)
    row_sums = np.sum(image, axis=1)

    # a row is a gap if its sum is above a threshold (mostly white)
    threshold = image.shape[1] * 255 * 0.95
    in_gap = row_sums > threshold # True for gaps, False for text lines

    lines = []
    in_line = False
    line_start = 0

    for i, is_gap in enumerate(in_gap):
        if not is_gap and not in_line:
            # gap ended, line starts here
            in_line = True
            line_start = i
        elif is_gap and in_line:
            # line ended, gap starts here
            in_line = False
            top = max(0, line_start - padding)
            bottom = min(image.shape[0], i + padding)
            lines.append(image[top:bottom, :])

    # catch last line if image doesn't end with a gap
    if in_line:
        top = max(0, line_start - padding)
        lines.append(image[top:, :])

    return lines


def normalize_line(line):
    h, w = line.shape[:2]
    new_width = int(w * (32 / h))
    resized = cv2.resize(line, (new_width, 32))
    rgb = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(rgb)


def preprocess(image):
    image = deskew(image)
    image = binarize(image)
    image = remove_noise(image)
    lines = segment_lines(image)
    return [normalize_line(line) for line in lines]
