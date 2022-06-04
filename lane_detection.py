import matplotlib.pylab as plt
import cv2
import numpy as np
import statistics
import math


class Color:
    lower = [0, 0, 0]
    upper = [0, 0, 0]


class Yellow(Color):
    lower = [22, 93, 0]
    upper = [30, 255, 255]


class Gray(Color):
    upper = [131, 255, 131]


def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)




def process_image(image, color: Color):
    original = image.copy()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array(color.lower, dtype="uint8")
    upper = np.array(color.upper, dtype="uint8")
    mask = cv2.inRange(image, lower, upper)
    masked = cv2.bitwise_and(original, original, mask=mask)

    return masked


def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    match_mask_color = 255
    cv2.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines_simple(img, lines):
    img = np.copy(img)
    blank_image = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    mean_m = None
    all_m = []
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                if abs(x1 - x2) > 25:
                    m = (y2 - y1) / (x2 - x1)
                    all_m.append(((x1, y1), (x2, y2), m))

        if len(all_m) > 0:
            all_means = [i[2] for i in all_m]
            mean_m = statistics.mean(all_means)
            for line in all_m:
                pt1, pt2, m = line
                if abs(m - mean_m) < 0.1:
                    cv2.line(blank_image, pt1, pt2, (255, 0, 0), thickness=5)
                else:
                    all_m.remove(line)

        if len(all_m) > 0:
            all_means = [i[2] for i in all_m]
            cv2.putText(blank_image, f"Mean = {round(statistics.mean(all_means), 2)}", (300, 500),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 2, cv2.LINE_AA)
            mean_m = statistics.mean(all_means)
        img = cv2.addWeighted(img, 0.8, blank_image, 1, 0.0)
    return img, mean_m


def draw_lines(img, lines):
    if lines is None:
        return img, -1, -1, -1, -1, -1, -1
    img = np.copy(img)
    width, height = img.shape[1], img.shape[0]
    blank_image = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

    """
    ANOTHER FUNCTION
    GET TURN DETECTION
    IMG is actually masked
    """

    pt1, pt2 = (int(2 * width / 3), int(height / 2.5)), (width, int(4 * height / 5))
    cv2.rectangle(blank_image, pt1, pt2, (0, 255, 0), thickness=3)

    search_area = img[pt1[1]:pt2[1], pt1[0]:pt2[0]]
    search_area = cv2.cvtColor(search_area, cv2.COLOR_RGBA2GRAY)
    ret, thresh = cv2.threshold(search_area, 127, 255, cv2.THRESH_BINARY)
    count_zero = cv2.countNonZero(thresh)
    cv2.putText(img, f'non zero pixels {count_zero}', (400, 175), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                cv2.LINE_AA)

    all_m = []
    for line in lines:
        for x1, y1, x2, y2 in line:
            if abs(x1 - x2) > 25:
                m = (y2 - y1) / (x2 - x1)
                if abs(m) > 0.1:
                    all_m.append(((x1, y1), (x2, y2), m))

    if len(all_m) > 0:
        mean_m = statistics.mean([i[2] for i in all_m])
        for point in all_m:
            if abs(point[2] - mean_m) > 0.1:
                all_m.remove(point)

    max_w = int(0.1 * width)
    max_h = height
    min_w = 0
    min_h = height - 60
    ## 600, 800

    count = 0
    right = 0
    left = 0
    right_x = []
    left_y = []

    max_y_detection = -1
    for point in all_m:
        color = (10, 255, 150)
        count_before = count
        for i in range(2):
            x, y = point[i]
            if x < max_w and min_h < y < max_h:
                color = (255, 0, 0)
                count += 1
            if y > max_y_detection:
                max_y_detection = y
        if count_before == count:
            for i in range(2):
                x, y = point[i]
                if x < max_w and y < min_h:
                    color = (0, 0, 255)
                    left += 1
                    left_y.append(y)
                    break
                if x > max_w and y > min_h:
                    color = (0, 0, 255)
                    right += 1
                    right_x.append(x)
                    break
                if y > max_y_detection:
                    max_y_detection = y
        cv2.line(blank_image, point[0], point[1], color, thickness=5)

    if max_y_detection < 400:
        img = cv2.addWeighted(img, 0.8, blank_image, 1, 0.0)
        return img, -1, -1, -1, -1, -1, -1

    score_right, score_left = 0, 0
    if len(right_x) > 0 and count < 5:
        mean_right_point = (int(statistics.mean(right_x)), height - 60)
        cv2.line(blank_image, mean_right_point, (0, height), (255, 255, 0), thickness=3)
        m_line = round((mean_right_point[1] - height) / (mean_right_point[0] - 0), 2)
        score_right = round((0.8 - abs(m_line)) * distance(mean_right_point, (0, height)), 2)
        score_right = score_right / 350

    if len(left_y) > 0 and count < 5:
        mean_left_point = (40, int(statistics.mean(left_y)))
        cv2.line(blank_image, mean_left_point, (0, height), (255, 255, 0), thickness=3)
        m_line = round((mean_left_point[1] - height) / (mean_left_point[0] - 0), 2)
        score_left = round(abs(m_line) * distance(mean_left_point, (0, height)), 2)
        score_left = score_left / 800
    img = cv2.addWeighted(img, 0.8, blank_image, 1, 0.0)
    img = cv2.rectangle(img, (min_w, min_h), (max_w, max_h), (255, 0, 0), thickness=5)
    cv2.line(img, (0, 400), (width, 400), (255, 255, 0), thickness=3)
    cv2.putText(img, str(count), (600, 500), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 0), 2, cv2.LINE_AA)
    return img, count, left, right, score_left, score_right, count_zero


class RegionsOfInterest:
    turn_detection = 0
    steer_adjust = 1


def detect_lanes(image, color, reg):
    image = process_image(image, color)
    height = image.shape[0]
    width = image.shape[1]
    if reg == RegionsOfInterest.turn_detection:
        region_of_interest_vertices = [
            (0, height // 2),
            (width, height // 2),
            (width, height),
            (0, height)
        ]
    elif reg == RegionsOfInterest.steer_adjust:
        region_of_interest_vertices = [
            (0, 0),
            (width, 0),
            (width, height),
            (0, height)
        ]
    else:
        return image, None
    # gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray_image = image
    canny_image = cv2.Canny(gray_image, 100, 200)
    cropped_image = region_of_interest(canny_image,
                                       np.array([region_of_interest_vertices], np.int32), )
    lines = cv2.HoughLinesP(cropped_image,
                            rho=6,
                            theta=np.pi / 180,
                            threshold=160,
                            lines=np.array([]),
                            minLineLength=20,
                            maxLineGap=100)
    return image, lines
