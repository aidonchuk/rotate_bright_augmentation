import cv2
import numpy as np


def rotate_box(bb, cx, cy, h, w, angle):
    new_bb = []
    for i, coord in enumerate(bb):
        m = cv2.getRotationMatrix2D((cx, cy), -angle, 1.0)
        cos = np.abs(m[0, 0])
        sin = np.abs(m[0, 1])
        n_w = int((h * sin) + (w * cos))
        n_h = int((h * cos) + (w * sin))
        m[0, 2] += (n_w / 2) - cx
        m[1, 2] += (n_h / 2) - cy
        v = [coord[0], coord[1], 1]
        calculated = np.dot(m, v)
        new_bb.append(int(round(calculated[0], 0)))
        new_bb.append(int(round(calculated[1], 0)))
    return new_bb


def rotate_bound(img, angle):
    (h, w) = img.shape[:2]
    (c_x, c_y) = (w // 2, h // 2)

    m = cv2.getRotationMatrix2D((c_x, c_y), -angle, 1.0)
    cos = np.abs(m[0, 0])
    sin = np.abs(m[0, 1])

    n_w = int((h * sin) + (w * cos))
    n_h = int((h * cos) + (w * sin))

    m[0, 2] += (n_w / 2) - c_x
    m[1, 2] += (n_h / 2) - c_y

    return cv2.warpAffine(img, m, (n_w, n_h))


def rot_clock(img, angle, coords):
    bb = coords_to_bb(coords)
    rotated_img = rotate_bound(img, angle)

    (h, w) = img.shape[:2]
    (cx, cy) = (w // 2, h // 2)

    new_bb = []
    for i in bb:
        new_bb.append(rotate_box(i, cx, cy, h, w, angle))

    return rotated_img, new_bb


def coords_to_bb(coords):
    bb = []
    for c in coords:
        c[0] = float(c[0])
        c[1] = float(c[1])
        c[2] = float(c[2])
        c[3] = float(c[3])
        bb.append([(c[0], c[1]), (c[2], c[1]), (c[2], c[3]), (c[0], c[3])])

    return bb
