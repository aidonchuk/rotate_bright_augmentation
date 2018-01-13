import os
import pickle
from multiprocessing import Pool

import cv2
import numpy as np

from com.aug.RatedSemaphore import RatedSemaphore
from com.aug.Rotation import rot_clock

aug_out_count = 55  #
aug_prob = 0.5
angle_clock = 359
angle_counter_clock = 100
bright = (0, 255)
draw_bb = True
max_img_per_sec = 10000


def image_generator():
    p = Pool()
    p.map(process_image, data)


def rotate(img, coords):
    if angle_clock != 0:
        if np.random.random_sample() <= aug_prob:
            return rot_clock(img, np.random.randint(1, angle_clock + 1), coords)

    if angle_counter_clock != 0:
        if np.random.random_sample() <= aug_prob:
            return rot_clock(img, 360 - np.random.randint(1, angle_counter_clock + 1), coords)

    return img, coords


def apply_bright(img):
    if np.random.random_sample() <= aug_prob:
        return change_brightness(img, np.random.randint(bright[0], bright[1] + 1))
    else:
        return img


def change_brightness(img, value):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img


def process_image(image, rate_limit=RatedSemaphore(value=max_img_per_sec, period=1)):
    with rate_limit:
        for i in range(0, aug_out_count):
            img = np.array(image[0])
            coords = np.array(image[1][1])
            name = image[1][0] + '_' + str(i)
            img = apply_bright(img)
            img, coords = rotate(img, coords)
            write_data(img, name, coords)
        pass


def write_data(img, name, coords, dir='../../output/'):
    pic_name = name + '.jpg'
    dict_ = {'img_name': pic_name, 'coors': list(coords)}  # сохраняем dict не оборачивая в лист, т.к. один файл
    img = draw_bbox(img, coords)
    cv2.imwrite(dir + pic_name, img)
    output = open(dir + name + '.pickle', 'wb')
    pickle.dump(dict_, output)
    output.close()


def draw_bbox(img, coords):
    if draw_bb:
        for c in coords:
            if len(c) > 4:
                cv2.line(img, (c[0], c[1]), (c[2], c[3]), (0, 255, 0), 2)
                cv2.line(img, (c[2], c[3]), (c[4], c[5]), (0, 255, 0), 2)
                cv2.line(img, (c[4], c[5]), (c[6], c[7]), (0, 255, 0), 2)
                cv2.line(img, (c[6], c[7]), (c[0], c[1]), (0, 255, 0), 2)
            else:
                cv2.rectangle(img, (c[0], c[1]), (c[2], c[3]), (0, 255, 0), 2)
        return img
    else:
        return img


def read_data(dir='../../input/'):
    data = []
    data_bb = pickle.load(open(dir + 'imgs_info.pickle', 'rb'))
    assert len(data_bb) is not None
    data_bb = list(map(lambda x: (x.get('img_name')[:x.get('img_name').rfind('.')], x.get('coors')), data_bb))
    imgs_file_name = os.listdir(dir)
    assert len(imgs_file_name) > 0
    for ifn in imgs_file_name:
        if ifn.lower().endswith(".jpg"):
            img_name = ifn[:ifn.rfind('.')]
            image = cv2.imread(dir + ifn)
            coords = list(filter(lambda x: x[0] == img_name, data_bb))
            data.append((image, coords[0]))

    assert len(data) > 0
    return data


if __name__ == '__main__':
    assert 0 <= angle_clock < 360
    assert 0 <= angle_counter_clock < 360
    assert aug_out_count >= 1
    assert 0 < aug_prob < 1
    assert 0 <= bright[0] <= 255 and 0 <= bright[1] <= 255 and bright[0] < bright[1]
    assert 0 < max_img_per_sec

    data = read_data()
    image_generator()
