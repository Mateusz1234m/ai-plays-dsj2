import os

from PIL import ImageGrab, ImageOps
import numpy as np
import time
import pandas as pd
import sys
import win32api

from pynput import mouse

import game_control_module

start_y = 126
total_dy = 0
last_nonzero_dy = 0

class DataGrabber:
    """
    Class for grabbing game frames.
    """
    def __init__(self, game_bbox=(1, 27, 321, 213),
                 game_over_bbox=(150, 215, 190, 225),
                 distance_bbox=(254, 214, 292, 226),
                 points_bbox=(254, 214, 292, 226),
                 points_digit_1_bbox=(0, 0, 7, 11),
                 points_digit_2_bbox=(6, 0, 13, 11),
                 points_digit_3_bbox=(12, 0, 19, 11),
                 points_digit_4_bbox=(18, 0, 25, 11),
                 points_digit_5_bbox=(30, 0, 37, 11)):

        """
        Initialize DataGrabber.
        :param game_bbox: coordinates of the bounding box with the game window
        :param game_over_bbox: coordinates of the bounding box with the disqualification and distance
        :param distance_bbox: coordinates of the bounding box with the distance
        :param points_bbox: coordinates of the bounding box with the points
        :param points_digit_1_bbox: coordinates of the bounding box with the first digit
        :param points_digit_2_bbox: coordinates of the bounding box with the second digit
        :param points_digit_3_bbox: coordinates of the bounding box with the third digit
        :param points_digit_4_bbox: coordinates of the bounding box with the fourth digit
        :param points_digit_5_bbox: coordinates of the bounding box with the fifth digit
        """
        self.game_bbox = game_bbox
        self.game_over_bbox = game_over_bbox
        self.distance_bbox = distance_bbox
        self.points_bbox = points_bbox
        self.points_digit_1_bbox = points_digit_1_bbox
        self.points_digit_2_bbox = points_digit_2_bbox
        self.points_digit_3_bbox = points_digit_3_bbox
        self.points_digit_4_bbox = points_digit_4_bbox
        self.points_digit_5_bbox = points_digit_5_bbox

    def get_points_img(self):
        """
        Gets image of received points.
        :return: points image
        """
        points_img = ImageGrab.grab(self.points_bbox)
        return points_img

    def get_game_img(self):
        """
        Gets game frame.
        :return:game image
        """
        game_img = ImageGrab.grab(self.game_bbox)
        return game_img

    def points_img_2_digits_img(self, points_img):
        """
        Crops points image to seperate digits.
        :param points_img: image of the points
        :return: images of separete digits
        """
        digit_1_img = points_img.crop(self.points_digit_1_bbox)
        digit_2_img = points_img.crop(self.points_digit_2_bbox)
        digit_3_img = points_img.crop(self.points_digit_3_bbox)
        digit_4_img = points_img.crop(self.points_digit_4_bbox)
        digit_5_img = points_img.crop(self.points_digit_5_bbox)

        return digit_1_img, digit_2_img, digit_3_img, digit_4_img, digit_5_img

    def sum_game_pixels(self):
        """
        Sums up values of all pixels in the game image.
        :return: pixels sum
        """

        # get game image
        game_img = ImageGrab.grab(self.game_bbox)

        return np.asarray(ImageOps.grayscale(game_img)).sum()

    def get_game_over(self, thresh=10_000):
        """
        Checks if game is over. When jump is finished, distance or disqualification appears on the bottom bar,
        which is detected by summing pixel values and using threshold.
        :param thresh: threshold
        :return: game over
        """

        # get image of jump distance
        # if there is something in this image, it means game is over
        game_over_img = ImageGrab.grab(self.game_over_bbox)
        game_over_img = np.asarray(ImageOps.grayscale(game_over_img))
        # print(game_over_img.sum())
        game_over = True if game_over_img.sum() > thresh else False

        return game_over


class PretrainDataGrabber(DataGrabber):
    """
    Class for grabbing game frames during acquisition of the pretrain dataset.
    """
    def __init__(self, img_path, labels_path):
        """
        Initialize pretrain data grabber.
        :param img_path: path to folder where images will be saved
        :param labels_path: path to csv file where labels will be stored
        """
        DataGrabber.__init__(self)

        # buffer for images and labels
        self.buffer = {
            "images": [],
            "labels_total_time": [],
            "labels_flight_time": [],
            "labels_position": [],
            "labels_inclination": []
        }

        self.img_path = img_path
        self.labels_path = labels_path

    def check_hill_displayed(self):
        """
        Checks if hill is displayed based on pixels sum.
        :return:
        """

        pixel_sum = self.sum_game_pixels()
        hill_displayed = True if pixel_sum > 5_000_000 else False

        return hill_displayed

    def save_buffer(self):
        """
        Saves buffer of data gathered during jump.
        :return:
        """

        # get list of images in image folder
        img_list = os.listdir(self.img_path)

        # specify start index
        start_idx = 0 if len(img_list) == 0 else max([int(x[:-4]) for x in img_list]) + 1

        # get dataframe with labels
        if os.path.exists(self.labels_path):
            df = pd.read_csv(self.labels_path)
        else:
            df = pd.DataFrame(columns=["img", "labels_total_time", "labels_flight_time", "labels_position",
                                       "labels_inclination"])

        # create list where filenames will be stored
        filenames = []

        # save images
        for i in range(len(self.buffer["images"])):

            # get filename
            filename = str(start_idx + i).zfill(6) + ".png"

            # append filename to list of filenames
            filenames.append(filename)

            # save image
            self.buffer["images"][i].save(self.img_path + filename)

        # create dataframe with all labels
        temp_df = pd.DataFrame({
            "img": filenames,
            "labels_total_time": self.buffer["labels_total_time"],
            "labels_flight_time": self.buffer["labels_flight_time"],
            "labels_position": self.buffer["labels_position"],
            "labels_inclination": self.buffer["labels_inclination"],
        })

        # concatenate created and existing dataframe
        df = pd.concat([df, temp_df], axis=0).reset_index(drop=True)
        # print(df.tail())

        # save labels to csv file
        df.to_csv(self.labels_path, index=False)

    def clear_buffer (self):
        """
        Clears buffer.
        :return:
        """
        self.buffer = {
            "images": [],
            "labels_total_time": [],
            "labels_flight_time": [],
            "labels_position": [],
            "labels_inclination": []
        }

    def create_dataset(self):
        """
        Creates the dataset while player performs jumps.
        :return:
        """

        # for rising edge detection
        prev_hill_displayed = False

        # listener for mouse events detection
        listener = mouse.Listener(on_move=on_move)

        # start the mouse listener
        print("Starting listener")
        listener.start()

        global total_dy
        global last_nonzero_dy

        while True:

            # check if hill is displayed
            hill_displayed = self.check_hill_displayed()

            if hill_displayed and not prev_hill_displayed:

                # wait until jump is started
                while True:
                    left_edge, right_edge = get_pressed_edge()
                    if left_edge or right_edge:
                        break

                # define start values
                total_start_time = time.time()
                flight_time = 0
                jump_finished = False
                position = 0

                # get
                while not jump_finished:

                    left_edge, right_edge = get_pressed_edge()

                    if position == 0:
                        if left_edge or right_edge < 0:
                            position = 1
                            takeoff_time = time.time()

                            total_dy = 0
                            last_nonzero_dy = 0

                    if position == 1:
                        if time.time() - takeoff_time > 0.5:

                            if left_edge and right_edge:
                                position = 2
                            if left_edge:
                                position = 3
                            if right_edge:
                                position = 4

                    if (position == 3 and right_edge) or (position == 4 and left_edge):
                        position = 2

                    if position > 0:
                        flight_time = time.time() - takeoff_time


                    if position == 0 or position > 1:
                        inclination = 0
                    else:

                        if total_dy > 90:
                            total_dy = 90

                        if total_dy < -50:
                            total_dy = -50

                        inclination = total_dy
                        # print(inclination)

                    # get current game image
                    img = self.get_game_img()
                    self.buffer["images"].append(img)

                    # get labels
                    self.buffer["labels_total_time"].append(time.time() - total_start_time)
                    self.buffer["labels_flight_time"].append(flight_time)
                    self.buffer["labels_position"].append(position)
                    self.buffer["labels_inclination"].append(inclination)

                    # check if jump is finished
                    jump_finished = self.get_game_over()

                # save and clear buffer
                self.save_buffer()
                self.clear_buffer()

            # for rising edge detection
            prev_hill_displayed = hill_displayed

        # stop mouse events listener
        print("Stopping listener")
        listener.stop()


# specify global variables
prev_left, prev_right, left_clicked, right_clicked = None, None, None, None


def get_pressed_edge():
    """
    Cheks if left or right mouse click was performed.
    :return: info about left and right clicks
    """

    global left, right, prev_left, prev_right, left_clicked, right_clicked

    # get left and right key states values
    left = win32api.GetKeyState(0x01)
    right = win32api.GetKeyState(0x02)

    # check if mouse clicks were performed
    left_edge = left < 0 and (left is None or left != prev_left)
    right_edge = right < 0 and (right is None or right != prev_right)

    # save previous values
    if left < 0:
        prev_left = left

    if right < 0:
        prev_right = right

    return left_edge, right_edge


def on_move(_x, y):
    """
    Function used when mouse movement is performed.
    :param _x: x movement, not used
    :param y: y movement
    :return:
    """
    global last_y, total_dy, last_nonzero_dy, start_y

    dy = y - start_y
    total_dy += dy
    last_y = y

if __name__ == "__main__":
    data_grabber = PretrainDataGrabber(img_path="pretrain_dataset/data/", labels_path="pretrain_dataset/labels.csv")
    data_grabber.create_dataset()
