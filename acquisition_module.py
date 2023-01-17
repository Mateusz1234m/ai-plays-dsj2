from PIL import ImageGrab, ImageOps
import numpy as np


class DataGrabber:
    """
    Class for grabbing game frames.
    """
    def __init__(self, game_bbox=(1, 27, 321, 213),
                 game_over_bbox=(150, 215, 190, 225),
                 points_bbox=(254, 214, 292, 226),
                 points_digit_1_bbox=(0, 0, 7, 11),
                 points_digit_2_bbox=(6, 0, 13, 11),
                 points_digit_3_bbox=(12, 0, 19, 11),
                 points_digit_4_bbox=(18, 0, 25, 11),
                 points_digit_5_bbox=(30, 0, 37, 11)):

        self.game_bbox = game_bbox
        self.game_over_bbox = game_over_bbox
        self.points_bbox = points_bbox
        self.points_digit_1_bbox = points_digit_1_bbox
        self.points_digit_2_bbox = points_digit_2_bbox
        self.points_digit_3_bbox = points_digit_3_bbox
        self.points_digit_4_bbox = points_digit_4_bbox
        self.points_digit_5_bbox = points_digit_5_bbox

    def get_points_img(self):
        """
        Gets image of received points.
        :return:
        """
        points_img = ImageGrab.grab(self.points_bbox)
        return points_img

    def get_game_img(self):
        """
        Gets game frame.
        :return:
        """
        points_img = ImageGrab.grab(self.game_bbox)
        return points_img

    def points_img_2_digits_img(self, points_img):
        """
        Crops points image to seperate digits.
        :param points_img:
        :return:
        """
        digit_1_img = points_img.crop(self.points_digit_1_bbox)
        digit_2_img = points_img.crop(self.points_digit_2_bbox)
        digit_3_img = points_img.crop(self.points_digit_3_bbox)
        digit_4_img = points_img.crop(self.points_digit_4_bbox)
        digit_5_img = points_img.crop(self.points_digit_5_bbox)

        return digit_1_img, digit_2_img, digit_3_img, digit_4_img, digit_5_img

    def get_game_over(self, thresh=10_000):
        """
        Checks if game is over. When jump is finished, distance or disqualification appears on the bottom bar,
            which is detected by summing pixel values and using threshold.
        :param thresh:
        :return:
        """
        # get image of jump distance
        # if there is something in this image, it means game is over
        game_over_img = ImageGrab.grab(self.game_over_bbox)
        game_over_img = np.asarray(ImageOps.grayscale(game_over_img))
        # print(game_over_img.sum())
        game_over = True if game_over_img.sum() > thresh else False

        return game_over
