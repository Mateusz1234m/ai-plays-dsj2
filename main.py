import game_control_module
import time
import pytesseract
from PIL import ImageGrab, Image
import acquisition_module
import neural_network_module
import torch
from statistics import mean
import sys

game_controller = game_control_module.GameController()
data_grabber = acquisition_module.DataGrabber()
points_classifier = neural_network_module.PointsClassifier()

game_controller.focus_game()
game_controller.choose_training()
game_controller.choose_hill(hill='FINLAND')
game_controller.test_jump()

points = None

times = [time.time()]
while points is None:
    points_img = data_grabber.get_points_img()
    points = points_classifier.get_points(points_img)
    # print(points)
    times.append(time.time())

game_controller.left_click()

delta_times = [times[i + 1] - times[i] for i in range(len(times) - 1)]
print(delta_times)
print(max(delta_times))
print(mean(delta_times))
print(points)
