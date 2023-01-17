import win32api
import win32con
import time


class GameController:
    """
    Class for controlling the game
    """
    def __init__(self, sleep_time=0.1):
        self.sleep_time = sleep_time
        self.hills = {
            "FINLAND": (200, 280)
        }

    @staticmethod
    def left_click():
        """
        Performs left click.
        :return:
        """
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0)
        time.sleep(0.1)
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0)

    @staticmethod
    def right_click():
        """
        Performs right click.
        :return:
        """
        win32api.mouse_event(win32con.MOUSEEVENTF_RIGHTDOWN, 0, 0)
        time.sleep(0.1)
        win32api.mouse_event(win32con.MOUSEEVENTF_RIGHTUP, 0, 0)

    @staticmethod
    def left_and_right_click():
        """
        Performs left and right click at the same time.
        :return:
        """
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0)
        win32api.mouse_event(win32con.MOUSEEVENTF_RIGHTDOWN, 0, 0)
        time.sleep(0.4)
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0)
        win32api.mouse_event(win32con.MOUSEEVENTF_RIGHTUP, 0, 0)

    def move_to_zero(self):
        """
        Moves cursor to base position.
        :return:
        """
        win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, -9999, -9999, 0, 0)
        time.sleep(self.sleep_time)

    def move_to(self, coords):
        """
        Moves cursor to given position.
        :param coords:
        :return:
        """
        x, y = coords
        self.move_to_zero()
        time.sleep(self.sleep_time)
        win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, x, y, 0, 0)

    @staticmethod
    def jumper_move(y):
        """
        Performs jumper move (up or down).
        :param self:
        :param y:
        :return:
        """
        win32api.mouse_event(win32con.MOUSEEVENTF_MOVE,0, y, 0, 0)

    def focus_game(self):
        """
        Focuses the game.
        :return:
        """
        self.move_to_zero()
        time.sleep(self.sleep_time)
        self.move_to((100, 100))
        time.sleep(self.sleep_time)
        self.left_click()
        time.sleep(self.sleep_time)

    def choose_training(self):
        """
        Chooses training mode in the game.
        :return:
        """
        self.move_to((200, 400))
        time.sleep(self.sleep_time)
        self.left_click()

    def choose_hill(self, hill='FINLAND'):
        """
        Chooses hill.
        :param hill:
        :return:
        """
        self.move_to(self.hills[hill])
        time.sleep(self.sleep_time)
        self.left_click()

    def test_jump(self, times=[3.8, 2.7, 10]):
        """
        Performs test jump (hardcoded).
        :param times:
        :return:
        """
        time.sleep(2)
        self.left_click()
        time.sleep(times[0])
        self.left_and_right_click()
        time.sleep(times[1])
        self.left_and_right_click()


if __name__ == 'main':
    GameController = GameController(sleep_time=0.3)

    time.sleep(2)
    GameController.focus_game()
    print('game focused')
    time.sleep(0.2)
    GameController.choose_training()
    print('training Finland')

    for i in range(3):
        print('------')
        GameController.choose_hill(hill='FINLAND')
        print('chosen Finland')
        GameController.test_jump(times=[3.8, 2.5, 5])
        print('test jump finished')