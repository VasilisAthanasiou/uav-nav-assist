import os
import cv2 as cv

# -------------------------------------------------- Misc Methods -------------------------------------------------------------- #

def wait_for_esc():
    while True:
        if cv.waitKey(1) == 27:
            cv.destroyAllWindows()
            break


def yes_no(arg):
    args = (['y', 'Y', 'yes', 'Yes', 'YES'], ['n', 'N', 'no', 'No', 'NO'])
    while True:
        if arg in args[0]:
            return True
        elif arg in args[1]:
            return False
        else:
            print("Please give a correct answer: ['y', 'Y', 'yes', 'Yes', 'YES'] or ['n', 'N', 'no', 'No', 'NO']")


def draw_image(img, x, y, radius=10, color=(0, 0, 255)):
    try:
        img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    except cv.error:
        cv.circle(img, (x, y), radius, color, -1)
        return img

    cv.circle(img, (x, y), radius, color, -1)
    return img


def snap_image(img, top_x, top_y, dim):
    return img[top_y:top_y + dim, top_x:top_x + dim]


# ------------------------------------------------------------------------------------------------------------------------------ #


# -------------------------------------------------- User Interface ------------------------------------------------------------ #

class UI:

    def __init__(self, method=''):
        self._method = method
        self.cwd = '../datasets'

    def experiment(self, method):
        """

        Args:
            method: Selects the method to run. Can be = 'simulation', 'plot', 'write text'
        """
        self._method = method
        return self._get_method()

    def _get_method(self):

        if self._method == 'simulation':
            self.cwd += '/travel-assist'
            return self._use_simulation()
        elif (self._method == 'plot') or (self._method == 'write text'):
            self.cwd += '/travel-assist'
            return self._use_evaluation()



# ------------------------------------------------------------------------------------------------------------------------------ #
