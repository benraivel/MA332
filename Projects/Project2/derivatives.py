import numpy as np

def three_point_derivative(y):
    dy = np.empty(y.shape)
    
    dy[1:-1] = (y[2:] - y[:-2])/2
    dy[0] = (-3*y[0] + 4*y[1] - y[2])/2
    dy[-1] = (y[-3] - 4*y[-2] + 3*y[-1])/2

    return dy

def five_point_derivative(y):

    dy = np.empty(y.shape)
    center_coeff = np.array([1, -8, 0, 8, -1])
    dy[0] = (-25*y[0] + 48*y[1] - 36*y[2] + 16*y[3] - 3*y[4])/12
    dy[1] = (-25*y[1] + 48*y[2] - 36*y[3] + 16*y[4] - 3*y[5])/12
    dy[2:-2] = (y[:-4] - 8*y[1:-3] + 8*y[3:-1] - y[4:])/12
    dy[-2] = (3*y[-6] - 16*y[-5] + 36*y[-4] - 48*y[-3] + 25*y[-2])/12
    dy[-1] = (3*y[-5] - 16*y[-4] + 36*y[-3] - 48*y[-2] + 25*y[-1])/12


class FiniteDifference:

    def __init__(self, derivative=1, points=3) -> None:
        pass