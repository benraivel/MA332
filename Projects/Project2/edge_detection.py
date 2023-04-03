import numpy as np
import matplotlib.pyplot as plt


# Derivative Functions
def three_point_derivative(y):
    ''' Compute the derivative of y using three point centered difference
    formula. Derivatives at endpoints are calculated with three point
    left/right difference formula
    '''

    dy = np.empty(y.shape)

    dy[1:-1] = (y[2:] - y[:-2])/2
    dy[0] = (-3*y[0] + 4*y[1] - y[2])/2
    dy[-1] = (y[-3] - 4*y[-2] + 3*y[-1])/2

    return dy


def five_point_derivative(y):
    ''' Compute the derivative of y using five point centered difference
    formula. Derivatives at endpoints are calculated with five point
    left/right difference formula
    '''
    dy = np.empty(y.shape)

    dy[:2] = (-25*y[:2] + 48*y[1:3] - 36*y[2:4]
              + 16*y[3:5] - 3*y[4:6])/12

    dy[2:-2] = (y[:-4] - 8*y[1:-3] + 8*y[3:-1] - y[4:])/12

    dy[-2:] = (3*y[-6:-4] - 16*y[-5:-3] + 36*y[-4:-2]
               - 48*y[-3:-1] + 25*y[-2:])/12

    return dy


def seven_point_derivative(y):
    ''' Compute the derivative of y using seven point centered difference
    formula. Derivatives at endpoints are calculated with seven point
    left/right difference formula
    '''
    dy = np.empty(y.shape)

    dy[:3] = ((-49/20)*y[:3] + 6*y[1:4] - (15/2)*y[2:5] + (20/3)*y[3:6]
              - (15/4)*y[4:7] + (6/5)*y[5:8] - (1/6)*y[6:9])

    dy[3:-3] = (-y[:-6] + 9*y[1:-5] - 45*y[2:-4] + 45*y[4:-2] - 9*y[5:-1]
                + y[6:])/60

    dy[-3:] = ((49/20)*y[-3:] - 6*y[-4:-1] + (15/2)*y[-5:-2] - (20/3)*y[-6:-3]
               + (15/4)*y[-7:-4] - (6/5)*y[-8:-5] + (1/6)*y[-9:-6])

    return dy


def nine_point_derivative(y):
    ''' Compute the derivative of y using nine point centered difference
    formula. Derivatives at endpoints are calculated with seven point
    left/right difference formula
    '''
    dy = np.empty(y.shape)

    dy[:4] = ((-49/20)*y[:4] + 6*y[1:5] - (15/2)*y[2:6] + (20/3)*y[3:7]
              - (15/4)*y[4:8] + (6/5)*y[5:9] - (1/6)*y[6:10])

    dy[4:-4] = ((1/280)*y[:-8] - (4/105)*y[1:-7] + (1/5)*y[2:-6]
                - (4/5)*y[3:-5] + (4/5)*y[5:-3] - (1/5)*y[6:-2]
                + (4/105)*y[7:-1] - (1/280)*y[8:])

    dy[-4:] = ((49/20)*y[-4:] - 6*y[-5:-1] + (15/2)*y[-6:-2] - (20/3)*y[-7:-3]
               + (15/4)*y[-8:-4] - (6/5)*y[-9:-5] + (1/6)*y[-10:-6])

    return dy


# Filtering Functions
def grayscale(img):
    ''' Returns grayscale version of color image using proper gamma expansion
    to preserve luminance'''

    below_threshold = img/12.92

    above_threshold = ((img + 0.055)/1.055)**2.4

    expanded = np.where(img <= 0.04045, above_threshold, below_threshold)

    coeff = [0.2126, 0.7152, 0.0722]
    y_lin = np.dot(expanded, coeff)

    return normalize(y_lin)


def gaussian(x: np.ndarray, val=0, var=1) -> np.ndarray:
    ''' Probability-density form (integrates to 1) of gaussian function
    with expectation value val and variance var'''

    const = 1/(var*np.sqrt(2*np.pi))

    exponent = -(x - val)**2/(2*var**2)

    return const*np.exp(exponent)


def gaussian2d(var, size=(1, 1, 1, 1)) -> np.ndarray:
    ''' Normalized 2d gaussian, size is the number of points
    to include on the (left, right, top, bottom) of the expectation value'''

    left, right, top, bottom = size

    x_size = left+right+1
    y_size = bottom+top+1

    x_wts = gaussian(np.linspace(-left, right, x_size), var=var)
    y_wts = gaussian(np.linspace(-top, bottom, y_size), var=var)

    # use meshgrid to create 2d gaussian
    xx_wts, yy_wts = np.meshgrid(x_wts, y_wts)
    wts = xx_wts * yy_wts

    # integrate over weights
    integral_wts = np.sum(wts)

    corrected_wts = wts/integral_wts

    return corrected_wts


def blur(img: np.ndarray, r: int, var=1) -> np.ndarray:
    ''' Apply a Gaussian blur to img.

    kernel shape is (2*r + 1, 2*r + 1)'''

    img_x, img_y = img.shape

    # array to hold results
    result = np.full(img.shape, np.nan)

    # do the center first, this reuses the full kernel
    # and avoids unecessary re-generation
    full_kernel = gaussian2d(var, (r, r, r, r)).flatten()

    for x in range(r, img_x - r):
        for y in range(r, img_y - r):

            sub_img = img[x - r: x + r + 1, y - r: y + r + 1]

            result[x, y] = np.dot(sub_img.flatten(), full_kernel)

    # now do the edge
    for x in range(img_x):
        for y in range(img_y):

            # nan indicates an unfilled value
            if np.isnan(result[x, y]):
                left = np.min([x, r])
                right = np.min([img_x - 1 - x, r])
                top = np.min([y, r])
                bottom = np.min([img_y - 1 - y, r])

                kernel = gaussian2d(var, (left, right, top, bottom))

                sub_img = img[x - left: x + right + 1, y - top: y + bottom + 1]

                result[x, y] = np.dot(kernel.flatten(), sub_img.flatten())

    return result


def normalize(data):
    return (data - np.amin(data))/(np.amax(data) - np.amin(data))


# Display Functions
def show_image_hist(img, size=6):
    ''' Display a grayscale image with inset pixel brightness histogram'''

    if img.shape[0] > img.shape[1]:
        figsize = (img.shape[1]/img.shape[0]*size, size)
    else:
        figsize = (size, img.shape[0]/img.shape[1]*size)

    fig, ax = plt.subplots(figsize=figsize, dpi=150)

    ax.imshow(img, cmap='gray')
    ax.set_axis_off()

    x0, y0, x1, y1 = ax.get_position().bounds
    sub_ax = fig.add_axes((x0, y0, 0.1, 0.1))
    sub_ax.hist(img.flatten(), 100, log=True)
    sub_ax.set_axis_off()

    plt.show()


def show_edges(edges, img, size=6, dpi=150):

    if img.shape[0] > img.shape[1]:
        figsize = (img.shape[1]/img.shape[0]*size, size)
    else:
        figsize = (size, img.shape[0]/img.shape[1]*size)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    ax.imshow(img, cmap='gray')

    ax.scatter(edges[0], edges[1], c='red', marker='+', s=0.5, alpha=0.01)

    ax.axis('off')

    plt.show()

    return fig


# Edge Detection Functions
def detect_edges(img, derivative_func, threshold=0.5, return_grad=False):
    ''' Detect edges in a grayscale image returns x and y coordinates of edges
    '''
    rows, cols = img.shape

    # compute derivative of each row
    r_grad = np.empty(img.shape)
    for r in range(rows):
        r_grad[r] = derivative_func(img[r])

    # compute derivative of each column
    c_grad = np.empty(img.shape)
    for c in range(cols):
        c_grad[:, c] = derivative_func(img[:, c])

    # find magnitude and normalize
    grad = normalize((r_grad**2 + c_grad**2)**0.5)

    edge_x = []
    edge_y = []

    # record x and y coords of edges
    for r in range(rows):
        for c in range(cols):
            if grad[r][c] > threshold:
                edge_x.append(c)
                edge_y.append(r)

    if return_grad:
        return grad

    return np.array([edge_x, edge_y])


def detect_edges_hessian(img, derivative_func, thresh=0.5):

    rows, cols = img.shape

    dx = np.empty(img.shape)
    dy = np.empty(img.shape)

    dx_dx = np.empty(img.shape)
    dx_dy = np.empty(img.shape)
    dy_dx = np.empty(img.shape)
    dy_dy = np.empty(img.shape)

    for r in range(rows):
        dx[r] = derivative_func(img[r])

    for c in range(cols):
        dy[:, c] = derivative_func(img[:, c])

    for r in range(rows):
        dx_dx[r] = derivative_func(dx[r])
        dy_dx[r] = derivative_func(dy[r])

    for c in range(cols):
        dy_dy[:, c] = derivative_func(dy[:, c])
        dx_dy[:, c] = derivative_func(dx[:, c])

    determinant = normalize(np.abs(dx_dx*dy_dy - dx_dy*dy_dx))

    edges_x = []
    edges_y = []

    for r in range(rows):
        for c in range(cols):
            if determinant[r][c] > thresh:
                edges_x.append(c)
                edges_y.append(r)

    return np.array([edges_x, edges_y])
