from math import gamma
import numpy as np
from skimage import img_as_float
import cv2
import sys
import scipy
import scipy.linalg
from scipy.interpolate import RectBivariateSpline, interp2d
from skimage.filters import sobel
import matplotlib.pyplot as plt
import skimage.io as skio

def display_image_in_actual_size(img, dpi = 80):
    height = img.shape[0]
    width = img.shape[1]

    # What size does the figure need to be in inches to fit the image?
    figsize = width / float(dpi), height / float(dpi)

    # Create a figure of the right size with one axes that takes up the full figure
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])

    # Hide spines, ticks, etc.
    ax.axis('off')

    # Display the image.
    ax.imshow(img, cmap='gray')

#     plt.show()
    
    return fig, ax

def save_mask(fname, snake, img):
    plt.ioff()
    fig, ax = display_image_in_actual_size(img)
    ax.fill(snake[:, 0], snake[:, 1], '-b', lw=3)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_frame_on(False)
    fig.savefig(fname, pad_inches=0, bbox_inches='tight', dpi='figure')
    plt.close(fig)
    
    mask = skio.imread(fname)
    blue = ((mask[:,:,2] == 255) & (mask[:,:,1] < 255) & (mask[:,:,0] < 255)) * 255
    skio.imsave(fname, blue)
    plt.ion()
    
def display_snake(img, init_snake, result_snake):
    fig, ax = display_image_in_actual_size(img)
    ax.plot(init_snake[:, 0], init_snake[:, 1], '-r', lw=2)
    ax.plot(result_snake[:, 0], result_snake[:, 1], '-b', lw=2)
    ax.set_xticks([]), ax.set_yticks([])
    ax.axis([0, img.shape[1], img.shape[0], 0])
    plt.show()

def active_contour(image, snake, alpha=0.01, beta=0.1,
                   w_line=0, w_edge=1, gamma=0.01,
                   bc='periodic', max_px_move=1.0,
                   max_iterations=2500, convergence=0.1):
    max_iterations = int(max_iterations)
    convergence_order = 10
    img = img_as_float(image)
    if w_edge != 0:
        edge = [sobel(img)]
        for i in range(1):
            edge[i][0, :] = edge[i][1, :]
            edge[i][-1, :] = edge[i][-2, :]
            edge[i][:, 0] = edge[i][:, 1]
            edge[i][:, -1] = edge[i][:, -2]
    else:
        edge = [0]
    img = w_line*img + w_edge*edge[0]
    intp = RectBivariateSpline(np.arange(img.shape[1]),
                                   np.arange(img.shape[0]),
                                   img.T, kx=2, ky=2, s=0)
    x, y = snake[:, 0].astype(float), snake[:, 1].astype(float)
    xsave = np.empty((convergence_order, len(x)))
    ysave = np.empty((convergence_order, len(x)))

    n = len(x)
    a = np.roll(np.eye(n), -1, axis=0) + \
        np.roll(np.eye(n), -1, axis=1) - \
        2*np.eye(n)  
    b = np.roll(np.eye(n), -2, axis=0) + \
        np.roll(np.eye(n), -2, axis=1) - \
        4*np.roll(np.eye(n), -1, axis=0) - \
        4*np.roll(np.eye(n), -1, axis=1) + \
        6*np.eye(n)
    A = -alpha*a + beta*b
    sfixed = False
    if bc.startswith('fixed'):
        A[0, :] = 0
        A[1, :] = 0
        A[1, :3] = [1, -2, 1]
        sfixed = True
    efixed = False
    if bc.endswith('fixed'):
        A[-1, :] = 0
        A[-2, :] = 0
        A[-2, -3:] = [1, -2, 1]
        efixed = True
    sfree = False
    if bc.startswith('free'):
        A[0, :] = 0
        A[0, :3] = [1, -2, 1]
        A[1, :] = 0
        A[1, :4] = [-1, 3, -3, 1]
        sfree = True
    efree = False
    if bc.endswith('free'):
        A[-1, :] = 0
        A[-1, -3:] = [1, -2, 1]
        A[-2, :] = 0
        A[-2, -4:] = [-1, 3, -3, 1]
        efree = True
    inv = scipy.linalg.inv(A+gamma*np.eye(n))

    for i in range(max_iterations):
        fx = intp(x, y, dx=1, grid=False)
        fy = intp(x, y, dy=1, grid=False)
        if sfixed:
            fx[0] = 0
            fy[0] = 0
        if efixed:
            fx[-1] = 0
            fy[-1] = 0
        if sfree:
            fx[0] *= 2
            fy[0] *= 2
        if efree:
            fx[-1] *= 2
            fy[-1] *= 2
        xn = np.dot(inv, gamma*x + fx)
        yn = np.dot(inv, gamma*y + fy)

        dx = max_px_move*np.tanh(xn-x)
        dy = max_px_move*np.tanh(yn-y)
        if sfixed:
            dx[0] = 0
            dy[0] = 0
        if efixed:
            dx[-1] = 0
            dy[-1] = 0
        x += dx
        y += dy

        j = i % (convergence_order+1)
        if j < convergence_order:
            xsave[j, :] = x
            ysave[j, :] = y
        else:
            dist = np.min(np.max(np.abs(xsave-x[None, :]) +
                   np.abs(ysave-y[None, :]), 1))
            if dist < convergence:
                break
    return np.array([x, y]).T

inputImageName = sys.argv[1]
initialSnakeName = sys.argv[2]
outputImageName = sys.argv[3]
alpha = float(sys.argv[4])
beta = float(sys.argv[5])
tau = float(sys.argv[6])
wLine = float(sys.argv[7])
wEdge = float(sys.argv[8])

imgSrc = cv2.imread(inputImageName, cv2.IMREAD_GRAYSCALE)
imgSizeHeight = len(imgSrc)
imgSizeWidth = len(imgSrc[0])
initSnake = np.loadtxt(initialSnakeName)
conture = active_contour(imgSrc, initSnake, alpha, beta, wLine, wEdge, tau)
print(conture)
cv2.imwrite(outputImageName, conture)
save_mask(outputImageName, conture, imgSrc)
#display_snake(imgSrc, initSnake, conture)