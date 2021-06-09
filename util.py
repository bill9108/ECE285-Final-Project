from numpy import uint8
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

def IoU(box1, box2):
    '''
    a, b: two bounding boxes. a is prediction and b is ground truth
    a = (xcenter, ycenter, sqrt(w), sqrt(h))
    '''
    box1 = box1.view(1, 5)
    box2 = box2.view(1, 5)
    if (box1[0, 2] <= 0 or box1[0, 3] <= 0):
        return torch.tensor(0.)
    ax1, ax2, ay1, ay2 = convert_center_to_corner(box1)
    bx1, bx2, by1, by2 = convert_center_to_corner(box2)
    a_n_b_w = max(0, min(ax2, bx2) - max(ax1, bx1)) 
    a_n_b_h = max(0, min(ay2, by2) - max(ay1, by1))
    a_n_b = a_n_b_w * a_n_b_h
    a_u_b = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - a_n_b
    return a_n_b / a_u_b

def convert_center_to_corner(box, cell_width=64, cell_height = 64):
    
    #if is_sqrt:
        # xmin = box[0] - box[2] ** 2 / 2
        # xmax = box[0] + box[2] ** 2 / 2
        # ymin = box[1] - box[3] ** 2 / 2
        # ymax = box[1] + box[3] ** 2 / 2
    #else:
        # xmin = box[0, 0] - box[0, 2] / 2
        # xmax = box[0, 0] + box[0, 2] / 2
        # ymin = box[0, 1] - box[0, 3] / 2
        # ymax = box[0, 1] + box[0, 3] / 2
    #box = box.view(1, 5)
    x_pixel = box[0, 0] * cell_width  #X coordinate of box center relative to the cell
    y_pixel = box[0, 1] * cell_height
    width = box[0, 2] * 640
    height = box[0, 3] * 480
    xmin = x_pixel - width / 2
    xmax = x_pixel + width / 2
    ymin = y_pixel - height / 2
    ymax = y_pixel + height / 2

    return (xmin, xmax, ymin, ymax)

def show_img_with_gt(img, box):
    if (isinstance(img, torch.Tensor)):
        img = img.cpu().numpy().astype('uint8')

    im = img.reshape((384, 512, 3))
    im = Image.fromarray(im, 'RGB')
    fig, ax = plt.subplots()
    ax.imshow(im)
    rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    plt.show()

def show_img_with_output(img, output_box, grid_cell):
    if (isinstance(img, torch.Tensor)):
        img = img.cpu().numpy().astype('uint8')
    
    im = img.reshape((384, 512, 3))
    im = Image.fromarray(im, 'RGB')
    fig, ax = plt.subplots()
    ax.imshow(im)

    for i in range(output_box.size()[0]):
        xc, yc = output_box[i, 0:2]
        width, height = output_box[i, 2:4]
        width = width * 640 * 0.8
        height = height * 480 * 0.8
        xc = xc * 64
        yc = yc * 64

        grid_y, grid_x = grid_cell[i, 0:2]
        xc += grid_x * 64
        yc += grid_y * 64

        xmin = xc - width / 2
    #xmax = xc + width / 2
        ymin = yc - height / 2
    #ymax = yc + height / 2

        if (width <= 0) or (height <= 0):
            print("Negative width or height!!!")

        rect = patches.Rectangle((xmin, ymin), width, height, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    plt.show()



# b1 = torch.Tensor([[21/64, 32/64, 15/640, 20/480,0]])
# b2 = torch.Tensor([[30/64, 40/64, 18/640, 25/480,0]])
# print(IoU(b1, b2))
