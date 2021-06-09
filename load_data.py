from PIL import Image
import numpy as np
from os import listdir
from os.path import isfile, join
import math
from numpy.core.fromnumeric import resize
import torch

def load_images(labeled_idx, test=False, resize_factor=0.8):
    size = (int(640*resize_factor), int(480*resize_factor))
    imgs = np.zeros((len(labeled_idx), 3, size[0] * size[1]), dtype=np.uint8)

    jpg_files = [f for f in listdir('./Lara3D_UrbanSeq1_JPG') if isfile(join('./Lara3D_UrbanSeq1_JPG/', f))]
    if test:
        num_imgs = 3
    else:
        num_imgs = len(labeled_idx)

    for i in range(0, num_imgs):
        im = Image.open(join('./Lara3D_UrbanSeq1_JPG/', jpg_files[labeled_idx[i]]))
        #im.show()
        im = im.resize(size)
        im = np.array(im)
        im = np.reshape(im, (3, size[0] * size[1]))
        imgs[i, :, :] = im
    
    return imgs



def load_truths(resize_factor=0.8):
    f = open('.\Lara_UrbanSeq1_GroundTruth_GT.txt', 'r')
    ground_truths = np.zeros((9168, 6, 8, 14), dtype=np.float32)

    lines = f.readlines()
    lines = lines[13:]
    class_dict = dict()
    class_dict['go'] = 0
    class_dict['stop'] = 2
    class_dict['warning'] = 1
    class_dict['ambiguous'] = 3
    labeled_idx = list()

    for i in range(len(lines)):
        l = lines[i]
        l = l.strip('\n')
        l = l.replace('\'', '')
        attributes = l.split(' ')
        attributes = attributes[2:]
        labeled_idx.append(int(attributes[0]))
        if (int(attributes[3]) > 640):
            attributes[3] = 640
        if (int(attributes[4]) > 480):
            attributes[4] = 480
            

        grid_x = int((int(attributes[1]) + int(attributes[3])) / 2 * resize_factor // (640 * resize_factor / ground_truths.shape[2]))
        grid_y = int((int(attributes[2]) + int(attributes[4])) / 2 * resize_factor // (480 * resize_factor / ground_truths.shape[1]))
        x = (int(attributes[1]) + int(attributes[3])) / 2 * resize_factor - grid_x * (640 * resize_factor / ground_truths.shape[2]) 
        y = (int(attributes[2]) + int(attributes[4])) / 2 * resize_factor - grid_y * (480 * resize_factor / ground_truths.shape[1])
        ground_truths[i, grid_y, grid_x, 0] = x / (640 * resize_factor / ground_truths.shape[2]) #x for center of box in a grid cell
        ground_truths[i, grid_y, grid_x, 1] = y / (480 * resize_factor / ground_truths.shape[1])  #y for center of box in a grid cell

        ground_truths[i, grid_y, grid_x, 2] = abs(int(attributes[3]) - int(attributes[1])) / 640  #width = sqrt((x2 - x1) / 640)
        ground_truths[i, grid_y, grid_x, 3] = abs(int(attributes[4]) - int(attributes[2])) / 480  #height = sqrt((y2 - y1) / 480)
        c = class_dict[attributes[-1]]  #Class
        ground_truths[i, grid_y, grid_x, 10 + c] = 1
        ground_truths[i, grid_y, grid_x, 4] = 1
        ground_truths[i, grid_y, grid_x, 5:10] = ground_truths[i, grid_y, grid_x, 0:5]
        

    return (torch.Tensor(ground_truths), labeled_idx)


def load_gt_raw(resize_factor=0.8):
    f = open('.\Lara_UrbanSeq1_GroundTruth_GT.txt', 'r')
    ground_truths_raw = np.zeros((9168, 4), dtype=np.uint16)

    lines = f.readlines()
    lines = lines[13:]
    class_dict = dict()
    class_dict['go'] = 0
    class_dict['stop'] = 2
    class_dict['warning'] = 1
    class_dict['ambiguous'] = 3
    labeled_idx = list()

    for i in range(len(lines)):
        l = lines[i]
        l = l.strip('\n')
        l = l.replace('\'', '')
        attributes = l.split(' ')
        attributes = attributes[2:]
        labeled_idx.append(int(attributes[0]))
        if (int(attributes[3]) > 640):
            attributes[3] = 640
        if (int(attributes[4]) > 480):
            attributes[4] = 480
            
        ground_truths_raw[i, :] = attributes[1:5]
    
    ground_truths_raw  = ground_truths_raw * resize_factor

    return (ground_truths_raw, labeled_idx)
# gt, idx = load_truths()
# #print(gt)

# test = load_images(idx, test=True)
# im = test[0, :, :].reshape((384, 512, 3))
# img = Image.fromarray(im, 'RGB')
# img.show()


