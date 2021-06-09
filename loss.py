import numpy as np
from numpy.core.fromnumeric import size
import torch
import math
import load_data
import util
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class YoloLoss(nn.Module):
    def __init__(self):
        super(YoloLoss,self).__init__()
        self.lambda_coord = 8
        self.lambda_noobj = 1 
        self.lambda_class = 0.7
        #self.lambda_co_conf = 2

    def forward(self, output, ground_truth):
        '''
        output dimension: N*6*8*14
        ground_truth: x, y, sqrt(w), sqrt(h), class
        '''
        
        num_sample = output.shape[0]

        co_mask = ground_truth[:,:,:,4] > 0
        no_mask = ground_truth[:,:,:,4] == 0
        co_mask = co_mask.unsqueeze(-1).expand_as(ground_truth)
        no_mask = no_mask.unsqueeze(-1).expand_as(ground_truth)

        co_pred = output[co_mask].view(-1,14)
        out_box = co_pred[:,:10].contiguous().view(-1,5)
        #zero_h_mask = out_box[:, 2] < 0  # Deal with neg height and width value at the beginning of the training
        #zero_w_mask = out_box[:, 3] < 0
        #out_box[zero_h_mask, 2] = 0
        #out_box[zero_w_mask, 3] = 0 
        out_class = co_pred[:,10:]                      
        
        co_gt = ground_truth[co_mask].view(-1,14)
        gt_box = co_gt[:,:10].contiguous().view(-1,5)
        gt_class = co_gt[:,10:]

        #No object grid cell loss: just the confidence loss
        no_out = output[no_mask].view(-1, 14)
        no_gt = ground_truth[no_mask].view(-1, 14)
        no_out_mask = torch.cuda.BoolTensor(no_out.size())
        no_out_mask.zero_()
        no_out_mask[:, 4] = 1
        no_out_mask[:, 9] = 1
        no_out_conf = no_out[no_out_mask]
        no_gt_conf = no_gt[no_out_mask]
        no_loss = F.mse_loss(no_out_conf, no_gt_conf, size_average=False)

        co_maxbox_mask = torch.cuda.BoolTensor(gt_box.size())
        co_maxbox_mask.zero_()
        co_box_mask = torch.cuda.BoolTensor(gt_box.size())
        co_box_mask.zero_()
        box_IoU_target = torch.zeros(gt_box.size())
        conf_idx_cuda = torch.LongTensor([4])

        for i in range(0, gt_box.size()[0], 2):
            box1 = out_box[i:i + 1]
            box2 = out_box[i + 1:i + 2]
            box_ground = gt_box[i:i + 1]
            IoUs = torch.zeros([1, 2]).cuda()
            IoUs[0, 0] = util.IoU(box1, box_ground)
            IoUs[0, 1] = util.IoU(box2, box_ground)
            max_IoU, max_idx = IoUs.max(1)
            max_idx = max_idx.data.cuda()

            co_maxbox_mask[i + max_idx] = 1
            co_box_mask[i + 1 - max_idx] = 1
            box_IoU_target[i + max_idx, conf_idx_cuda] = max_IoU.data

        box_IoU_target = Variable(box_IoU_target).cuda()

        # Loss for box responsible for the prediction
        out_box_max = out_box[co_maxbox_mask].view(-1, 5)
        gt_box_max = gt_box[co_maxbox_mask].view(-1, 5)
        box_IoU_target = box_IoU_target[co_maxbox_mask].view(-1, 5)
        #conf_loss = F.mse_loss(out_box_max[:, 4], gt_box_max[:, 4], size_average=False)
        conf_loss = F.mse_loss(out_box_max[:, 4], box_IoU_target[:, 4], size_average=False)
        #loc_loss = F.mse_loss(out_box_max[:, :2], gt_box_max[:, :2], size_average=False) + F.mse_loss(torch.sqrt(out_box_max[:, 2:4]), torch.sqrt(gt_box_max[:, 2:4]), size_average=False)
        loc_loss = F.mse_loss(out_box_max[:, :2], gt_box_max[:, :2], size_average=False) + F.mse_loss(out_box_max[:, 2:4], gt_box_max[:, 2:4], size_average=False)

        # Loss for box not responsible for the prediction
        out_box_other = out_box[co_box_mask].view(-1, 5)
        gt_box_other = gt_box[co_box_mask].view(-1, 5)
        no_box_conf_loss = F.mse_loss(out_box_other[:, 4], gt_box_other[:, 4], size_average=False)

        #Classification loss
        class_loss = F.mse_loss(out_class, gt_class, size_average=False)

        #print("No obj loss = " + str(no_loss.item()))
        #print("Loc loss = " + str(loc_loss.item()))
        #print()
        #print(box_IoU_target)
        #print("Conf loss = " + str(conf_loss.item()))

        total_loss = (self.lambda_noobj * no_loss + conf_loss + self.lambda_coord * loc_loss + self.lambda_noobj * no_box_conf_loss + self.lambda_class * class_loss) / num_sample

        return total_loss



