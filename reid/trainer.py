from __future__ import print_function, absolute_import
import time

from torch.nn import functional as F
import torch
import torch.nn as nn
from .utils.meters import AverageMeter
from .utils.feature_tools import *

from reid.utils.make_loss import make_loss
import copy

from reid.metric_learning.distance import cosine_similarity
class Trainer(object):
    def __init__(self,cfg,args, model, num_classes, writer=None):
        super(Trainer, self).__init__()
        self.cfg = cfg
        self.args = args
        self.model = model
        self.writer = writer
        self.AF_weight = args.AF_weight

        self.loss_fn, center_criterion = make_loss(cfg, num_classes=num_classes)

      
        self.KLDivLoss = nn.KLDivLoss(reduction='batchmean')
    def train(self, epoch, data_loader_train,  optimizer, training_phase,
              train_iters=200, add_num=0, old_model=None,         
              ):

        self.model.train()
        # freeze the bn layer totally
        for m in self.model.module.base.modules():
            if isinstance(m, nn.BatchNorm2d):
                if m.weight.requires_grad == False and m.bias.requires_grad == False:
                    m.eval()
        
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses_ce = AverageMeter()
        losses_tr = AverageMeter()

        end = time.time()

        for i in range(train_iters):
            train_inputs = data_loader_train.next()
            data_time.update(time.time() - end)

            s_inputs, targets, cids, domains, = self._parse_data(train_inputs)
            targets += add_num
            s_features, bn_feat, cls_outputs, feat_final_layer = self.model(s_inputs)

            '''calculate the base loss'''
            loss_ce, loss_tp = self.loss_fn(cls_outputs, s_features, targets, target_cam=None)
            loss = loss_ce + loss_tp
            losses_ce.update(loss_ce.item())
            losses_tr.update(loss_tp.item())

            if old_model is not None:
                with torch.no_grad():
                    s_features_old, bn_feat_old, cls_outputs_old, feat_final_layer_old = old_model(s_inputs, get_all_feat=True)
                if isinstance(s_features_old, tuple):
                    s_features_old=s_features_old[0]
                Affinity_matrix_new = self.get_normal_affinity(s_features)
                Affinity_matrix_old = self.get_normal_affinity(s_features_old)
                divergence = self.cal_KL(Affinity_matrix_new, Affinity_matrix_old, targets)
                loss = loss + divergence * self.AF_weight
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()           

            batch_time.update(time.time() - end)
            end = time.time()
            if self.writer != None :
                self.writer.add_scalar(tag="loss/Loss_ce_{}".format(training_phase), scalar_value=losses_ce.val,
                          global_step=epoch * train_iters + i)
                self.writer.add_scalar(tag="loss/Loss_tr_{}".format(training_phase), scalar_value=losses_tr.val,
                          global_step=epoch * train_iters + i)

                self.writer.add_scalar(tag="time/Time_{}".format(training_phase), scalar_value=batch_time.val,
                          global_step=epoch * train_iters + i)
            if (i + 1) == train_iters:
            #if 1 :
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Loss_ce {:.3f} ({:.3f})\t'
                      'Loss_tp {:.3f} ({:.3f})\t'
                      .format(epoch, i + 1, train_iters,
                              batch_time.val, batch_time.avg,
                              losses_ce.val, losses_ce.avg,
                              losses_tr.val, losses_tr.avg,
                  ))       

    def get_normal_affinity(self,x,Norm=0.1):
        pre_matrix_origin=cosine_similarity(x,x)
        pre_affinity_matrix=F.softmax(pre_matrix_origin/Norm, dim=1)
        return pre_affinity_matrix
    def _parse_data(self, inputs):
        imgs, _, pids, cids, domains = inputs
        inputs = imgs.cuda()
        targets = pids.cuda()
        return inputs, targets, cids, domains
    def cal_KL(self,Affinity_matrix_new, Affinity_matrix_old,targets):
        Gts = (targets.reshape(-1, 1) - targets.reshape(1, -1)) == 0  # Gt-matrix
        Gts = Gts.float().to(targets.device)
        '''obtain TP,FP,TN,FN'''
        attri_new = self.get_attri(Gts, Affinity_matrix_new, margin=0)
        attri_old = self.get_attri(Gts, Affinity_matrix_old, margin=0)

        '''# prediction is correct on old model'''
        Old_Keep = attri_old['TN'] + attri_old['TP']
        Target_1 = Affinity_matrix_old * Old_Keep
        '''# prediction is false on old model but correct on mew model'''
        New_keep = (attri_new['TN'] + attri_new['TP']) * (attri_old['FN'] + attri_old['FP'])
        Target_2 = Affinity_matrix_new * New_keep
        '''# both missed correct person'''
        Hard_pos = attri_new['FN'] * attri_old['FN']
        Thres_P = torch.maximum(attri_new['Thres_P'], attri_old['Thres_P'])
        Target_3 = Hard_pos * Thres_P

        '''# both false wrong person'''
        Hard_neg = attri_new['FP'] * attri_old['FP']
        Thres_N = torch.minimum(attri_new['Thres_N'], attri_old['Thres_N'])
        Target_4 = Hard_neg * Thres_N

        Target__ = Target_1 + Target_2 + Target_3 + Target_4
        Target = Target__ / (Target__.sum(1, keepdim=True))  # score normalization


        Affinity_matrix_new_log = torch.log(Affinity_matrix_new)
        divergence=self.KLDivLoss(Affinity_matrix_new_log, Target)

        return divergence

    def get_attri(self, Gts, pre_affinity_matrix,margin=0):
        Thres_P=((1-Gts)*pre_affinity_matrix).max(dim=1,keepdim=True)[0]
        T_scores=pre_affinity_matrix*Gts

        TP=((T_scores-Thres_P)>margin).float()
        TP=torch.maximum(TP, torch.eye(TP.size(0)).to(TP.device))

        FN=Gts-TP

        Mapped_affinity=(1-Gts) +pre_affinity_matrix
        Mapped_affinity = Mapped_affinity+torch.eye(Mapped_affinity.size(0)).to(Mapped_affinity.device)
        Thres_N = Mapped_affinity.min(dim=1, keepdim=True)[0]
        N_scores=pre_affinity_matrix*(1-Gts)

        FP=(N_scores>Thres_N ).float()
        TN=(1-Gts) -FP
        attris={
            'TP':TP,
            'FN':FN,
            'FP':FP,
            'TN':TN,
            "Thres_P":Thres_P,
            "Thres_N":Thres_N
        }
        return attris

