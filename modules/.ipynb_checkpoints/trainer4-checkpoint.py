import os
from abc import abstractmethod
import torch.nn as nn
import time
import torch
import pandas as pd
from numpy import inf
from torch.autograd import Variable
import logging
import torch.nn.functional as F
import numpy as np
from torch.optim.lr_scheduler import LambdaLR
from nltk import ngrams
from nltk.translate.bleu_score import sentence_bleu
from sklearn.feature_extraction.text import TfidfVectorizer

import logging
def get_logger(dirname,  filename, verbosity=1):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "%(message)s"
    )
    logger = logging.getLogger(filename)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(dirname+filename, "a")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


class BaseTrainer(object):
    def __init__(self, model, criterion, metric_ftns, optimizer, args):
        self.args = args

        # setup GPU device if available, move model into configured device
        self.device, device_ids = self._prepare_device(args.n_gpu)
        self.model = model.to(self.device)
        if len(device_ids) > 1:
            self.model = torch.nn.DataParallel(model, device_ids=device_ids)

        self.criterion = criterion
        self.bce_criterion = self._init_bce_criterion()
        self.bce_log_criterion = self._init_bce_log_criterion()
        self.metric_ftns = metric_ftns
        self.optimizer = optimizer

        self.epochs = self.args.epochs
        self.save_period = self.args.save_period

        self.mnt_mode = args.monitor_mode
        self.mnt_metric = 'val_' + args.monitor_metric
        self.mnt_metric_test = 'test_' + args.monitor_metric
        assert self.mnt_mode in ['min', 'max']

        self.mnt_best = inf if self.mnt_mode == 'min' else -inf
        self.early_stop = getattr(self.args, 'early_stop', inf)

        self.start_epoch = 1
        self.checkpoint_dir = args.save_dir

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        if args.resume is not None:
            self._resume_checkpoint(args.resume)

        self.best_recorder = {'val': {self.mnt_metric: self.mnt_best},
                              'test': {self.mnt_metric_test: self.mnt_best}}

    @staticmethod
    def _init_bce_criterion():
        # return F.cross_entropy()
        return nn.BCELoss()
    def _init_bce_log_criterion(self):
        return nn.BCEWithLogitsLoss()

    def label_smoothed_nll_loss(self, lprobs, target, eps=0.1):
        nll_loss = -lprobs.gather(dim=-1, index=target.unsqueeze(-1))
        smooth_loss = -lprobs.mean(dim=-1)
        loss = (1.0 - eps) * nll_loss + eps * smooth_loss
        return loss.sum()

    def _to_var(self, x, requires_grad=False):
        x = x.to(self.device)
        return Variable(x, requires_grad=requires_grad)

    def composite_loss(self, predicted_indices, target_indices, top_n=1):
        """
        计算综合损失函数，考虑命中率、命中Top N和Jaccard相似度

        参数:
            predicted_indices (torch.Tensor): 模型生成的文本索引 [batch_size, word_nums]
            target_indices (torch.Tensor): 主题词索引 [batch_size, topic_nums]
            top_n (int): 命中率和命中Top N考虑的前N个位置，默认为1

        返回:
            loss (torch.Tensor): 综合损失
        """
        batch_size, _ = predicted_indices.size()

        total_loss = 0.0

        for i in range(batch_size):
            # 取出当前实例的文本索引和主题词索引
            pred_indices_i = predicted_indices[i].clone().detach()
            target_indices_i = torch.tensor(target_indices[i]).clone().detach().to(self.device)
            # print(pred_indices_i.shape, target_indices_i.shape)
            word_nums_i = len(pred_indices_i)
            topic_nums_i = len(target_indices_i)


            # 扩展主题词索引的维度，使其与生成文本索引的维度相匹配
            target_indices_expanded_i = target_indices_i.unsqueeze(0).expand(word_nums_i, topic_nums_i)
            # print(target_indices_expanded_i.shape)
            # 计算命中率
            hits_i = torch.any(pred_indices_i.unsqueeze(-1).expand(word_nums_i, top_n) == target_indices_expanded_i,
                               dim=-1)
            hit_rate_i = torch.mean(hits_i.float())

            # # 计算命中Top N
            # hits_top_n_i = torch.all(pred_indices_i[:top_n].unsqueeze(-1) == target_indices_expanded_i, dim=-1)
            # hit_rate_top_n_i = torch.mean(hits_top_n_i.float())

            # 计算Jaccard相似度
            intersection_i = torch.sum(torch.logical_and(
                pred_indices_i.unsqueeze(-1).expand(word_nums_i, topic_nums_i) == target_indices_expanded_i,
                target_indices_expanded_i != 0), dim=-1)
            union_i = torch.sum(torch.logical_or(pred_indices_i.unsqueeze(-1).expand(word_nums_i, topic_nums_i) != 0,
                                                 target_indices_expanded_i != 0), dim=-1)
            jaccard_similarity_i = intersection_i.float() / union_i.float()

            # 计算平均Jaccard相似度
            avg_jaccard_similarity_i = torch.mean(jaccard_similarity_i)

            # 计算当前实例的综合损失
            loss_i = 1.0 - hit_rate_i + (1.0 - avg_jaccard_similarity_i)

            # 累加到总损失
            total_loss += loss_i

        # 取平均
        loss = total_loss / batch_size

        return loss

    def _calculate_accuracy(self, tags, labels):
        accuracy_th = 0.5
        pred_result = tags > accuracy_th
        pred_result = pred_result.float()
        TP = torch.sum(torch.logical_and(torch.eq(labels, 1), torch.eq(pred_result, 1)))
        FP = torch.sum(torch.logical_and(torch.eq(labels, 0), torch.eq(pred_result, 1)))
        FN = torch.sum(torch.logical_and(torch.eq(labels, 1), torch.eq(pred_result, 0)))
        TN = torch.sum(torch.logical_and(torch.eq(labels, 0), torch.eq(pred_result, 0)))
        AUC = (TP + TN) / (TP + FP + FN + TN)
        P = TP / (TP + FP)
        R = TP / (TP + FN)
        F1 = 2 * P * R / (P + R)
        pred_one_num = torch.sum(pred_result)

        TP_0 = torch.sum(torch.logical_and(torch.eq(labels[:, 0], 1), torch.eq(pred_result[:, 0], 1))).item()
        FP_0 = torch.sum(torch.logical_and(torch.eq(labels[:, 0], 0), torch.eq(pred_result[:, 0], 1))).item()
        FN_0 = torch.sum(torch.logical_and(torch.eq(labels[:, 0], 1), torch.eq(pred_result[:, 0], 0))).item()
        TN_0 = torch.sum(torch.logical_and(torch.eq(labels[:, 0], 0), torch.eq(pred_result[:, 0], 0))).item()

        TP_1 = torch.sum(torch.logical_and(torch.eq(labels[:, 1], 1), torch.eq(pred_result[:, 1], 1))).item()
        FP_1 = torch.sum(torch.logical_and(torch.eq(labels[:, 1], 0), torch.eq(pred_result[:, 1], 1))).item()
        FN_1 = torch.sum(torch.logical_and(torch.eq(labels[:, 1], 1), torch.eq(pred_result[:, 1], 0))).item()
        TN_1 = torch.sum(torch.logical_and(torch.eq(labels[:, 1], 0), torch.eq(pred_result[:, 1], 0))).item()

        TP_2 = torch.sum(torch.logical_and(torch.eq(labels[:, 2], 1), torch.eq(pred_result[:, 2], 1))).item()
        FP_2 = torch.sum(torch.logical_and(torch.eq(labels[:, 2], 0), torch.eq(pred_result[:, 2], 1))).item()
        FN_2 = torch.sum(torch.logical_and(torch.eq(labels[:, 2], 1), torch.eq(pred_result[:, 2], 0))).item()
        TN_2 = torch.sum(torch.logical_and(torch.eq(labels[:, 2], 0), torch.eq(pred_result[:, 2], 0))).item()
        
        TP_3 = torch.sum(torch.logical_and(torch.eq(labels[:, 3], 1), torch.eq(pred_result[:, 3], 1))).item()
        FP_3 = torch.sum(torch.logical_and(torch.eq(labels[:, 3], 0), torch.eq(pred_result[:, 3], 1))).item()
        FN_3 = torch.sum(torch.logical_and(torch.eq(labels[:, 3], 1), torch.eq(pred_result[:, 3], 0))).item()
        TN_3 = torch.sum(torch.logical_and(torch.eq(labels[:, 3], 0), torch.eq(pred_result[:, 3], 0))).item()
        
        TP_4 = torch.sum(torch.logical_and(torch.eq(labels[:, 4], 1), torch.eq(pred_result[:, 4], 1))).item()
        FP_4 = torch.sum(torch.logical_and(torch.eq(labels[:, 4], 0), torch.eq(pred_result[:, 4], 1))).item()
        FN_4 = torch.sum(torch.logical_and(torch.eq(labels[:, 4], 1), torch.eq(pred_result[:, 4], 0))).item()
        TN_4 = torch.sum(torch.logical_and(torch.eq(labels[:, 4], 0), torch.eq(pred_result[:, 4], 0))).item()
        
        TP_5 = torch.sum(torch.logical_and(torch.eq(labels[:, 5], 1), torch.eq(pred_result[:, 5], 1))).item()
        FP_5 = torch.sum(torch.logical_and(torch.eq(labels[:, 5], 0), torch.eq(pred_result[:, 5], 1))).item()
        FN_5 = torch.sum(torch.logical_and(torch.eq(labels[:, 5], 1), torch.eq(pred_result[:, 5], 0))).item()
        TN_5 = torch.sum(torch.logical_and(torch.eq(labels[:, 5], 0), torch.eq(pred_result[:, 5], 0))).item()
        
        TP_6 = torch.sum(torch.logical_and(torch.eq(labels[:, 6], 1), torch.eq(pred_result[:, 6], 1))).item()
        FP_6 = torch.sum(torch.logical_and(torch.eq(labels[:, 6], 0), torch.eq(pred_result[:, 6], 1))).item()
        FN_6 = torch.sum(torch.logical_and(torch.eq(labels[:, 6], 1), torch.eq(pred_result[:, 6], 0))).item()
        TN_6 = torch.sum(torch.logical_and(torch.eq(labels[:, 6], 0), torch.eq(pred_result[:, 6], 0))).item()
        
        TP_7 = torch.sum(torch.logical_and(torch.eq(labels[:, 7], 1), torch.eq(pred_result[:, 7], 1))).item()
        FP_7 = torch.sum(torch.logical_and(torch.eq(labels[:, 7], 0), torch.eq(pred_result[:, 7], 1))).item()
        FN_7 = torch.sum(torch.logical_and(torch.eq(labels[:, 7], 1), torch.eq(pred_result[:, 7], 0))).item()
        TN_7 = torch.sum(torch.logical_and(torch.eq(labels[:, 7], 0), torch.eq(pred_result[:, 7], 0))).item()
        
        TP_8 = torch.sum(torch.logical_and(torch.eq(labels[:, 8], 1), torch.eq(pred_result[:, 8], 1))).item()
        FP_8 = torch.sum(torch.logical_and(torch.eq(labels[:, 8], 0), torch.eq(pred_result[:, 8], 1))).item()
        FN_8 = torch.sum(torch.logical_and(torch.eq(labels[:, 8], 1), torch.eq(pred_result[:, 8], 0))).item()
        TN_8 = torch.sum(torch.logical_and(torch.eq(labels[:, 8], 0), torch.eq(pred_result[:, 8], 0))).item()
        
        TP_9 = torch.sum(torch.logical_and(torch.eq(labels[:, 9], 1), torch.eq(pred_result[:, 9], 1))).item()
        FP_9 = torch.sum(torch.logical_and(torch.eq(labels[:, 9], 0), torch.eq(pred_result[:, 9], 1))).item()
        FN_9 = torch.sum(torch.logical_and(torch.eq(labels[:, 9], 1), torch.eq(pred_result[:, 9], 0))).item()
        TN_9 = torch.sum(torch.logical_and(torch.eq(labels[:, 9], 0), torch.eq(pred_result[:, 9], 0))).item()
        
        TP_10 = torch.sum(torch.logical_and(torch.eq(labels[:, 10], 1), torch.eq(pred_result[:, 10], 1))).item()
        FP_10 = torch.sum(torch.logical_and(torch.eq(labels[:, 10], 0), torch.eq(pred_result[:, 10], 1))).item()
        FN_10 = torch.sum(torch.logical_and(torch.eq(labels[:, 10], 1), torch.eq(pred_result[:, 10], 0))).item()
        TN_10 = torch.sum(torch.logical_and(torch.eq(labels[:, 10], 0), torch.eq(pred_result[:, 10], 0))).item()
        
        TP_11 = torch.sum(torch.logical_and(torch.eq(labels[:, 11], 1), torch.eq(pred_result[:, 11], 1))).item()
        FP_11 = torch.sum(torch.logical_and(torch.eq(labels[:, 11], 0), torch.eq(pred_result[:, 11], 1))).item()
        FN_11 = torch.sum(torch.logical_and(torch.eq(labels[:, 11], 1), torch.eq(pred_result[:, 11], 0))).item()
        TN_11 = torch.sum(torch.logical_and(torch.eq(labels[:, 11], 0), torch.eq(pred_result[:, 11], 0))).item()
        
        TP_12 = torch.sum(torch.logical_and(torch.eq(labels[:, 12], 1), torch.eq(pred_result[:, 12], 1))).item()
        FP_12 = torch.sum(torch.logical_and(torch.eq(labels[:, 12], 0), torch.eq(pred_result[:, 12], 1))).item()
        FN_12 = torch.sum(torch.logical_and(torch.eq(labels[:, 12], 1), torch.eq(pred_result[:, 12], 0))).item()
        TN_12 = torch.sum(torch.logical_and(torch.eq(labels[:, 12], 0), torch.eq(pred_result[:, 12], 0))).item()
        
        TP_13 = torch.sum(torch.logical_and(torch.eq(labels[:, 13], 1), torch.eq(pred_result[:, 13], 1))).item()
        FP_13 = torch.sum(torch.logical_and(torch.eq(labels[:, 13], 0), torch.eq(pred_result[:, 13], 1))).item()
        FN_13 = torch.sum(torch.logical_and(torch.eq(labels[:, 13], 1), torch.eq(pred_result[:, 13], 0))).item()
        TN_13 = torch.sum(torch.logical_and(torch.eq(labels[:, 13], 0), torch.eq(pred_result[:, 13], 0))).item()

        # if pred_one_num == 0:
        #     return 0, 0, 0, 0
        return AUC.item(), P.item(), R.item(), F1.item(), (TP_0+TN_0)/(TP_0+FP_0+FN_0+TN_0), (TP_1+TN_1)/(TP_1+FP_1+FN_1+TN_1),\
               (TP_2+TN_2)/(TP_2+FP_2+FN_2+TN_2), (TP_3+TN_3)/(TP_3+FP_3+FN_3+TN_3), (TP_4+TN_4)/(TP_4+FP_4+FN_4+TN_4),\
               (TP_5+TN_5) / (TP_5+FP_5+FN_5+TN_5),(TP_6+TN_6)/(TP_6+FP_6+FN_6+TN_6),(TP_7+TN_7)/(TP_7+FP_7+FN_7+TN_7),\
               (TP_8+TN_8) / (TP_8+FP_8+FN_8+TN_8),(TP_9+TN_9)/(TP_9+FP_9+FN_9+TN_9),(TP_10+TN_10)/(TP_10+FP_10+FN_10+TN_10),\
               (TP_11+TN_11) / (TP_11+FP_11+FN_11+TN_11),(TP_12+TN_12)/(TP_12+FP_12+FN_12+TN_12),(TP_13+TN_13)/(TP_13+FP_13+FN_13+TN_13)

    @abstractmethod
    def _train_epoch(self, epoch):
        raise NotImplementedError

    def train(self):
        log_dir = self.args.log_dir
        if self.args.dataset_name == 'iu_xray':
            log_file = 'iu_xray.log'
        elif self.args.dataset_name == 'mimic_cxr':
            log_file = 'mimic_cxr.log'
        else:
            log_file = 'cov_ctr.log'
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        logger = get_logger(dirname=log_dir, filename=log_file)
        not_improved_count = 0
        for epoch in range(self.start_epoch, self.epochs + 1):
            result = self._train_epoch(epoch)
            # save logged informations into log dict
            log = {'epoch': epoch}
            log.update(result)
            logger.info(log)
            self._record_best(log)

            # print logged informations to the screen
            for key, value in log.items():
                print('\t{:15s}: {}'.format(str(key), value))

            # evaluate model performance according to configured metric, save best checkpoint as model_best
            best = False
            if self.mnt_mode != 'off':
                try:
                    # check whether model performance improved or not, according to specified metric(mnt_metric)
                    improved = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.mnt_best) or \
                               (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.mnt_best)
                except KeyError:
                    print("Warning: Metric '{}' is not found. " "Model performance monitoring is disabled.".format(
                        self.mnt_metric))
                    self.mnt_mode = 'off'
                    improved = False

                if improved:
                    self.mnt_best = log[self.mnt_metric]
                    not_improved_count = 0
                    best = True
                else:
                    not_improved_count += 1

                if not_improved_count > self.early_stop:
                    print("Validation performance didn\'t improve for {} epochs. " "Training stops.".format(
                        self.early_stop))
                    break

            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch, save_best=best)
        self._print_best()
        self._print_best_to_file()

    def _print_best_to_file(self):
        crt_time = time.asctime(time.localtime(time.time()))
        self.best_recorder['val']['time'] = crt_time
        self.best_recorder['test']['time'] = crt_time
        self.best_recorder['val']['seed'] = self.args.seed
        self.best_recorder['test']['seed'] = self.args.seed
        self.best_recorder['val']['best_model_from'] = 'val'
        self.best_recorder['test']['best_model_from'] = 'test'

        if not os.path.exists(self.args.record_dir):
            os.makedirs(self.args.record_dir)
        record_path = os.path.join(self.args.record_dir, self.args.dataset_name+'.csv')
        if not os.path.exists(record_path):
            record_table = pd.DataFrame()
        else:
            record_table = pd.read_csv(record_path)
        record_table = record_table._append(self.best_recorder['val'], ignore_index=True)
        record_table = record_table._append(self.best_recorder['test'], ignore_index=True)
        record_table.to_csv(record_path, index=False)

    def _prepare_device(self, n_gpu_use):
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            print("Warning: There\'s no GPU available on this machine," "training will be performed on CPU.")
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            print(
                "Warning: The number of GPU\'s configured to use is {}, but only {} are available " "on this machine.".format(
                    n_gpu_use, n_gpu))
            n_gpu_use = n_gpu
        device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
        list_ids = list(range(n_gpu_use))
        return device, list_ids

    def _save_checkpoint(self, epoch, save_best=False):
        state = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.mnt_best
        }
        filename = os.path.join(self.checkpoint_dir, 'current_checkpoint.pth')
        torch.save(state, filename)
        print("Saving checkpoint: {} ...".format(filename))
        if save_best:
            best_path = os.path.join(self.checkpoint_dir, 'model_best.pth')
            torch.save(state, best_path)
            print("Saving current best: model_best.pth ...")

    def _resume_checkpoint(self, resume_path):
        resume_path = str(resume_path)
        print("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

        print("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))

    def _record_best(self, log):
        improved_val = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.best_recorder['val'][
            self.mnt_metric]) or \
                       (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.best_recorder['val'][self.mnt_metric])
        if improved_val:
            self.best_recorder['val'].update(log)

        improved_test = (self.mnt_mode == 'min' and log[self.mnt_metric_test] <= self.best_recorder['test'][
            self.mnt_metric_test]) or \
                        (self.mnt_mode == 'max' and log[self.mnt_metric_test] >= self.best_recorder['test'][
                            self.mnt_metric_test])
        if improved_test:
            self.best_recorder['test'].update(log)

    def _print_best(self):
        print('Best results (w.r.t {}) in validation set:'.format(self.args.monitor_metric))
        for key, value in self.best_recorder['val'].items():
            print('\t{:15s}: {}'.format(str(key), value))

        print('Best results (w.r.t {}) in test set:'.format(self.args.monitor_metric))
        for key, value in self.best_recorder['test'].items():
            print('\t{:15s}: {}'.format(str(key), value))


class Trainer(BaseTrainer):
    def __init__(self, model, criterion, metric_ftns, optimizer, args, lr_scheduler, train_dataloader, val_dataloader,
                 test_dataloader):
        super(Trainer, self).__init__(model, criterion, metric_ftns, optimizer, args)
        self.lr_scheduler = lr_scheduler
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.save_dir = args.save_dir
        # 初始权重值，可以根据需要进行调整
        self.weight_train_loss = nn.Parameter(torch.tensor(5.0, requires_grad=True))
        self.weight_tag_loss = nn.Parameter(torch.tensor(1.0, requires_grad=True))
        self.weight_tag_embed_pred_loss = nn.Parameter(torch.tensor(1.0, requires_grad=True))
        self.weight_tag_sim_loss = nn.Parameter(torch.tensor(5.0, requires_grad=True))

    def _train_epoch(self, epoch):
        val_AUC_total = 0.0
        val_P_total = 0.0
        val_R_total = 0.0
        val_F1_total = 0.0
        val_auc_total_00, val_auc_total_01, val_auc_total_02, val_auc_total_03, val_auc_total_04, val_auc_total_05, val_auc_total_06, val_auc_total_07, val_auc_total_08, val_auc_total_09, val_auc_total_10, val_auc_total_11, val_auc_total_12, val_auc_total_13 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        test_AUC_total = 0.0
        test_P_total = 0.0
        test_R_total = 0.0
        test_F1_total = 0.0
        test_auc_total_00, test_auc_total_01, test_auc_total_02, test_auc_total_03, test_auc_total_04, test_auc_total_05, test_auc_total_06, test_auc_total_07, test_auc_total_08, test_auc_total_09, test_auc_total_10, test_auc_total_11, test_auc_total_12, test_auc_total_13 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        train_total_loss = 0.0
        train_tag_loss = 0.0
        train_tag_pred_loss = 0.0

        self.model.train()
        for batch_idx, (images_id, images, reports_ids, reports_masks, labels, tags_id, tags_index) in enumerate(self.train_dataloader):
            images, reports_ids, reports_masks, labels, tags_id, tags_index = images.to(self.device), reports_ids.to(self.device), reports_masks.to(
                self.device), labels.to(self.device), tags_id.to(self.device), tags_index.to(self.device)

            output, class_pred, tags_pred, tag_embed, tag_embed_pred = self.model(images, labels, reports_ids, tags_id, mode='train')

            # # 训练阶段使用贪婪解码，选择概率最高的词作为当前步的输出
            # _, predicted_indices = torch.max(output, dim=-1)
            # _, pred_tag_indices = torch.max(tag_vocab, dim=-1)
            # print(predicted_indices.shape)
            #print(output.shape, reports_ids.shape)
            train_loss = self.criterion(output, reports_ids, reports_masks)
            tag_loss = self.bce_criterion(tags_pred, self._to_var(tags_id.to(torch.float), requires_grad=False))
            tag_embed_pred_loss = self.bce_log_criterion(tag_embed, self._to_var((tag_embed_pred.to(torch.float)), requires_grad=False))
            #print(predicted_indices.shape, pred_tag_indices.shape)
            #tag_sim_loss = self.label_smoothed_nll_loss(pred_tag_indices.to(torch.float), tags_index.to(torch.float))

            train_total_loss += train_loss.item()
            train_tag_loss += tag_loss.item()
            train_tag_pred_loss += tag_embed_pred_loss.item()
            #train_tag_sim_loss += tag_sim_loss.item()

            loss = self.weight_train_loss * train_loss + self.weight_tag_loss * tag_loss \
                    + self.weight_tag_embed_pred_loss * tag_embed_pred_loss

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.model.parameters(), 0.1)
            self.optimizer.step()
            print('the batch number is: {}, current loss: {}'.format(batch_idx, loss))
        #print('train loss: {}'.format(train_loss / len(self.train_dataloader)),'category loss: {}'.format(category_loss / len(self.train_dataloader)))
        log = {'train_loss': train_total_loss / len(self.train_dataloader),
               'tag_loss': train_tag_loss / len(self.train_dataloader),
               'tag_logic_loss': train_tag_pred_loss / len(self.train_dataloader)}

        self.model.eval()
        with torch.no_grad():
            val_gts, val_res = [], []
            for batch_idx, (images_id, images, reports_ids, reports_masks, labels, tags_id, tags_index) in enumerate(
                    self.val_dataloader):
                print(f"Epoch [{epoch}], Iteration [{batch_idx}]")
                images, reports_ids, reports_masks, labels, tags_id, tags_index = images.to(
                    self.device), reports_ids.to(self.device), reports_masks.to(
                    self.device), labels.to(self.device), tags_id.to(self.device), tags_index.to(self.device)
                output, class_pred, tags_pred = self.model(images, labels, mode='sample')
                reports = self.model.tokenizer.decode_batch(output.cpu().numpy())

                ground_truths = self.model.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
                val_res.extend(reports)
                val_gts.extend(ground_truths)
                AUC, P, R, F1, AUC_0, AUC_1, AUC_2, AUC_3, AUC_4, AUC_5, AUC_6, AUC_7, AUC_8, AUC_9\
                    , AUC_10, AUC_11, AUC_12, AUC_13 = self._calculate_accuracy(class_pred, labels)
                val_AUC_total = val_AUC_total + AUC
                val_P_total = val_P_total + P
                val_R_total = val_R_total + R
                val_F1_total = val_F1_total + F1
                val_auc_total_00 = val_auc_total_00 + AUC_0
                val_auc_total_01 = val_auc_total_01 + AUC_1
                val_auc_total_02 = val_auc_total_02 + AUC_2
                val_auc_total_03 = val_auc_total_03 + AUC_3
                val_auc_total_04 = val_auc_total_04 + AUC_4
                val_auc_total_05 = val_auc_total_05 + AUC_5
                val_auc_total_06 = val_auc_total_06 + AUC_6
                val_auc_total_07 = val_auc_total_07 + AUC_7
                val_auc_total_08 = val_auc_total_08 + AUC_8
                val_auc_total_09 = val_auc_total_09 + AUC_9
                val_auc_total_10 = val_auc_total_10 + AUC_10
                val_auc_total_11 = val_auc_total_11 + AUC_11
                val_auc_total_12 = val_auc_total_12 + AUC_12
                val_auc_total_13 = val_auc_total_13 + AUC_13
            val_met = self.metric_ftns({i: [gt] for i, gt in enumerate(val_gts)},
                                       {i: [re] for i, re in enumerate(val_res)})

            log.update(**{'val_' + k: v for k, v in val_met.items()})
            log.update({'val_AUC_total': val_AUC_total / len(self.val_dataloader)})
            log.update({'val_P_total': val_P_total / len(self.val_dataloader)})
            log.update({'val_R_total': val_R_total / len(self.val_dataloader)})
            log.update({'val_F1_total': val_F1_total / len(self.val_dataloader)})
            log.update({'val_auc_total_00': val_auc_total_00 / len(self.val_dataloader)})
            log.update({'val_auc_total_01': val_auc_total_01 / len(self.val_dataloader)})
            log.update({'val_auc_total_02': val_auc_total_02 / len(self.val_dataloader)})
            log.update({'val_auc_total_03': val_auc_total_03 / len(self.val_dataloader)})
            log.update({'val_auc_total_04': val_auc_total_04 / len(self.val_dataloader)})
            log.update({'val_auc_total_05': val_auc_total_05 / len(self.val_dataloader)})
            log.update({'val_auc_total_06': val_auc_total_06 / len(self.val_dataloader)})
            log.update({'val_auc_total_07': val_auc_total_07 / len(self.val_dataloader)})
            log.update({'val_auc_total_08': val_auc_total_08 / len(self.val_dataloader)})
            log.update({'val_auc_total_09': val_auc_total_09 / len(self.val_dataloader)})
            log.update({'val_auc_total_10': val_auc_total_10 / len(self.val_dataloader)})
            log.update({'val_auc_total_11': val_auc_total_11 / len(self.val_dataloader)})
            log.update({'val_auc_total_12': val_auc_total_12 / len(self.val_dataloader)})
            log.update({'val_auc_total_13': val_auc_total_13 / len(self.val_dataloader)})

            val_met = self.metric_ftns({i: [gt] for i, gt in enumerate(val_gts)},
                                       {i: [re] for i, re in enumerate(val_res)})
            log.update(**{'val_' + k: v for k, v in val_met.items()})

        self.model.eval()
        with torch.no_grad():
            test_gts, test_res = [], []
            for batch_idx, (images_id, images, reports_ids, reports_masks, labels, tags_id, tags_index) in enumerate(
                    self.test_dataloader):
                print(f"Epoch [{epoch}], Iteration [{batch_idx}]")
                images, reports_ids, reports_masks, labels, tags_id, tags_index = images.to(
                    self.device), reports_ids.to(self.device), reports_masks.to(
                    self.device), labels.to(self.device), tags_id.to(self.device), tags_index.to(self.device)
                output, class_pred, tags_pred = self.model(images, labels, mode='sample')
                reports = self.model.tokenizer.decode_batch(output.cpu().numpy())
                ground_truths = self.model.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
                # if epoch == 1:
                #     print(ground_truths)
                #     print("-----------------------------------")
                # print(reports)
                test_res.extend(reports)
                test_gts.extend(ground_truths)
                AUC, P, R, F1, AUC_0, AUC_1, AUC_2, AUC_3, AUC_4, AUC_5, AUC_6, AUC_7, AUC_8, AUC_9, \
                AUC_10, AUC_11, AUC_12, AUC_13 = self._calculate_accuracy(class_pred, labels)
                test_AUC_total = test_AUC_total + AUC
                test_P_total = test_P_total + P
                test_R_total = test_R_total + R
                test_F1_total = test_F1_total + F1
                test_auc_total_00 = test_auc_total_00 + AUC_0
                test_auc_total_01 = test_auc_total_01 + AUC_1
                test_auc_total_02 = test_auc_total_02 + AUC_2
                test_auc_total_03 = test_auc_total_03 + AUC_3
                test_auc_total_04 = test_auc_total_04 + AUC_4
                test_auc_total_05 = test_auc_total_05 + AUC_5
                test_auc_total_06 = test_auc_total_06 + AUC_6
                test_auc_total_07 = test_auc_total_07 + AUC_7
                test_auc_total_08 = test_auc_total_08 + AUC_8
                test_auc_total_09 = test_auc_total_09 + AUC_9
                test_auc_total_10 = test_auc_total_10 + AUC_10
                test_auc_total_11 = test_auc_total_11 + AUC_11
                test_auc_total_12 = test_auc_total_12 + AUC_12
                test_auc_total_13 = test_auc_total_13 + AUC_13
                #print("----------------------------------------------")

            test_met = self.metric_ftns({i: [gt] for i, gt in enumerate(test_gts)},
                                        {i: [re] for i, re in enumerate(test_res)})
            log.update(**{'test_' + k: v for k, v in test_met.items()})
            log.update({'test_AUC_total': test_AUC_total / len(self.test_dataloader)})
            log.update({'test_P_total': test_P_total / len(self.test_dataloader)})
            log.update({'test_R_total': test_R_total / len(self.test_dataloader)})
            log.update({'test_F1_total': test_F1_total / len(self.test_dataloader)})
            log.update({'test_auc_total_00': test_auc_total_00 / len(self.test_dataloader)})
            log.update({'test_auc_total_01': test_auc_total_01 / len(self.test_dataloader)})
            log.update({'test_auc_total_02': test_auc_total_02 / len(self.test_dataloader)})
            log.update({'test_auc_total_03': test_auc_total_03 / len(self.test_dataloader)})
            log.update({'test_auc_total_04': test_auc_total_04 / len(self.test_dataloader)})
            log.update({'test_auc_total_05': test_auc_total_05 / len(self.test_dataloader)})
            log.update({'test_auc_total_06': test_auc_total_06 / len(self.test_dataloader)})
            log.update({'test_auc_total_07': test_auc_total_07 / len(self.test_dataloader)})
            log.update({'test_auc_total_08': test_auc_total_08 / len(self.test_dataloader)})
            log.update({'test_auc_total_09': test_auc_total_09 / len(self.test_dataloader)})
            log.update({'test_auc_total_10': test_auc_total_10 / len(self.test_dataloader)})
            log.update({'test_auc_total_11': test_auc_total_11 / len(self.test_dataloader)})
            log.update({'test_auc_total_12': test_auc_total_12 / len(self.test_dataloader)})
            log.update({'test_auc_total_13': test_auc_total_13 / len(self.test_dataloader)})

            test_met = self.metric_ftns({i: [gt] for i, gt in enumerate(test_gts)},
                                        {i: [re] for i, re in enumerate(test_res)})
            log.update(**{'test_' + k: v for k, v in test_met.items()})

            test_res, test_gts = pd.DataFrame(test_res), pd.DataFrame(test_gts)

            save_res_name = str(epoch)+"_res.csv"
            save_gts_name = str(epoch) + "_gts.csv"
            test_res.to_csv(os.path.join(self.save_dir, save_res_name), index=False, header=False)
            if epoch==1:
                test_gts.to_csv(os.path.join(self.save_dir, save_gts_name), index=False, header=False)

        self.lr_scheduler.step()

        return log
