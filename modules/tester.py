import logging
import os
from abc import abstractmethod

import cv2
import pandas as pd
import torch

from modules.utils import generate_heatmap
from tqdm import tqdm
import torch.nn.functional as F

class BaseTester(object):
    def __init__(self, model, criterion, metric_ftns, args):
        self.args = args

        logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                            datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # setup GPU device if available, move model into configured device
        self.device, device_ids = self._prepare_device(args.n_gpu)
        self.model = model.to(self.device)
        if len(device_ids) > 1:
            self.model = torch.nn.DataParallel(model, device_ids=device_ids)

        self.criterion = criterion
        self.metric_ftns = metric_ftns

        self.epochs = self.args.epochs
        self.save_dir = self.args.save_dir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        self._load_checkpoint(args.load)

    @abstractmethod
    def test(self):
        raise NotImplementedError

    @abstractmethod
    def plot(self):
        raise NotImplementedError

    def _prepare_device(self, n_gpu_use):
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            self.logger.warning(
                "Warning: There\'s no GPU available on this machine," "training will be performed on CPU.")
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            self.logger.warning(
                "Warning: The number of GPU\'s configured to use is {}, but only {} are available " "on this machine.".format(
                    n_gpu_use, n_gpu))
            n_gpu_use = n_gpu
        device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
        list_ids = list(range(n_gpu_use))
        return device, list_ids

    def _load_checkpoint(self, load_path):
        load_path = str(load_path)
        self.logger.info("Loading checkpoint: {} ...".format(load_path))
        checkpoint = torch.load(load_path)
        self.model.load_state_dict(checkpoint['state_dict'])
        # print("权重文件中的键:")
        # for key in checkpoint['state_dict'].keys():
        #     print(key)
        # print("\n模型中的参数名:")
        # for name, param in self.model.named_parameters():
        #     print(name)


class Tester(BaseTester):
    def __init__(self, model, criterion, metric_ftns, args, test_dataloader):
        super(Tester, self).__init__(model, criterion, metric_ftns, args)
        self.test_dataloader = test_dataloader

    def test(self):
        self.logger.info('Start to evaluate in the test set.')
        log = dict()
        self.model.eval()
        with torch.no_grad():
            test_gts, test_res = [], []
            for batch_idx, (images_id, images, reports_ids, reports_masks, labels, tags_id, tags_index) in enumerate(
                    self.test_dataloader):

                images, reports_ids, reports_masks, labels, tags_id, tags_index = images.to(
                    self.device), reports_ids.to(self.device), reports_masks.to(
                    self.device), labels.to(self.device), tags_id.to(self.device), tags_index.to(self.device)
                output, class_pred, tags_pred = self.model(images, labels, mode='sample')
                probabilities = F.softmax(tags_pred, dim=1)
   
                num_classes = probabilities.size(1)
                one_hot_preds = F.one_hot(probabilities, num_classes)
                print(one_hot_preds)
                reports = self.model.tokenizer.decode_batch(output.cpu().numpy())
                ground_truths = self.model.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
                test_res.extend(reports)
                test_gts.extend(ground_truths)
            test_met = self.metric_ftns({i: [gt] for i, gt in enumerate(test_gts)},
                                        {i: [re] for i, re in enumerate(test_res)})
            log.update(**{'test_' + k: v for k, v in test_met.items()})
            print(log)

            test_res, test_gts = pd.DataFrame(test_res), pd.DataFrame(test_gts)
            test_res.to_csv(os.path.join(self.save_dir, "res.csv"), index=False, header=False)
            test_gts.to_csv(os.path.join(self.save_dir, "gts.csv"), index=False, header=False)

        return log

    def plot(self):
        assert self.args.batch_size == 1 and self.args.beam_size == 1
        self.logger.info('Start to plot attention weights in the test set.')
        os.makedirs(os.path.join(self.save_dir, "attentions"), exist_ok=True)
        mean = torch.tensor((0.485, 0.456, 0.406))
        std = torch.tensor((0.229, 0.224, 0.225))
        mean = mean[:, None, None]
        std = std[:, None, None]

        self.model.eval()
        with torch.no_grad():
            test_gts, test_res = [], []
            for batch_idx, (images_id, images, reports_ids, reports_masks, labels, tags_id, tags_index) in enumerate(
                    self.test_dataloader):
                print(f"Iteration [{batch_idx}]")
                images, reports_ids, reports_masks, labels, tags_id, tags_index = images.to(
                    self.device), reports_ids.to(self.device), reports_masks.to(
                    self.device), labels.to(self.device), tags_id.to(self.device), tags_index.to(self.device)
                output, class_pred, tags_pred = self.model(images, labels, mode='sample')
                image = torch.clamp((images[0].cpu() * std + mean) * 255, 0, 255).int().cpu().numpy()
                report = self.model.tokenizer.decode_batch(output.cpu().numpy())[0].split()
                ground_truth = self.model.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
                test_res.extend(report)
                test_gts.extend(ground_truth)

                attention_weights = [layer.src_attn.attn.cpu().numpy()[:, :, :-1].mean(0).mean(0) for layer in
                                     self.model.encoder_decoder.model.decoder.layers]
                for layer_idx, attns in enumerate(attention_weights):
                    assert len(attns) == len(report)
                    for word_idx, (attn, word) in enumerate(zip(attns, report)):
                        os.makedirs(os.path.join(self.save_dir, "attentions", "{:04d}".format(batch_idx),
                                                 "layer_{}".format(layer_idx)), exist_ok=True)

                        heatmap = generate_heatmap(image, attn)
                        cv2.imwrite(os.path.join(self.save_dir, "attentions", "{:04d}".format(batch_idx),
                                                 "layer_{}".format(layer_idx), "{:04d}_{}.png".format(word_idx, word)),
                                    heatmap)
            print(batch_idx)
            test_res, test_gts = pd.DataFrame(test_res), pd.DataFrame(test_gts)
            res_name = str(batch_idx) + "_res.csv"
            gts_name = str(batch_idx) + "_gts.csv"
            save_path = os.path.join(self.save_dir, 'reports')
            test_res.to_csv(os.path.join(save_path, res_name), index=False, header=False)
            test_gts.to_csv(os.path.join(save_path, gts_name), index=False, header=False)