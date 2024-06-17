import torch
import torch.nn as nn
import numpy as np

from modules.visual_extractor import VisualExtractor
from modules.encoder_decoder5 import EncoderDecoder

from modules.classifier import Classifier

device = torch.device('cuda:1')

avg_feats = 2048
num_classes = 14
num_gcn_layers = 3

class R2GenModel(nn.Module):
    def __init__(self, args, tokenizer):
        super(R2GenModel, self).__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.visual_extractor = VisualExtractor(args)
        self.encoder_decoder = EncoderDecoder(args, tokenizer)

        self.classify = Classifier(avg_feats, num_classes)

        self.fc_class_iu = nn.Sequential(
            nn.Linear(self.args.d_vf * 2, 14),
            nn.Sigmoid()
        )
        self.fc_class_cov_ctr = nn.Sequential(
            nn.Linear(self.args.d_vf, 14),
            nn.Sigmoid()
        )

        self.fc_tags_iu = nn.Sequential(
            nn.Linear(self.args.d_vf*2, 163),
            nn.Sigmoid()
        )

        self.fc_fuse_iu = nn.Sequential(
            nn.Linear(self.args.d_vf * 2, 14),
            nn.Sigmoid()
        )

        if args.dataset_name == 'iu_xray':
            self.forward = self.forward_iu_xray
        else:
            self.forward = self.forward_mimic_cxr

    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)

    
    def forward_mimic_cxr(self, images, labels, targets=None, tags_truth=None, mode='train'):
        att_feats, fc_feats = self.visual_extractor(images, labels, targets)
        class_pred_prob = self.fc_class_cov_ctr(fc_feats)  
        tags_pred_prob = self.fc_tags_cov_ctr(fc_feats)  

        if mode == 'train':
            output, tag_embed, tag_pred_embed= self.encoder_decoder(fc_feats, att_feats, targets, tags=tags_truth, mode='forward')
            return output, class_pred_prob, tags_pred_prob, tag_embed, tag_pred_embed
        elif mode == 'sample':
            output, _ = self.encoder_decoder(fc_feats, att_feats, tags=tags_pred_prob, mode='sample')
        else:
            raise ValueError
        return output, class_pred_prob, tags_pred_prob

