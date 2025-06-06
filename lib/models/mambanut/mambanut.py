"""
Basic MambaNUT model.
"""
import os

import torch
from torch import nn
from torch.nn.modules.transformer import _get_clones

from lib.models.layers.head import build_box_head
from lib.utils.box_ops import box_xyxy_to_cxcywh

from lib.models.mambanut.models_mamba_R import mambar_small_patch16_224

import numpy as np

import importlib
from torch.nn.functional import l1_loss


class MambaNUT(nn.Module):
    """ This is the base class for MambaNUT """

    def __init__(self, transformer, box_head, aux_loss=False, head_type="CORNER"):
        """ Initializes the model.
        Parameters:
            transformer: torch module of the transformer architecture.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.backbone = transformer
        self.box_head = box_head

        self.aux_loss = aux_loss
        self.head_type = head_type
        if head_type == "CORNER" or head_type == "CENTER":
            self.feat_sz_s = int(box_head.feat_sz)
            self.feat_sz_t = int(box_head.feat_template_sz)
            self.feat_len_s = int(box_head.feat_sz ** 2)
            self.feat_len_t = int(self.feat_sz_t ** 2)


        if self.aux_loss:
            self.box_head = _get_clones(self.box_head, 6)
        self.l1_loss = l1_loss

    def forward(self, template: torch.Tensor,
                search: torch.Tensor,
                training_dataset='',
                ):

        x = self.backbone.forward_features(z=template, x=search,
                                           inference_params=None)
        # if self.training:
        #     training_datasets = training_dataset

        # Forward head
        feat_last = x
        if isinstance(x, list):
            feat_last = x[-1]
        out = self.forward_head(feat_last, None)

        # out.update(aux_dict)
        out['backbone_feat'] = x
        out['training_datasets'] = training_dataset

        return out

    def forward_head(self, cat_feature, gt_score_map=None):
        """
        cat_feature: output embeddings of the backbone, it can be (HW1+HW2, B, C) or (HW2, B, C)
        """
        enc_opt = cat_feature[:, -self.feat_len_s:]  # encoder output for the search region (B, HW, C)
        opt = (enc_opt.unsqueeze(-1)).permute((0, 3, 2, 1)).contiguous()
        bs, Nq, C, HW = opt.size()
        opt_feat = opt.view(-1, C, self.feat_sz_s, self.feat_sz_s)

        # opt_feat = cat_feature.permute(0,3,1,2)
        # bs, Nq, C, HW = opt_feat.size()

        if self.head_type == "CORNER":
            # run the corner head
            pred_box, score_map = self.box_head(opt_feat, True)
            outputs_coord = box_xyxy_to_cxcywh(pred_box)
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map,
                   }
            return out

        elif self.head_type == "CENTER":
            # run the center head
            score_map_ctr, bbox, size_map, offset_map = self.box_head(opt_feat, gt_score_map)
            # outputs_coord = box_xyxy_to_cxcywh(bbox)
            outputs_coord = bbox
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map_ctr,
                   'size_map': size_map,
                   'offset_map': offset_map}
            return out
        else:
            raise NotImplementedError


def build_mambanut(cfg, training=True):
    current_dir = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root
    pretrained_path = os.path.join(current_dir, '../../../pretrained_models')
    if cfg.MODEL.PRETRAIN_FILE and ('MambaNUT' not in cfg.MODEL.PRETRAIN_FILE) and training:
        pretrained = os.path.join(pretrained_path, cfg.MODEL.PRETRAIN_FILE)
    else:
        pretrained = ''

    if cfg.MODEL.BACKBONE.TYPE == 'mambar_small_patch16_224':
        backbone = mambar_small_patch16_224(num_classes=0, pretrained=True)
        hidden_dim = 384
        patch_start_index = 1
    else:
        raise NotImplementedError

    if (cfg.MODEL.BACKBONE.TYPE == 'mambar_small_patch16_224'):
        pass
    else:
        backbone.finetune_track(cfg=cfg, patch_start_index=patch_start_index)

    # backbone.finetune_track(cfg=cfg, patch_start_index=patch_start_index)

    box_head = build_box_head(cfg, hidden_dim)

    model = MambaNUT(
        backbone,
        box_head,
        aux_loss=False,
        head_type=cfg.MODEL.HEAD.TYPE,
    )

    if 'MambaNUT' in cfg.MODEL.PRETRAIN_FILE and training:
        checkpoint = torch.load(cfg.MODEL.PRETRAIN_FILE, map_location="cpu")
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint["net"], strict=False)
        print('Load pretrained model from: ' + cfg.MODEL.PRETRAIN_FILE)

    return model
