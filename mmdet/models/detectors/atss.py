# Copyright (c) OpenMMLab. All rights reserved.
import logging
import functools

from .l2sp_detector_mixin import L2SPDetectorMixin
from .sam_detector_mixin import SAMDetectorMixin
from ..builder import DETECTORS
from .single_stage import SingleStageDetector


logger = logging.getLogger(__name__)


@DETECTORS.register_module()
class ATSS(SingleStageDetector):
    """Implementation of `ATSS <https://arxiv.org/abs/1912.02424>`_."""

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(ATSS, self).__init__(backbone, neck, bbox_head, train_cfg,
                                   test_cfg, pretrained, init_cfg)


def map_class_names(src_classes, dst_classes):
    """Computes src to dst index mapping

    src2dst[src_idx] = dst_idx
    #  according to class name matching, -1 for non-matched ones
    assert(len(src2dst) == len(src_classes))
    ex)
      src_classes = ['person', 'car', 'tree']
      dst_classes = ['tree', 'person', 'sky', 'ball']
      -> Returns src2dst = [1, -1, 0]
    """
    src2dst = []
    for src_class in src_classes:
        if src_class in dst_classes:
            src2dst.append(dst_classes.index(src_class))
        else:
            src2dst.append(-1)
    return src2dst


@DETECTORS.register_module()
class CustomATSS(SAMDetectorMixin, L2SPDetectorMixin, ATSS):
    """SAM optimizer & L2SP regularizer enabled custom ATSS"""

    def __init__(self, *args, task_adapt=None, **kwargs):
        super().__init__(*args, **kwargs)

        # Hook for class-sensitive weight loading
        if task_adapt:
            self._register_load_state_dict_pre_hook(
                functools.partial(
                    self.load_state_dict_pre_hook,
                    self,  # model
                    task_adapt["dst_classes"],  # model_classes
                    task_adapt["src_classes"],  # chkpt_classes
                )
            )

    def forward_train(self, img, img_metas, gt_bboxes, gt_labels, gt_bboxes_ignore=None, **kwargs):
        return super().forward_train(img, img_metas, gt_bboxes, gt_labels, gt_bboxes_ignore=gt_bboxes_ignore)

    @staticmethod
    def load_state_dict_pre_hook(model, model_classes, chkpt_classes, chkpt_dict, prefix, *args, **kwargs):
        """Modify input state_dict according to class name matching before weight loading"""
        logger.info(f"----------------- CustomATSS.load_state_dict_pre_hook() called w/ prefix: {prefix}")

        # Dst to src mapping index
        model_classes = list(model_classes)
        chkpt_classes = list(chkpt_classes)
        model2chkpt = map_class_names(model_classes, chkpt_classes)
        logger.info(f"{chkpt_classes} -> {model_classes} ({model2chkpt})")

        model_dict = model.state_dict()
        param_names = [
            "bbox_head.atss_cls.weight",
            "bbox_head.atss_cls.bias",
        ]
        for model_name in param_names:
            chkpt_name = prefix + model_name
            if model_name not in model_dict or chkpt_name not in chkpt_dict:
                logger.info(f"Skipping weight copy: {chkpt_name}")
                continue

            # Mix weights
            model_param = model_dict[model_name].clone()
            chkpt_param = chkpt_dict[chkpt_name]
            for m, c in enumerate(model2chkpt):
                if c >= 0:
                    model_param[m].copy_(chkpt_param[c])

            # Replace checkpoint weight by mixed weights
            chkpt_dict[chkpt_name] = model_param
