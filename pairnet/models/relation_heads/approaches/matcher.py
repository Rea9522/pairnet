# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmdet.core import AssignResult, BaseAssigner, bbox_cxcywh_to_xyxy
from mmdet.core.bbox.builder import BBOX_ASSIGNERS
from mmdet.core.bbox.match_costs import build_match_cost
from scipy.optimize import linear_sum_assignment

sorted_dict = {'on': 712409, 'has': 277936, 'in': 251756, 'of': 146339, 'wearing': 136099, 'near': 96589, 'with': 66425, 'above': 47341, 'holding': 42722, 'behind': 41356, 'under': 22596, 'sitting on': 18643, 'wears': 15457, 'standing on': 14185, 'in front of': 13715, 'attached to': 10190, 'at': 9903, 'hanging from': 9894, 'over': 9317, 'for': 9145, 'riding': 8856, 'carrying': 5213, 'eating': 4688, 'walking on': 4613, 'playing': 3810, 'covering': 3806, 'laying on': 3739, 'along': 3624, 'watching': 3490, 'and': 3477, 'between': 3411, 'belonging to': 3288, 'painted on': 3095, 'against': 3092, 'looking at': 3083, 'from': 2945, 'parked on': 2721, 'to': 2517, 'made of': 2380, 'covered in': 2312, 'mounted on': 2253, 'says': 2241, 'part of': 2065, 'across': 1996, 'flying in': 1973, 'using': 1925, 'on back of': 1914, 'lying on': 1869, 'growing on': 1853, 'walking in': 1740}

@BBOX_ASSIGNERS.register_module()
class MaskHTriMatcher(BaseAssigner):
    def __init__(
        self,
        s_cls_cost=dict(type="ClassificationCost", weight=2.0),
        s_mask_cost=dict(type="BBoxL1Cost", weight=5.0),
        s_dice_cost=dict(type="DiceCost", weight=5.0, pred_act=True, eps=1.0),
        o_cls_cost=dict(type="ClassificationCost", weight=1.0),
        o_mask_cost=dict(type="BBoxL1Cost", weight=5.0),
        o_dice_cost=dict(type="DiceCost", weight=5.0, pred_act=True, eps=1.0),
        r_cls_cost=dict(type="ClassificationCost", weight=2.0),
    ):
        self.s_cls_cost = build_match_cost(s_cls_cost)
        self.s_mask_cost = build_match_cost(s_mask_cost)
        self.s_dice_cost = build_match_cost(s_dice_cost)
        self.o_cls_cost = build_match_cost(o_cls_cost)
        self.o_mask_cost = build_match_cost(o_mask_cost)
        self.o_dice_cost = build_match_cost(o_dice_cost)
        self.r_cls_cost = build_match_cost(r_cls_cost)

    def assign(
        self,
        s_cls_score,
        o_cls_score,
        r_cls_score,
        # gt_sub_bboxes,
        # gt_obj_bboxes,
        sub_mask_points_pred,
        obj_mask_points_pred,
        sub_gt_points_masks,
        obj_gt_points_masks,
        gt_sub_labels,
        gt_obj_labels,
        gt_rel_labels,
        img_meta,
        gt_bboxes_ignore,
    ):
        assert (
            gt_bboxes_ignore is None
        ), "Only case when gt_bboxes_ignore is None is supported."
        num_gt, num_query = gt_sub_labels.shape[0], sub_mask_points_pred.shape[0]

        # 1. assign -1 by default
        assigned_gt_inds = sub_mask_points_pred.new_full(
            (num_query,), -1, dtype=torch.long
        )
        assigned_labels = sub_mask_points_pred.new_full(
            (num_query,), -1, dtype=torch.long
        )
        if num_gt == 0 or num_query == 0:
            # No ground truth or boxes, return empty assignment
            if num_gt == 0:
                # No ground truth, assign all to background
                assigned_gt_inds[:] = 0
            return AssignResult(num_gt, assigned_gt_inds, None, labels=assigned_labels)

        # 2. compute the weighted costs
        # classification and maskcost.
        sub_cls_cost = self.s_cls_cost(s_cls_score, gt_sub_labels)
        sub_mask_cost = self.s_mask_cost(sub_mask_points_pred, sub_gt_points_masks)
        sub_dice_cost = self.s_dice_cost(sub_mask_points_pred, sub_gt_points_masks)
        obj_cls_cost = self.o_cls_cost(o_cls_score, gt_obj_labels)
        obj_mask_cost = self.o_mask_cost(obj_mask_points_pred, obj_gt_points_masks)
        obj_dice_cost = self.o_dice_cost(obj_mask_points_pred, obj_gt_points_masks)
        rel_cls_cost = self.r_cls_cost(r_cls_score, gt_rel_labels)

        cost = (
            sub_cls_cost
            + sub_mask_cost
            + sub_dice_cost
            + obj_cls_cost
            + obj_mask_cost
            + obj_dice_cost
            + rel_cls_cost
        )

        # 3. do Hungarian matching on CPU using linear_sum_assignment
        cost = cost.detach().cpu()

        matched_row_inds, matched_col_inds = linear_sum_assignment(cost)
        matched_row_inds = torch.from_numpy(matched_row_inds).to(
            sub_gt_points_masks.device
        )
        matched_col_inds = torch.from_numpy(matched_col_inds).to(
            sub_gt_points_masks.device
        )

        # 4. assign backgrounds and foregrounds
        # assign all indices to backgrounds first
        assigned_gt_inds[:] = 0
        # assign foregrounds based on matching results
        assigned_gt_inds[matched_row_inds] = matched_col_inds + 1
        assigned_labels[matched_row_inds] = gt_sub_labels[matched_col_inds]
        return AssignResult(num_gt, assigned_gt_inds, None, labels=assigned_labels)


@BBOX_ASSIGNERS.register_module()
class HTriMatcher(BaseAssigner):
    def __init__(
        self,
        s_cls_cost=dict(type="ClassificationCost", weight=1.0),
        s_reg_cost=dict(type="BBoxL1Cost", weight=1.0),
        s_iou_cost=dict(type="IoUCost", iou_mode="giou", weight=1.0),
        o_cls_cost=dict(type="ClassificationCost", weight=1.0),
        o_reg_cost=dict(type="BBoxL1Cost", weight=1.0),
        o_iou_cost=dict(type="IoUCost", iou_mode="giou", weight=1.0),
        r_cls_cost=dict(type="ClassificationCost", weight=1.0),
    ):
        self.s_cls_cost = build_match_cost(s_cls_cost)
        self.s_reg_cost = build_match_cost(s_reg_cost)
        self.s_iou_cost = build_match_cost(s_iou_cost)
        self.o_cls_cost = build_match_cost(o_cls_cost)
        self.o_reg_cost = build_match_cost(o_reg_cost)
        self.o_iou_cost = build_match_cost(o_iou_cost)
        self.r_cls_cost = build_match_cost(r_cls_cost)

    def assign(
        self,
        sub_bbox_pred,
        obj_bbox_pred,
        sub_cls_score,
        obj_cls_score,
        rel_cls_score,
        gt_sub_bboxes,
        gt_obj_bboxes,
        gt_sub_labels,
        gt_obj_labels,
        gt_rel_labels,
        img_meta,
        gt_bboxes_ignore=None,
        eps=1e-7,
    ):
        assert (
            gt_bboxes_ignore is None
        ), "Only case when gt_bboxes_ignore is None is supported."
        num_gts, num_bboxes = gt_sub_bboxes.size(0), sub_bbox_pred.size(0)

        # 1. assign -1 by default
        assigned_gt_inds = sub_bbox_pred.new_full((num_bboxes,), -1, dtype=torch.long)
        assigned_s_labels = sub_bbox_pred.new_full((num_bboxes,), -1, dtype=torch.long)
        assigned_o_labels = sub_bbox_pred.new_full((num_bboxes,), -1, dtype=torch.long)
        if num_gts == 0 or num_bboxes == 0:
            # No ground truth or boxes, return empty assignment
            if num_gts == 0:
                # No ground truth, assign all to background
                assigned_gt_inds[:] = 0
            return AssignResult(
                num_gts, assigned_gt_inds, None, labels=assigned_s_labels
            ), AssignResult(num_gts, assigned_gt_inds, None, labels=assigned_o_labels)
        img_h, img_w, _ = img_meta["img_shape"]
        factor = gt_sub_bboxes.new_tensor([img_w, img_h, img_w, img_h]).unsqueeze(0)

        # 2. compute the weighted costs
        # classification and bboxcost.
        s_cls_cost = self.s_cls_cost(sub_cls_score, gt_sub_labels)
        o_cls_cost = self.o_cls_cost(obj_cls_score, gt_obj_labels)
        r_cls_cost = self.r_cls_cost(rel_cls_score, gt_rel_labels)
        # regression L1 cost
        normalize_gt_sub_bboxes = gt_sub_bboxes / factor
        normalize_gt_obj_bboxes = gt_obj_bboxes / factor
        s_reg_cost = self.s_reg_cost(sub_bbox_pred, normalize_gt_sub_bboxes)
        o_reg_cost = self.o_reg_cost(obj_bbox_pred, normalize_gt_obj_bboxes)
        # regression iou cost, defaultly giou is used in official DETR.
        sub_bboxes = bbox_cxcywh_to_xyxy(sub_bbox_pred) * factor
        obj_bboxes = bbox_cxcywh_to_xyxy(obj_bbox_pred) * factor
        s_iou_cost = self.s_iou_cost(sub_bboxes, gt_sub_bboxes)
        o_iou_cost = self.o_iou_cost(obj_bboxes, gt_obj_bboxes)
        # weighted sum of above three costs
        beta_1, beta_2 = 1.2, 1
        alpha_s, alpha_o, alpha_r = 1, 1, 1
        cls_cost = (
            alpha_s * s_cls_cost + alpha_o * o_cls_cost + alpha_r * r_cls_cost
        ) / (alpha_s + alpha_o + alpha_r)
        bbox_cost = (s_reg_cost + o_reg_cost + s_iou_cost + o_iou_cost) / 2
        cost = beta_1 * cls_cost + beta_2 * bbox_cost

        # 3. do Hungarian matching on CPU using linear_sum_assignment
        cost = cost.detach().cpu()
        if linear_sum_assignment is None:
            raise ImportError(
                'Please run "pip install scipy" ' "to install scipy first."
            )
        matched_row_inds, matched_col_inds = linear_sum_assignment(cost)
        matched_row_inds = torch.from_numpy(matched_row_inds).to(sub_bbox_pred.device)
        matched_col_inds = torch.from_numpy(matched_col_inds).to(sub_bbox_pred.device)

        # 4. assign backgrounds and foregrounds
        # assign all indices to backgrounds first
        assigned_gt_inds[:] = 0
        # assign foregrounds based on matching results
        assigned_gt_inds[matched_row_inds] = matched_col_inds + 1
        assigned_s_labels[matched_row_inds] = gt_sub_labels[matched_col_inds]
        assigned_o_labels[matched_row_inds] = gt_obj_labels[matched_col_inds]
        return AssignResult(
            num_gts, assigned_gt_inds, None, labels=assigned_s_labels
        ), AssignResult(num_gts, assigned_gt_inds, None, labels=assigned_o_labels)


@BBOX_ASSIGNERS.register_module()
class IdMatcher(BaseAssigner):
    def __init__(
        self,
        sub_id_cost=dict(type="ClassificationCost", weight=1.0),
        obj_id_cost=dict(type="ClassificationCost", weight=1.0),
        r_cls_cost=dict(type="ClassificationCost", weight=1.0),
    ):
        self.sub_id_cost = build_match_cost(sub_id_cost)
        self.obj_id_cost = build_match_cost(obj_id_cost)
        self.r_cls_cost = build_match_cost(r_cls_cost)

    def assign(
        self,
        sub_score,
        obj_score,
        rel_cls_score,
        gt_sub_cls,
        gt_obj_cls,
        gt_rel_labels,
        img_meta,
        gt_bboxes_ignore=None,
        eps=1e-7,
    ):
        """gt_ids are mapped from previous Hungarian matching results.

        ~[0,99]
        """
        assert (
            gt_bboxes_ignore is None
        ), "Only case when gt_bboxes_ignore is None is supported."
        num_gts, num_bboxes = gt_rel_labels.shape[0], rel_cls_score.shape[0]

        # 1. assign -1 by default
        assigned_gt_inds = rel_cls_score.new_full((num_bboxes,), -1, dtype=torch.long)
        assigned_s_labels = rel_cls_score.new_full((num_bboxes,), -1, dtype=torch.long)
        assigned_o_labels = rel_cls_score.new_full((num_bboxes,), -1, dtype=torch.long)
        if num_gts == 0 or num_bboxes == 0:
            # No ground truth or boxes, return empty assignment
            if num_gts == 0:
                # No ground truth, assign all to background
                assigned_gt_inds[:] = 0
            return AssignResult(
                num_gts, assigned_gt_inds, None, labels=assigned_s_labels
            ), AssignResult(num_gts, assigned_gt_inds, None, labels=assigned_o_labels)

        # 2. compute the weighted costs
        # -object confidence
        sub_id_cost = self.sub_id_cost(sub_score, gt_sub_cls)
        obj_id_cost = self.obj_id_cost(obj_score, gt_obj_cls)
        r_cls_cost = self.r_cls_cost(rel_cls_score, gt_rel_labels)

        # weighted sum of above three costs
        cost = sub_id_cost + obj_id_cost + r_cls_cost

        # 3. do Hungarian matching on CPU using linear_sum_assignment
        cost = cost.detach().cpu()
        matched_row_inds, matched_col_inds = linear_sum_assignment(cost)
        matched_row_inds = torch.from_numpy(matched_row_inds).to(rel_cls_score.device)
        matched_col_inds = torch.from_numpy(matched_col_inds).to(rel_cls_score.device)

        # 4. assign backgrounds and foregrounds
        # assign all indices to backgrounds first
        assigned_gt_inds[:] = 0
        # assign foregrounds based on matching results
        assigned_gt_inds[matched_row_inds] = matched_col_inds + 1
        assigned_s_labels[matched_row_inds] = gt_sub_cls[matched_col_inds]
        return AssignResult(num_gts, assigned_gt_inds, None, labels=assigned_s_labels)


# the old Id matcher
@BBOX_ASSIGNERS.register_module()
class OldIdMatcher(BaseAssigner):
    def __init__(
        self,
        sub_id_cost=dict(type="ClassificationCost", weight=1.0),
        obj_id_cost=dict(type="ClassificationCost", weight=1.0),
        r_cls_cost=dict(type="ClassificationCost", weight=1.0),
    ):
        self.sub_id_cost = build_match_cost(sub_id_cost)
        self.obj_id_cost = build_match_cost(obj_id_cost)
        self.r_cls_cost = build_match_cost(r_cls_cost)

    def assign(
        self,
        sub_match_score,
        obj_match_score,
        rel_cls_score,
        gt_sub_ids,
        gt_obj_ids,
        gt_rel_labels,
        img_meta,
        gt_bboxes_ignore=None,
        eps=1e-7,
    ):
        """gt_ids are mapped from previous Hungarian matchinmg results.
        ~[0,99]
        """
        assert (
            gt_bboxes_ignore is None
        ), "Only case when gt_bboxes_ignore is None is supported."
        num_gts, num_bboxes = gt_rel_labels.size(0), rel_cls_score.size(0)

        # 1. assign -1 by default
        assigned_gt_inds = rel_cls_score.new_full((num_bboxes,), -1, dtype=torch.long)
        assigned_s_labels = rel_cls_score.new_full((num_bboxes,), -1, dtype=torch.long)
        assigned_o_labels = rel_cls_score.new_full((num_bboxes,), -1, dtype=torch.long)
        if num_gts == 0 or num_bboxes == 0:
            # No ground truth or boxes, return empty assignment
            if num_gts == 0:
                # No ground truth, assign all to background
                assigned_gt_inds[:] = 0
            return AssignResult(
                num_gts, assigned_gt_inds, None, labels=assigned_s_labels
            ), AssignResult(num_gts, assigned_gt_inds, None, labels=assigned_o_labels)

        # 2. compute the weighted costs
        # classification and bboxcost.
        sub_id_cost = self.sub_id_cost(sub_match_score, gt_sub_ids)
        obj_id_cost = self.obj_id_cost(obj_match_score, gt_obj_ids)
        r_cls_cost = self.r_cls_cost(rel_cls_score, gt_rel_labels)

        # weighted sum of above three costs
        cost = sub_id_cost + obj_id_cost + r_cls_cost

        # 3. do Hungarian matching on CPU using linear_sum_assignment
        cost = cost.detach().cpu()
        if linear_sum_assignment is None:
            raise ImportError(
                'Please run "pip install scipy" ' "to install scipy first."
            )
        matched_row_inds, matched_col_inds = linear_sum_assignment(cost)
        matched_row_inds = torch.from_numpy(matched_row_inds).to(rel_cls_score.device)
        matched_col_inds = torch.from_numpy(matched_col_inds).to(rel_cls_score.device)

        # 4. assign backgrounds and foregrounds
        # assign all indices to backgrounds first
        assigned_gt_inds[:] = 0
        # assign foregrounds based on matching results
        assigned_gt_inds[matched_row_inds] = matched_col_inds + 1
        assigned_s_labels[matched_row_inds] = gt_sub_ids[matched_col_inds]
        assigned_o_labels[matched_row_inds] = gt_obj_ids[matched_col_inds]
        return AssignResult(
            num_gts, assigned_gt_inds, None, labels=assigned_s_labels
        ), AssignResult(num_gts, assigned_gt_inds, None, labels=assigned_o_labels)

@BBOX_ASSIGNERS.register_module()
class SpeaQHungarianMatcherWrapper(BaseAssigner):
    def __init__(self, matcher_cfg):
        from .speaq_matcher import SpeaQHungarianMatcher
        self.matcher = SpeaQHungarianMatcher(cfg=matcher_cfg)

    def assign(self, outputs, targets, img_meta, **kwargs):
        combined_indices, _, _ = self.matcher.forward_relation(outputs, targets)
        # TODO: 你需要将 combined_indices 转换成 AssignResult 格式
        raise NotImplementedError("需要将 combined_indices 转换成 AssignResult")
    
# SpeaQMatcher
import numpy as np
@BBOX_ASSIGNERS.register_module()
class SpeaQMatcher(BaseAssigner):
    def __init__(
        self,
        sub_id_cost=dict(type="ClassificationCost", weight=1.0),
        obj_id_cost=dict(type="ClassificationCost", weight=1.0),
        r_cls_cost=dict(type="ClassificationCost", weight=1.0),
    ):
        self.sub_id_cost = build_match_cost(sub_id_cost)
        self.obj_id_cost = build_match_cost(obj_id_cost)
        self.r_cls_cost = build_match_cost(r_cls_cost)
        self.relation_freq = torch.tensor(list(sorted_dict.values()))
        self.num_mul_so_queries = 100 # 查询总数
        self.relation_order = torch.tensor([30, 19, 21, 29, 47, 28, 49, 0, 20, 7, 42, 39, 48, 40, 22, 6, 5, 18, 32, 15, 37, 10, 13, 45, 36, 12, 23, 3, 46, 4, 9, 8, 33, 2, 24, 16, 34, 41, 26, 11, 27, 38, 35, 1, 14, 43, 31, 25, 17, 44])
        self.num_groups = 5 # 查询分组组数
        self.quality_weight = 3.0 # 质量分数权重
        self.confidence_weight = 0.5 # 关系分类置信度权重
        self.dice_merge_op = 'min' # Dice 合并操作
        self.use_pred_confidence = True # 是否使用预测置信度增强质量评估
        self.dynamic_k = True # 是否动态确定每个GT的查询数量
        self.k = 6 # 每个GT分配的最大查询数量
        self.quality_threshold = 0.5 # 动态k时的质量分数阈值
        self.size_of_groups = self.get_group_list_by_n_groups(self.num_groups)
        self.grouping()

    def fill_list(self, num, n):
        quotient, remainder = divmod(num, n)
        lst = [quotient] * n
        for i in range(remainder):
            lst[-1 * (i + 1)] += 1
        return lst

    def grouping(self):
        # Initialize
        group_tensor = [-1] * 50

        # 按频率分组
        # self.size_of_groups = self.get_group_list_by_n_groups(self.num_groups)
        relation_freq_np = np.array(self.relation_freq)

        # 求每组的频率总和
        freq_splits = np.split(relation_freq_np, np.cumsum(self.size_of_groups)[:-1])
        sum_of_each_groups = np.array([x.sum() for x in freq_splits])

        n_queries_per_group = (sum_of_each_groups * self.num_mul_so_queries / sum_of_each_groups.sum()).astype(int)

        # 弥补除法误差
        fill = self.fill_list(self.num_mul_so_queries - n_queries_per_group.sum(), len(n_queries_per_group))
        n_queries_per_group += np.array(fill)

        # 保存最终查询数量配置
        self.n_queries_per_group = n_queries_per_group
        assert self.num_mul_so_queries == n_queries_per_group.sum()

        # 将 relation_order 按照 size_of_groups 分组
        rel_order_splits = np.split(self.relation_order, np.cumsum(self.size_of_groups)[:-1])
        self.rel_order = rel_order_splits

        # 构建 group_tensor：表示每个 relation 对应哪个 group
        for g, group in enumerate(self.rel_order):
            for rel_idx in group:
                group_tensor[rel_idx] = g
        self.group_tensor = group_tensor
        self.freq_list = list(n_queries_per_group)
        self.n_groups = len(self.freq_list)

    def get_group_list_by_n_groups(self, n_groups):
        total_list = []
        last_checked_index = 0
        current_idx = 0
        size_of_whole_groups = 0

        for i in range(n_groups - 1):
            sum_of_this_group = 0
            size_of_this_group = 0
            remaining_list = self.relation_freq[last_checked_index:]
            remaining_half_cnt = remaining_list.sum() // 2

            while (sum_of_this_group + self.relation_freq[current_idx]) < remaining_half_cnt:
                sum_of_this_group += self.relation_freq[current_idx]
                size_of_this_group += 1
                size_of_whole_groups += 1
                current_idx += 1

            total_list.append(size_of_this_group)
            last_checked_index = current_idx

        total_list.append(50 - size_of_whole_groups)
        return total_list


    def assign(
        self,
        sub_score,
        obj_score,
        rel_cls_score,
        gt_sub_cls,
        gt_obj_cls,
        gt_rel_labels,
        gt_sub_masks,
        gt_obj_masks,
        pred_sub_mask,
        pred_obj_mask,
        img_meta,
        gt_bboxes_ignore=None,
    ):
        """gt_ids are mapped from previous Hungarian matching results.

        ~[0,99]
        """
        """SpeaQ-inspired assignment with group-wise one-to-many matching.
        
        Args:
            k: maximum number of queries to assign per GT (one-to-many factor)
        """
        assert (
            gt_bboxes_ignore is None
        ), "Only case when gt_bboxes_ignore is None is supported."
        num_gts, num_bboxes = gt_rel_labels.shape[0], rel_cls_score.shape[0]
        device = rel_cls_score.device

        # 1. assign -1 by default
        assigned_gt_inds = rel_cls_score.new_full((num_bboxes,), -1, dtype=torch.long)
        assigned_s_labels = rel_cls_score.new_full((num_bboxes,), -1, dtype=torch.long)
        assigned_o_labels = rel_cls_score.new_full((num_bboxes,), -1, dtype=torch.long)

        if num_gts == 0 or num_bboxes == 0:
            # No ground truth or boxes, return empty assignment
            if num_gts == 0:
                # No ground truth, assign all to background
                assigned_gt_inds[:] = 0
            return AssignResult(
                num_gts, assigned_gt_inds, None, labels=assigned_s_labels
            ), AssignResult(num_gts, assigned_gt_inds, None, labels=assigned_o_labels)

        # 2. compute the weighted costs
        # -object confidence
        sub_id_cost = self.sub_id_cost(sub_score, gt_sub_cls)
        obj_id_cost = self.obj_id_cost(obj_score, gt_obj_cls)
        r_cls_cost = self.r_cls_cost(rel_cls_score, gt_rel_labels)  # [num_pred, num_gt]
        # print("r cls cost shape:", r_cls_cost.shape)

        # DICE mask similarity
        def dice_similarity(pred, target):
            pred_flat = pred.flatten(1)   # [num_pred, H*W]
            target_flat = target.flatten(1)  # [num_gt, H*W]
            inter = (pred_flat[:, None] * target_flat[None]).sum(-1)
            union = pred_flat[:, None].sum(-1) + target_flat[None].sum(-1)
            dice = (2 * inter + 1e-6) / (union + 1e-6)
            return dice

        # 计算 mask dice 相似度
        print("pred_sub_mask shape:", pred_sub_mask.shape)
        print("gt_sub_masks shape:", gt_sub_masks.shape)
        sub_dice = dice_similarity(pred_sub_mask.sigmoid(), gt_sub_masks.float())  # [num_pred, num_gt]
        print("sub_dice shape:", sub_dice.shape)
        obj_dice = dice_similarity(pred_obj_mask.sigmoid(), gt_obj_masks.float())

        # 质量分数
        quality_score = None
        if self.dice_merge_op == 'min':
            quality_score = torch.min(sub_dice, obj_dice)
        elif self.dice_merge_op == 'max':
            quality_score = torch.max(sub_dice, obj_dice)
        else:
            raise ValueError(f"Unsupported dice_merge_op: {self.dice_merge_op}")

        # weighted sum 
        # cost = sub_id_cost + obj_id_cost + r_cls_cost
        # cost = cost - 3.0 * quality_score

        # 可选：加入预测置信度增强质量评估
        print("quality score shape:", quality_score.shape)
        print("r_cls_cost shape:", r_cls_cost.shape)
        if hasattr(self, 'use_pred_confidence') and self.use_pred_confidence:
            quality_score = quality_score + self.confidence_weight * r_cls_cost

        # 3. do Hungarian matching on CPU using linear_sum_assignment
        # group_indices = [torch.where(torch.tensor(self.group_tensor) == i)[0] for i in range(self.n_groups)]
        # group_caps = self.n_queries_per_group.cpu().tolist()
        # groupwise_matches = [[] for _ in range(num_bboxes)]

        # cost = cost.detach().cpu()

        # for gt_idx in range(num_gts):
        #     group_id = self.group_tensor[gt_rel_labels[gt_idx].item()]
        #     max_k = group_caps[group_id]

        #     topk_val, topk_idx = torch.topk(-cost_cpu[:, gt_idx], max_k, largest=True)
        #     for pred_idx in topk_idx:
        #         groupwise_matches[pred_idx].append(gt_idx)

        # 3. 组掩码机制 - 确保GT只能分配给对应组内的查询
        group_mask = torch.zeros(num_bboxes, num_gts, device=device)

        for gt_idx in range(num_gts):
            gt_rel_label = gt_rel_labels[gt_idx].item()
            group_id = self.group_tensor[gt_rel_label].item()
            
            # 找到属于该组的查询索引
            group_query_start = sum(self.n_queries_per_group[:group_id]).item()
            group_query_end = group_query_start + self.n_queries_per_group[group_id].item()
            
            # 对于不在对应组内的查询，设置极大惩罚
            group_mask[:group_query_start, gt_idx] = 1e6
            group_mask[group_query_end:, gt_idx] = 1e6

        # 4. 计算总成本（越小越好）
        total_cost = sub_id_cost + obj_id_cost + r_cls_cost - self.quality_weight * quality_score + group_mask

        # 对每个 pred 选最高质量匹配的 gt
        # for pred_idx, gts in enumerate(groupwise_matches):
        #     if not gts:
        #         continue
        #     best_gt = gts[0] if len(gts) == 1 else \
        #         sorted(gts, key=lambda g: cost[pred_idx, g])[0]
        
        # 5. SpeaQ风格的一对多分配
        # 为每个GT动态确定要分配的查询数量
        matched_pairs = []
        assigned_gts_expanded = []
        
        for gt_idx in range(num_gts):
            gt_rel_label = gt_rel_labels[gt_idx].item()
            group_id = self.group_tensor[gt_rel_label].item()
            
            # 确定该组可用的查询范围
            group_query_start = sum(self.n_queries_per_group[:group_id]).item()
            group_query_end = group_query_start + self.n_queries_per_group[group_id].item()
            group_queries = torch.arange(group_query_start, group_query_end, device=device)
            
            # 基于质量分数选择top-k个查询
            gt_quality_scores = quality_score[group_queries, gt_idx]
            
            # 动态确定k值：基于质量阈值或固定k值
            if hasattr(self, 'dynamic_k') and self.dynamic_k:
                # 动态k：选择质量分数大于阈值的查询
                valid_mask = gt_quality_scores > self.quality_threshold
                dynamic_k = min(valid_mask.sum().item(), self.k)
                dynamic_k = max(dynamic_k, 1)  # 至少分配一个
            else:
                dynamic_k = min(self.k, len(group_queries))
            
            # 选择top-k个最高质量的查询
            if len(group_queries) > 0:
                topk_values, topk_indices = torch.topk(gt_quality_scores, dynamic_k, largest=True)
                selected_queries = group_queries[topk_indices]
                
                # 记录匹配对
                for query_idx in selected_queries:
                    matched_pairs.append((query_idx.item(), gt_idx))
                    assigned_gts_expanded.append(gt_idx)


        # matched_row_inds, matched_col_inds = linear_sum_assignment(cost)
        # matched_row_inds = torch.from_numpy(matched_row_inds).to(rel_cls_score.device)
        # matched_col_inds = torch.from_numpy(matched_col_inds).to(rel_cls_score.device)

        # 4. assign backgrounds and foregrounds
        # assign all indices to backgrounds first
        # assigned_gt_inds[:] = 0
        # # assign foregrounds based on matching results
        # assigned_gt_inds[matched_row_inds] = matched_col_inds + 1
        # assigned_s_labels[matched_row_inds] = gt_sub_cls[matched_col_inds]

        # 6. 应用匹配结果
        assigned_gt_inds[:] = 0  # 先全部设为背景
        
        if matched_pairs:
            matched_query_inds, matched_gt_inds = zip(*matched_pairs)
            matched_query_inds = torch.tensor(matched_query_inds, device=device)
            matched_gt_inds = torch.tensor(matched_gt_inds, device=device)
            
            # 分配前景
            assigned_gt_inds[matched_query_inds] = matched_gt_inds + 1
            assigned_s_labels[matched_query_inds] = gt_sub_cls[matched_gt_inds]
            # assigned_o_labels[matched_query_inds] = gt_obj_cls[matched_gt_inds]
        return AssignResult(num_gts, assigned_gt_inds, None, labels=assigned_s_labels)