class CrossHead2WithSpeaQ(AnchorFreeHead):
    def __init__(
        self,
        num_classes,
        in_channels,
        num_relations,
        num_obj_query=100,
        num_rel_query=100,
        mapper: str = "conv_tiny",
        use_mask=True,
        pixel_decoder=None,
        transformer_decoder=None,
        feat_channels=256,
        out_channels=256,
        num_transformer_feat_level=3,
        embed_dims=256,
        relation_decoder=None,
        enforce_decoder_input_project=False,
        n_heads=8,
        positional_encoding=dict(
            type="SinePositionalEncoding", num_feats=128, normalize=True
        ),
        rel_cls_loss=None,
        subobj_cls_loss=None,
        importance_match_loss=None,
        loss_cls=None,
        loss_mask=None,
        loss_dice=None,
        train_cfg=None,
        test_cfg=dict(max_per_img=100),
        init_cfg=None,
        # SpeaQ specific parameters
        speaq_cfg=None,  # 新增：SpeaQ配置
        relation_freq=None,  # 新增：关系频率统计
        **kwargs,
    ):
        super(AnchorFreeHead, self).__init__(init_cfg)
        
        # 原有的初始化代码...
        self.num_classes = num_classes
        self.num_rel_query = num_rel_query
        self.num_relations = num_relations
        self.use_mask = use_mask
        self.relation_decoder = build_transformer_layer_sequence(relation_decoder)
        
        # SpeaQ相关初始化
        self._init_speaq_components(speaq_cfg, relation_freq)
        
        # 根据分组调整query embedding
        self._init_grouped_queries()
        
        # 其余原有初始化代码保持不变...
        self.rel_query_embed = nn.Embedding(self.num_rel_query, feat_channels)
        self.rel_query_embed2 = nn.Embedding(self.num_rel_query * 2, feat_channels)
        self.rel_query_embed3 = nn.Embedding(self.num_rel_query * 2, feat_channels)
        self.rel_query_feat = nn.Embedding(self.num_rel_query, feat_channels)
        self.update_importance = creat_cnn(mapper)
        
        # mask2former init
        self.n_heads = n_heads
        self.embed_dims = embed_dims
        assert "num_feats" in positional_encoding
        num_feats = positional_encoding["num_feats"]
        assert num_feats * 2 == self.embed_dims, (
            "embed_dims should"
            f" be exactly 2 times of num_feats. Found {self.embed_dims}"
            f" and {num_feats}."
        )

        self.num_classes = num_classes
        self.num_queries = num_obj_query
        self.num_transformer_feat_level = num_transformer_feat_level
        self.num_heads = transformer_decoder.transformerlayers.attn_cfgs.num_heads
        assert (
            pixel_decoder.encoder.transformerlayers.attn_cfgs.num_levels
            == num_transformer_feat_level
        )
        pixel_decoder_ = copy.deepcopy(pixel_decoder)
        pixel_decoder_.update(
            in_channels=in_channels,
            feat_channels=feat_channels,
            out_channels=out_channels,
        )
        self.pixel_decoder = build_plugin_layer(pixel_decoder_)[1]
        self.transformer_decoder = build_transformer_layer_sequence(transformer_decoder)
        self.decoder_embed_dims = self.transformer_decoder.embed_dims

        self.decoder_input_projs = ModuleList()
        # from low resolution to high resolution
        for _ in range(num_transformer_feat_level):
            if (
                self.decoder_embed_dims != feat_channels
                or enforce_decoder_input_project
            ):
                self.decoder_input_projs.append(
                    Conv2d(feat_channels, self.decoder_embed_dims, kernel_size=1)
                )
            else:
                self.decoder_input_projs.append(nn.Identity())
        self.decoder_positional_encoding = build_positional_encoding(
            positional_encoding
        )
        self.query_embed = nn.Embedding(self.num_queries, feat_channels)
        self.query_feat = nn.Embedding(self.num_queries, feat_channels)
        # from low resolution to high resolution
        self.level_embed = nn.Embedding(self.num_transformer_feat_level, feat_channels)

        self.cls_embed = nn.Linear(feat_channels, self.num_classes + 1)
        self.mask_embed = nn.Sequential(
            nn.Linear(feat_channels, feat_channels),
            nn.ReLU(inplace=True),
            nn.Linear(feat_channels, feat_channels),
            nn.ReLU(inplace=True),
            nn.Linear(feat_channels, out_channels),
        )

        self.test_cfg = test_cfg
        self.train_cfg = train_cfg
        if train_cfg:
            self.mask_assigner = build_assigner(self.train_cfg.mask_assigner)
            self.sampler = build_sampler(self.train_cfg.sampler, context=self)
            self.num_points = self.train_cfg.get("num_points", 12544)
            self.oversample_ratio = self.train_cfg.get("oversample_ratio", 3.0)
            self.importance_sample_ratio = self.train_cfg.get(
                "importance_sample_ratio", 0.75
            )
            self.id_assigner = build_assigner(self.train_cfg.id_assigner)
        self.num_obj_query = num_obj_query
        self.use_mask = use_mask
        self.in_channels = in_channels

        self.class_weight = loss_cls.class_weight
        self.loss_cls = build_loss(loss_cls)
        self.loss_mask = build_loss(loss_mask)
        self.loss_dice = build_loss(loss_dice)

        self.rel_cls_loss = build_loss(rel_cls_loss)
        self.subobj_cls_loss = build_loss(subobj_cls_loss)
        self.importance_match_loss = build_loss(importance_match_loss)
        if self.loss_cls.use_sigmoid:
            self.cls_out_channels = num_classes
        else:
            self.cls_out_channels = num_classes + 1

        self._init_layers()

    def _init_layers(self):
        """Initialize layers of the transformer head."""

        self.sub_query_update = nn.Sequential(
            Linear(self.embed_dims, self.embed_dims),
            nn.ReLU(inplace=True),
            Linear(self.embed_dims, self.embed_dims),
            nn.ReLU(inplace=True),
            Linear(self.embed_dims, self.embed_dims),
        )

        self.obj_query_update = nn.Sequential(
            Linear(self.embed_dims, self.embed_dims),
            nn.ReLU(inplace=True),
            Linear(self.embed_dims, self.embed_dims),
            nn.ReLU(inplace=True),
            Linear(self.embed_dims, self.embed_dims),
        )

        self.rel_cls_embed = Linear(self.embed_dims, self.num_relations)
    
    def init_weights(self):
        """Initialize weights of the transformer head."""
        # The initialization for transformer is important
        for m in self.decoder_input_projs:
            if isinstance(m, Conv2d):
                caffe2_xavier_init(m, bias=0)

        self.pixel_decoder.init_weights()

        for p in self.transformer_decoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)

        for p in self.relation_decoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        """load checkpoints."""
        super(AnchorFreeHead, self)._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
        
    def _init_speaq_components(self, speaq_cfg, relation_freq):
        """初始化SpeaQ组件"""
        if speaq_cfg is None:
            self.use_speaq = False
            return
            
        self.use_speaq = True
        self.speaq_cfg = speaq_cfg
        
        # Groupwise Query Specialization
        self.num_groups = speaq_cfg.get('num_groups', 5)
        self.query_multiply = speaq_cfg.get('query_multiply', 2)
        
        # Quality-Aware Multi-Assignment
        self.o2m_scheme = speaq_cfg.get('o2m_scheme', 'dynamic')
        self.o2m_k = speaq_cfg.get('o2m_k', 3)
        self.o2m_dynamic_scheme = speaq_cfg.get('o2m_dynamic_scheme', 'min')
        self.o2m_predicate_score = speaq_cfg.get('o2m_predicate_score', True)
        self.o2m_predicate_weight = speaq_cfg.get('o2m_predicate_weight', 0.1)
        
        # 关系频率和分组
        if relation_freq is not None:
            self.relation_freq = torch.tensor(relation_freq)
            self._init_relation_groups()
        else:
            # 如果没有提供频率，使用均匀分组
            self._init_uniform_groups()
            
    def _init_relation_groups(self):
        """基于频率初始化关系分组"""
        # 按频率排序关系
        sorted_indices = torch.argsort(self.relation_freq, descending=True)
        self.relation_order = sorted_indices
        
        # 计算每组的大小
        self.size_of_groups = self._get_group_list_by_n_groups(self.num_groups)
        
        # 创建分组
        device_group = 'cuda' if torch.cuda.is_available() else 'cpu'
        group_tensor = -torch.ones(self.num_relations, device=device_group)
        
        # 计算每组应分配的查询数量
        sum_of_each_groups = torch.as_tensor([
            x.sum().item() for x in torch.split(self.relation_freq, self.size_of_groups)
        ], device=device_group)
        
        total_rel_queries = self.num_rel_query * self.query_multiply
        n_queries_per_group = (sum_of_each_groups * total_rel_queries / sum_of_each_groups.sum()).int()
        
        # 调整确保总数匹配
        diff = total_rel_queries - n_queries_per_group.sum()
        if diff != 0:
            n_queries_per_group[-1] += diff
            
        self.n_queries_per_group = n_queries_per_group.long()
        
        # 为每个关系分配组ID
        self.rel_order = torch.split(self.relation_order, self.size_of_groups)
        for g, row in enumerate(self.rel_order):
            group_tensor[row] = g
        self.group_tensor = group_tensor
        
    def _get_group_list_by_n_groups(self, n_groups):
        """计算每组包含的关系数量"""
        total_list = []
        last_checked_index = 0
        current_idx = 0
        size_of_whole_groups = 0
        
        for i in range(n_groups - 1):
            sum_of_this_group = 0
            size_of_this_group = 0
            remaining_list = self.relation_freq.numpy()[last_checked_index:]
            remaining_half_cnt = remaining_list.sum() // 2
            
            while (sum_of_this_group + self.relation_freq.numpy()[current_idx] < remaining_half_cnt and
                   current_idx < len(self.relation_freq) - 1):
                sum_of_this_group += self.relation_freq.numpy()[current_idx]
                size_of_this_group += 1
                size_of_whole_groups += 1
                current_idx += 1
                
            total_list.append(size_of_this_group)
            last_checked_index = current_idx
            
        total_list.append(self.num_relations - size_of_whole_groups)
        return total_list
    
    def _init_uniform_groups(self):
        """均匀分组初始化"""
        group_size = self.num_relations // self.num_groups
        self.size_of_groups = [group_size] * self.num_groups
        self.size_of_groups[-1] += self.num_relations % self.num_groups
        
        self.n_queries_per_group = torch.tensor([
            self.num_rel_query * self.query_multiply // self.num_groups
        ] * self.num_groups)
        
        device_group = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.group_tensor = torch.arange(self.num_relations, device=device_group) // group_size
    
    def _init_grouped_queries(self):
        """初始化分组查询"""
        if not self.use_speaq:
            return
            
        # 为每个组创建专门的查询嵌入
        self.group_query_embeds = nn.ModuleList()
        self.group_query_feats = nn.ModuleList()
        
        for i, n_queries in enumerate(self.n_queries_per_group):
            self.group_query_embeds.append(
                nn.Embedding(n_queries, self.embed_dims)
            )
            self.group_query_feats.append(
                nn.Embedding(n_queries, self.embed_dims)
            )

    def _apply_group_mask(self, cost_matrix, gt_rel_labels, batch_size, num_queries):
        """应用组掩码确保查询只能匹配对应组的关系"""
        if not self.use_speaq or len(gt_rel_labels) == 0:
            return cost_matrix
            
        # 创建组掩码
        group_mask = 1 - F.one_hot(
            self.group_tensor[gt_rel_labels].long(), 
            num_classes=len(self.n_queries_per_group)
        ).t()
        
        # 扩展掩码到查询维度
        query_group_mask = torch.zeros(batch_size, num_queries, len(gt_rel_labels))
        start_idx = 0
        
        for group_idx, n_queries in enumerate(self.n_queries_per_group):
            end_idx = start_idx + n_queries
            # 该组查询只能匹配对应组的关系
            query_group_mask[:, start_idx:end_idx] = group_mask[group_idx].unsqueeze(0).unsqueeze(0)
            start_idx = end_idx
            
        # 应用大的惩罚值
        cost_matrix = cost_matrix + query_group_mask.to(cost_matrix.device) * 1e6
        return cost_matrix
    
    def _apply_multi_assignment(self, outputs, targets):
        """应用Quality-Aware Multi-Assignment"""
        if not self.use_speaq or self.o2m_scheme == 'none':
            return targets
            
        copy_targets = copy.deepcopy(targets)
        bs, num_queries = outputs["rel_cls_scores"].shape[:2]
        
        if self.o2m_scheme == 'static':
            # 静态多分配：每个GT重复k次
            for t in copy_targets:
                if 'gt_rels' in t:
                    t['gt_rels'] = t['gt_rels'].repeat_interleave(self.o2m_k, dim=0)
                    
        elif self.o2m_scheme == 'dynamic':
            # 动态多分配：基于质量分数
            self._apply_dynamic_multi_assignment(outputs, copy_targets, bs, num_queries)
            
        return copy_targets
    
    def _apply_dynamic_multi_assignment(self, outputs, targets, bs, num_queries):
        """应用动态多分配"""
        # 计算IoU分数
        for batch_idx, target in enumerate(targets):
            if 'gt_rels' not in target or len(target['gt_rels']) == 0:
                continue
                
            gt_rels = target['gt_rels']
            
            # 这里需要根据你的具体输出格式调整
            # 假设你有sub_bbox_pred和obj_bbox_pred
            if 'sub_bbox_pred' in outputs and 'obj_bbox_pred' in outputs:
                sub_pred_bbox = outputs['sub_bbox_pred'][batch_idx]  # [num_queries, 4]
                obj_pred_bbox = outputs['obj_bbox_pred'][batch_idx]  # [num_queries, 4]
                
                # 计算与GT的IoU
                sub_iou = self._compute_bbox_iou(sub_pred_bbox, target['sub_gt_bbox'])
                obj_iou = self._compute_bbox_iou(obj_pred_bbox, target['obj_gt_bbox'])
                
                # 根据方案组合IoU
                if self.o2m_dynamic_scheme == 'min':
                    cum_iou = torch.min(sub_iou, obj_iou)
                elif self.o2m_dynamic_scheme == 'max':
                    cum_iou = torch.max(sub_iou, obj_iou)
                elif self.o2m_dynamic_scheme == 'gm':  # geometric mean
                    cum_iou = (sub_iou * obj_iou) ** 0.5
                elif self.o2m_dynamic_scheme == 'am':  # arithmetic mean
                    cum_iou = 0.5 * (sub_iou + obj_iou)
                    
                # 添加谓词置信分数
                if self.o2m_predicate_score and 'rel_cls_scores' in outputs:
                    rel_scores = F.softmax(outputs['rel_cls_scores'][batch_idx], dim=-1)
                    gt_rel_labels = gt_rels[:, 2]
                    predicate_scores = rel_scores[:, gt_rel_labels]
                    cum_iou += self.o2m_predicate_weight * predicate_scores
                    
                # 选择top-k进行多分配
                topk_scores, topk_indices = torch.topk(cum_iou, self.o2m_k, dim=0)
                k_per_relation = (topk_scores > 0).sum(dim=0).clamp_(min=1)
                
                # 扩展GT
                target['gt_rels'] = target['gt_rels'].repeat_interleave(k_per_relation, dim=0)
    
    def _compute_bbox_iou(self, pred_bbox, gt_bbox):
        """计算bbox IoU"""
        # 这里需要根据你的bbox格式实现IoU计算
        # 假设使用box_iou函数
        return box_iou(pred_bbox, gt_bbox)[0]

    def _get_target_single_with_speaq(
        self,
        subject_score,
        object_score,
        cls_score,
        mask_pred,
        r_cls_score,
        gt_rels,
        gt_labels,
        gt_masks,
        img_metas,
        sub_mask_pred=None,
        obj_mask_pred=None,
        gt_bboxes_ignore=None,
    ):
        """带SpeaQ增强的目标生成"""
        # 首先执行原始的目标生成逻辑
        (
            r_labels,
            r_label_weights,
            gt_subject_ids,
            gt_object_ids,
            gt_importance,
        ) = self._get_target_single_original(
            subject_score, object_score, cls_score, mask_pred, r_cls_score,
            gt_rels, gt_labels, gt_masks, img_metas, sub_mask_pred, obj_mask_pred, gt_bboxes_ignore
        )
        
        if not self.use_speaq:
            return r_labels, r_label_weights, gt_subject_ids, gt_object_ids, gt_importance
            
        # 应用SpeaQ增强
        # 1. 应用组掩码到分配结果
        if len(gt_rels) > 0:
            gt_rel_labels = gt_rels[:, 2] - 1  # 关系类别
            
            # 重新计算考虑组约束的分配
            # 这里需要重新调用assigner，传入组掩码
            enhanced_assign_result = self._enhanced_triplet_assignment(
                subject_score, object_score, r_cls_score,
                gt_rels, gt_labels, img_metas, gt_rel_labels
            )
            
            # 更新分配结果
            if enhanced_assign_result is not None:
                enhanced_sampling_result = self.sampler.sample(
                    enhanced_assign_result,
                    torch.ones_like(subject_score),
                    torch.ones_like(subject_score),
                )
                
                # 更新标签
                pos_inds = enhanced_sampling_result.pos_inds
                r_labels = torch.full(
                    (self.num_rel_query,), -1, dtype=torch.long, device=gt_labels.device
                )
                r_labels[pos_inds] = gt_rel_labels[enhanced_sampling_result.pos_assigned_gt_inds]
                r_label_weights = gt_labels.new_zeros(self.num_rel_query)
                r_label_weights[pos_inds] = 1.0
        
        return r_labels, r_label_weights, gt_subject_ids, gt_object_ids, gt_importance
    
    def _enhanced_triplet_assignment(self, subject_score, object_score, r_cls_score, 
                                   gt_rels, gt_labels, img_metas, gt_rel_labels):
        """增强的三元组分配，考虑组约束"""
        # 这里需要修改你的id_assigner来支持组掩码
        # 或者在成本矩阵上应用组掩码
        
        # 调用原始的assigner
        assign_result = self.id_assigner.assign(
            subject_score, object_score, r_cls_score,
            gt_labels[gt_rels[:, 0]], gt_labels[gt_rels[:, 1]], gt_rel_labels,
            img_metas, None
        )
        
        # 应用组掩码到成本矩阵（如果assigner支持的话）
        # 或者后处理分配结果
        
        return assign_result

    def loss_single_with_speaq(
        self,
        sub_cls_preds,
        obj_cls_preds,
        importance,
        od_cls_scores,
        mask_preds,
        r_cls_scores,
        gt_rels_list,
        gt_labels_list,
        gt_masks_list,
        img_metas,
        sub_mask_preds=None,
        obj_mask_preds=None,
        gt_bboxes_ignore_list=None,
    ):
        """带SpeaQ增强的loss计算"""
        
        # 如果启用了SpeaQ，应用多分配
        if self.use_speaq:
            # 构建outputs字典用于多分配
            outputs = {
                'rel_cls_scores': r_cls_scores.unsqueeze(0),  # 添加batch维度
                # 根据需要添加其他输出
            }
            targets = [{'gt_rels': gt_rels} for gt_rels in gt_rels_list]
            enhanced_targets = self._apply_multi_assignment(outputs, targets)
            gt_rels_list = [t['gt_rels'] for t in enhanced_targets]
        
        # 调用原始的loss_single函数
        return self.loss_single_original(
            sub_cls_preds, obj_cls_preds, importance,
            od_cls_scores, mask_preds, r_cls_scores,
            gt_rels_list, gt_labels_list, gt_masks_list,
            img_metas, sub_mask_preds, obj_mask_preds, gt_bboxes_ignore_list
        )

    def forward(self, feats, img_metas):
        """前向传播，集成SpeaQ组件"""
        batch_size = len(img_metas)
        mask_features, multi_scale_memorys = self.pixel_decoder(feats)
        
        # ... 原有的前向传播逻辑保持不变，直到关系查询部分 ...
        
        # 如果使用SpeaQ，使用分组查询
        if self.use_speaq:
            rel_query_feat_list = []
            rel_query_embed_list = []
            
            for group_idx, n_queries in enumerate(self.n_queries_per_group):
                group_rel_feat = self.group_query_feats[group_idx].weight.unsqueeze(1).repeat(
                    (1, batch_size, 1)
                )
                group_rel_embed = self.group_query_embeds[group_idx].weight.unsqueeze(1).repeat(
                    (1, batch_size, 1)
                )
                rel_query_feat_list.append(group_rel_feat)
                rel_query_embed_list.append(group_rel_embed)
                
            rel_query_feat = torch.cat(rel_query_feat_list, dim=0)
            rel_query_embed = torch.cat(rel_query_embed_list, dim=0)
        else:
            # 使用原始的查询嵌入
            rel_query_feat = self.rel_query_feat.weight.unsqueeze(1).repeat(
                (1, batch_size, 1)
            )
            rel_query_embed = self.rel_query_embed.weight.unsqueeze(1).repeat(
                (1, batch_size, 1)
            )
        
        # ... 继续原有的前向传播逻辑 ...
        
        # 其余部分保持不变
        return all_cls_scores, all_mask_preds