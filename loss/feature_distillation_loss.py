import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureProjection(nn.Module):
    """
    Projects student features to teacher feature dimensions.
    """
    def __init__(self, student_channels, teacher_channels):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Conv2d(student_channels, teacher_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(teacher_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.projector(x)

class FeatureDistillationLoss(nn.Module):
    """
    Combined loss for logit-based and feature-based distillation.
    """
    def __init__(self, student_channels_list, teacher_channels_list, 
                 temperature=1.0, alpha=0.5, beta=1.0):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha # CE weight
        self.beta = beta   # Feature loss weight
        # (1-alpha) is for logit distillation weight
        
        self.kl_div = nn.KLDivLoss(reduction='batchmean')
        
        self.projections = nn.ModuleList([
            FeatureProjection(s_ch, t_ch) 
            for s_ch, t_ch in zip(student_channels_list, teacher_channels_list)
        ])

    def forward(self, student_logits, teacher_logits, student_features, teacher_features, targets):
        # 1. Logit/Classification Loss (only if alpha is not None)
        logit_loss = 0
        if self.alpha is not None:
            T = self.temperature
            ce_loss = F.cross_entropy(student_logits, targets)
            kd_loss = self.kl_div(
                F.log_softmax(student_logits / T, dim=1),
                F.softmax(teacher_logits / T, dim=1)
            ) * (T * T)
            logit_loss = self.alpha * ce_loss + (1 - self.alpha) * kd_loss

        # 2. Feature Distillation Loss
        feature_loss = 0
        for i, (s_feat, t_feat) in enumerate(zip(student_features, teacher_features)):
            # Project student feature to teacher channel space
            s_proj = self.projections[i](s_feat)
            
            # Spatial alignment if necessary (pooling student to match teacher)
            if s_proj.shape[2:] != t_feat.shape[2:]:
                s_proj = F.interpolate(s_proj, size=t_feat.shape[2:], mode='bilinear', align_corners=False)
            
            feature_loss += F.mse_loss(s_proj, t_feat)

        return logit_loss + self.beta * feature_loss
