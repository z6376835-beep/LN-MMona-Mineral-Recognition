import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from .fusion_blocks import SEBlock, MineralFusion, INNER_DIM

class MMona(nn.Module):
    def __init__(self, in_dim, factor=4):
        super().__init__()
        self.INNER_DIM = max(in_dim // factor, INNER_DIM)
        self.project1 = nn.Linear(in_dim, self.INNER_DIM)
        self.project2 = nn.Linear(self.INNER_DIM, in_dim)
        self.dropout = nn.Dropout(0.1)
        self.adapter_conv = MineralFusion(self.INNER_DIM)
        self.norm = nn.LayerNorm(in_dim)
        self.gamma = nn.Parameter(torch.ones(in_dim) * 1e-6)
        self.gammax = nn.Parameter(torch.ones(in_dim))
        nn.init.normal_(self.project1.weight, 0.0, 0.02)
        nn.init.constant_(self.project1.bias, 0)
        nn.init.normal_(self.project2.weight, 0.0, 0.02)
        nn.init.constant_(self.project2.bias, 0)

    def forward(self, x, hw_shapes=(14, 14)):
        b, n, c = x.shape
        cls_token = x[:, :1, :]
        patch_tokens = x[:, 1:, :]
        identity = patch_tokens
        x = self.norm(patch_tokens) * self.gamma + patch_tokens * self.gammax
        project1 = self.project1(x)
        h, w = hw_shapes
        project1 = project1.reshape(b, h, w, self.INNER_DIM).permute(0, 3, 1, 2)
        project1 = self.adapter_conv(project1)
        project1 = project1.permute(0, 2, 3, 1).reshape(b, h*w, self.INNER_DIM)
        x = F.gelu(project1)
        x = self.dropout(x)
        project2 = self.project2(x)
        patch_output = identity + project2
        return torch.cat([cls_token, patch_output], dim=1)

class ModifiedEncoderBlock(nn.Module):
    def __init__(self, original_block, dim=768):
        super().__init__()
        self.original_block = original_block
        self.MMona_msa = MMona(dim)
        self.MMona_mlp = MMona(dim)

    def forward(self, x, hw_shapes=(14, 14)):
        res = x
        x = self.original_block.ln_1(x)
        x = self.original_block.self_attention(x, x, x)[0]
        x = res + self.original_block.dropout(x)
        x = self.MMona_msa(x, hw_shapes)
        res = x
        x = self.original_block.ln_2(x)
        x = self.original_block.mlp(x)
        x = res + self.original_block.dropout(x)
        x = self.MMona_mlp(x, hw_shapes)
        return x

class LNMMonaClassifier(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super().__init__()
        self.vit = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None)
        for param in self.vit.parameters():
            param.requires_grad = False
        original_layers = self.vit.encoder.layers
        self.vit.encoder.layers = nn.ModuleList([
            ModifiedEncoderBlock(original_layers[i], 768) for i in range(len(original_layers))
        ])
        for layer in self.vit.encoder.layers:
            for param in layer.MMona_msa.parameters():
                param.requires_grad = True
            for param in layer.MMona_mlp.parameters():
                param.requires_grad = True
        self.vit.heads.head = nn.Linear(self.vit.heads.head.in_features, num_classes)
        for param in self.vit.heads.head.parameters():
            param.requires_grad = True
        for name, param in self.vit.named_parameters():
            if 'ln' in name:
                param.requires_grad = True

    def forward(self, x):
        x = self.vit._process_input(x)
        cls_token = self.vit.class_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        x = x + self.vit.encoder.pos_embedding
        x = self.vit.encoder.dropout(x)
        for layer in self.vit.encoder.layers:
            x = layer(x, hw_shapes=(14, 14))
        x = self.vit.encoder.ln(x)
        cls_token = x[:, 0]
        return self.vit.heads(cls_token)
