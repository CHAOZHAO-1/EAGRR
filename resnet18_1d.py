from utils import *

import torch
import torch.nn as nn
import torch.nn.functional as F




class GCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.gc1 = nn.Linear(in_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.gc2 = nn.Linear(hidden_dim, out_dim)
        self.bn2 = nn.BatchNorm1d(out_dim)

    def forward(self, x, adj, return_embeddings=False):

        h = torch.bmm(adj, x)
        h = self.gc1(h)
        B, N, FF = h.shape
        h = self.bn1(h.view(-1, FF)).view(B, N, FF)
        h = F.relu(h)

        h = torch.bmm(adj, h)
        h = self.gc2(h)
        B, N, FF = h.shape
        h = self.bn2(h.view(-1, FF)).view(B, N, FF)
        h = F.relu(h)

        if return_embeddings:
            return h
        else:
            return h.mean(dim=1)




class CausalAttNetV1(nn.Module):
    def __init__(self, feature_dim=1024, attn_hidden=256, causal_ratio=0.6, temperature=0.5):
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * feature_dim, attn_hidden),
            nn.ReLU(),
            nn.Linear(attn_hidden, 1)
        )
        self.ratio = causal_ratio
        self.temperature = temperature

    def forward(self, x):
        B, N, F = x.shape
        edge_scores = []
        for b in range(B):
            node_feat = x[b]
            row_feat = node_feat.unsqueeze(1).repeat(1, N, 1)
            col_feat = node_feat.unsqueeze(0).repeat(N, 1, 1)
            edge_feat = torch.cat([row_feat, col_feat], dim=-1)

            edge_score = self.edge_mlp(edge_feat).squeeze(-1)
            edge_scores.append(edge_score)

        edge_scores = torch.stack(edge_scores)


        flat_scores = edge_scores.view(B, -1)


        soft_mask = torch.softmax(flat_scores / self.temperature, dim=1)


        k = int(self.ratio * flat_scores.shape[1])
        topk_vals, _ = torch.topk(soft_mask, k, dim=1)
        threshold = topk_vals[:, -1].unsqueeze(1)


        causal_mask = torch.where(soft_mask >= threshold, soft_mask, torch.zeros_like(soft_mask))


        causal_mask = causal_mask / (causal_mask.sum(dim=1, keepdim=True) + 1e-12)


        conf_mask = 1.0 - causal_mask

        causal_adj = causal_mask.view(B, N, N)
        conf_adj = conf_mask.view(B, N, N)

        return causal_adj, conf_adj, edge_scores, soft_mask






class GATLayer(nn.Module):
    def __init__(self, in_features, out_features, num_heads=1):
        super(GATLayer, self).__init__()
        self.num_heads = num_heads
        self.out_per_head = out_features // num_heads

        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.attn = nn.Parameter(torch.Tensor(num_heads, self.out_per_head * 2))
        nn.init.xavier_uniform_(self.attn)

        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, h, adj=None):
        B, N, _ = h.shape
        h_proj = self.linear(h)
        h_proj = h_proj.view(B, N, self.num_heads, self.out_per_head).permute(0, 2, 1, 3)

        h_i = h_proj.unsqueeze(3).repeat(1, 1, 1, N, 1)
        h_j = h_proj.unsqueeze(2).repeat(1, 1, N, 1, 1)

        e_ij = torch.cat([h_i, h_j], dim=-1)
        e_ij = torch.einsum('bhijn,hf->bhij', e_ij, self.attn)
        e_ij = self.leaky_relu(e_ij)

        if adj is not None:
            adj_mask = adj.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
            e_ij = e_ij.masked_fill(adj_mask == 0, float('-inf'))

        alpha = torch.softmax(e_ij, dim=-1)
        h_new = torch.matmul(alpha, h_proj)
        h_new = h_new.permute(0, 2, 1, 3).contiguous().view(B, N, -1)
        return h_new, alpha.mean(dim=1)



class EAGRR(nn.Module):
    def __init__(self, num_classes=10, causal_ratio=0.8):
        super(EAGRR, self).__init__()


        self.node_attn = GATLayer(in_features=1024, out_features=1024, num_heads=4)


        self.causal_attn = CausalAttNetV1(feature_dim=1024, attn_hidden=256, causal_ratio=causal_ratio)


        self.shared_encoder = GCN(in_dim=1024, hidden_dim=512, out_dim=128)


        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x, labels=None, env_labels=None, mode='loss'):


        B, N, _ = x.shape


        node_feats, node_attn_scores = self.node_attn(x)


        causal_adj, spurious_adj, edge_scores, soft_mask = self.causal_attn(node_feats)


        node_embeds = self.shared_encoder(node_feats, causal_adj, return_embeddings=True)

        feat_causal = node_embeds.mean(dim=1)


        pred_causal = self.classifier(feat_causal)

        # ===== 多模式返回 =====
        if mode == 'rep':
            return pred_causal, feat_causal, node_attn_scores, causal_adj

        elif mode == 'pred':
            return pred_causal

        elif mode == 'feat':
            return feat_causal

        elif mode == 'loss':

            if env_labels is None or labels is None:
                loss_erm = F.cross_entropy(pred_causal, labels) if labels is not None else torch.tensor(0.0, device=x.device)
                return loss_erm

            unique_envs = torch.unique(env_labels)
            erm_losses = []
            penalties = []


            for env in unique_envs:
                env_mask = (env_labels == env)
                if env_mask.sum() == 0:
                    continue

                pred_env = pred_causal[env_mask]

                labels_env = labels[env_mask]


                erm_loss = F.cross_entropy(pred_env, labels_env)
                erm_losses.append(erm_loss)


                scale = torch.tensor(1.0, requires_grad=True, device=pred_env.device)

                loss_scaled = F.cross_entropy(pred_env * scale, labels_env)

                grad = torch.autograd.grad(loss_scaled, [scale], create_graph=True)[0]

                penalty = torch.sum(grad ** 2)

                penalties.append(penalty)

            loss_mean = torch.stack(erm_losses).mean()

            penalty_mean = torch.stack(penalties).mean()

            return loss_mean, penalty_mean

        else:
            raise ValueError(f"Unknown mode {mode}")
