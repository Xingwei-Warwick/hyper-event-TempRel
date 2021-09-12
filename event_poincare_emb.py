import torch as th
from torch import nn
import geoopt as gt
import numpy as np
from transformers import AutoModel

'''
EventTempRel Poincare Embedding model classes
'''

class EventTempRel_poincare_static(nn.Module):
    def __init__(self, device, dim_in, dim_out=64, num_neg=1, alpha=0.5):
        super().__init__()
        self.dim_out = dim_out
        self.dim_in = dim_in
        self.device = device
        #self.name = 'myP'

        self.ball = gt.Stereographic(-1) # declare a Poincare model using Geoopt's Stereographic class. The curvature is -1

        self.W_ff_u = nn.Parameter(data=th.zeros(self.dim_out, self.dim_in))
        
        nn.init.uniform_(self.W_ff_u, -0.05, 0.05)

        self.hyper_para = [self.W_ff_u]
        self.euclid_para = []

        self.num_neg = num_neg # number of negative samples
        self.alpha = alpha

        self.epsilon = th.tensor([1e-9], device=self.device, requires_grad=False)
    
    def set_neg(self, neg_num):
        self.num_neg = neg_num
    
    def get_embeds(self, sequence, mask1, mask2):
        projected = self.ball.expmap0(sequence)
        
        u = projected * mask1 # boardcast
        v = projected * mask2
        u = th.sum(u, dim=1)
        v = th.sum(v, dim=1)
        u = self.ball.mobius_matvec(self.W_ff_u, u)
        v = self.ball.mobius_matvec(self.W_ff_u, v)

        return u.detach().cpu().numpy(), v.detach().cpu().numpy()

    def get_score(self, sequence, mask1, mask2):
        projected = self.ball.expmap0(sequence)
        
        u = projected * mask1 # boardcast
        v = projected * mask2
        u = th.sum(u, dim=1)
        v = th.sum(v, dim=1)
        u = self.ball.mobius_matvec(self.W_ff_u, u)
        v = self.ball.mobius_matvec(self.W_ff_u, v)
        dsq = self.ball.dist(u, v)

        u_hnorm = self.ball.dist0(u)
        v_hnorm = self.ball.dist0(v)

        score = (u_hnorm - v_hnorm) / (dsq + self.epsilon)

        return score.detach().cpu().numpy()

    def forward(self, sequence, mask1, mask2, neg_seq, mask_u_neg):
        # sequence [batch, len, embeddings_dim]
        projected = self.ball.expmap0(sequence)
        
        u = projected * mask1 # boardcast
        v = projected * mask2
        u = th.sum(u, dim=1)
        v = th.sum(v, dim=1)
        u = self.ball.mobius_matvec(self.W_ff_u, u)
        v = self.ball.mobius_matvec(self.W_ff_u, v)

        # compute angle
        norm_u = th.norm(u, dim=1) # [batch,1]
        norm_v = th.norm(v, dim=1)
        euclid_dist_uv = th.norm((u-v), dim=1)
        dot_prod_uv = th.sum(u * v, dim=1)

        cos_angle_child_vu = (dot_prod_uv * (1+norm_v**2) - norm_v ** 2 * (1+norm_u**2)) /\
            (norm_v * euclid_dist_uv * th.sqrt(1+norm_u**2*norm_v**2-2*dot_prod_uv))
        angles_child_vu = th.acos(cos_angle_child_vu)# [0,\pi]

        neg_projected = self.ball.expmap0(neg_seq)
        u_neg = neg_projected * mask_u_neg
        u_neg = th.sum(u_neg, dim=1)
        u_neg = self.ball.mobius_matvec(self.W_ff_u, u_neg).view(u.size(0),self.num_neg,self.dim_out)

        dsq = self.ball.dist(u, v)
        exp_neg_dists = th.exp(-dsq)

        Z1 = th.zeros(exp_neg_dists.size(), device=self.device)
        neg_ang_loss = th.zeros(exp_neg_dists.size(), device=self.device)
        for neg_n in range(self.num_neg):
            dsq_n = self.ball.dist(u, u_neg[:,neg_n,:])
            neg_dsq_n = th.exp(-dsq_n)
            Z1 += neg_dsq_n

        ns_loss = -th.log(exp_neg_dists / (Z1 + exp_neg_dists))
        score = angles_child_vu

        loss = th.mean(2*(1-self.alpha)*score + 2*self.alpha*ns_loss)

        return loss


class EventTempRel_poincare(nn.Module):
    def __init__(self, device, dim_in, dim_out=64, num_neg=1, alpha=0.5, model_name='roberta-base'):
        super().__init__()
        self.dim_out = dim_out
        self.dim_in = dim_in
        self.device = device
        #self.name = 'myP'

        self.ball = gt.Stereographic(-1) # declare a Poincare model using Geoopt's Stereographic class. The curvature is -1

        self.W_ff_u = nn.Parameter(data=th.zeros(self.dim_out, self.dim_in))
        
        nn.init.uniform_(self.W_ff_u, -0.05, 0.05)

        self.model_name = model_name
        self.encoder_model = AutoModel.from_pretrained(self.model_name, return_dict=True)
        self.encoder_model.train()

        self.hyper_para = [self.W_ff_u]
        self.euclid_para = self.encoder_model.parameters()

        self.num_neg = num_neg
        self.alpha = alpha

        self.epsilon = th.tensor([1e-9], device=self.device, requires_grad=False)
    
    def set_neg(self, neg_num):
        self.num_neg = neg_num
    
    def get_embeds(self, sequence, s_a_mask, mask1, mask2):
        encoded = self.encoder_model(sequence, s_a_mask).last_hidden_state
        projected = self.ball.expmap0(encoded)
        
        u = projected * mask1 # boardcast
        v = projected * mask2
        u = th.sum(u, dim=1)
        v = th.sum(v, dim=1)
        u = self.ball.mobius_matvec(self.W_ff_u, u)
        v = self.ball.mobius_matvec(self.W_ff_u, v)

        return u.detach().cpu().numpy(), v.detach().cpu().numpy()

    def get_score(self, sequence, s_a_mask, mask1, mask2):
        encoded = self.encoder_model(sequence, s_a_mask).last_hidden_state
        projected = self.ball.expmap0(encoded)
        
        u = projected * mask1 # boardcast
        v = projected * mask2
        u = th.sum(u, dim=1)
        v = th.sum(v, dim=1)
        u = self.ball.mobius_matvec(self.W_ff_u, u)
        v = self.ball.mobius_matvec(self.W_ff_u, v)
        dsq = self.ball.dist(u, v)

        u_hnorm = self.ball.dist0(u)
        v_hnorm = self.ball.dist0(v)

        score = (u_hnorm - v_hnorm) / (dsq + self.epsilon)

        return score.detach().cpu().numpy()

    def forward(self, sequence, s_a_mask, mask1, mask2, neg_seq, n_a_mask, mask_u_neg):
        # sequence [batch, len, embeddings_dim]
        encoded = self.encoder_model(sequence, s_a_mask).last_hidden_state
        projected = self.ball.expmap0(encoded)
        
        u = projected * mask1 # boardcast
        v = projected * mask2
        u = th.sum(u, dim=1)
        v = th.sum(v, dim=1)
        u = self.ball.mobius_matvec(self.W_ff_u, u)
        v = self.ball.mobius_matvec(self.W_ff_u, v)

        # compute angle
        norm_u = th.norm(u, dim=1) # [batch,1]
        norm_v = th.norm(v, dim=1)
        euclid_dist_uv = th.norm((u-v), dim=1)
        dot_prod_uv = th.sum(u * v, dim=1)

        cos_angle_child_vu = (dot_prod_uv * (1+norm_v**2) - norm_v ** 2 * (1+norm_u**2)) /\
            (norm_v * euclid_dist_uv * th.sqrt(1+norm_u**2*norm_v**2-2*dot_prod_uv))
        angles_child_vu = th.acos(cos_angle_child_vu)# [0,\pi]

        n_encoded = self.encoder_model(neg_seq, n_a_mask).last_hidden_state
        neg_projected = self.ball.expmap0(n_encoded)
        u_neg = neg_projected * mask_u_neg
        u_neg = th.sum(u_neg, dim=1)
        u_neg = self.ball.mobius_matvec(self.W_ff_u, u_neg).view(u.size(0),self.num_neg,self.dim_out)

        dsq = self.ball.dist(u, v)
        exp_neg_dists = th.exp(-dsq)

        Z1 = th.zeros(exp_neg_dists.size(), device=self.device)
        neg_ang_loss = th.zeros(exp_neg_dists.size(), device=self.device)
        for neg_n in range(self.num_neg):
            dsq_n = self.ball.dist(u, u_neg[:,neg_n,:])
            neg_dsq_n = th.exp(-dsq_n)
            Z1 += neg_dsq_n

        ns_loss = -th.log(exp_neg_dists / (Z1 + exp_neg_dists))
        score = angles_child_vu

        loss = th.mean(2*(1-self.alpha)*score + 2*self.alpha*ns_loss)

        return loss
    
    def get_reduced_subject_object_embs(self, events):
        sub = events[:1]

        norm_s = th.norm(sub, dim=1, p=2) # [1]
        objs = events

        out_x = th.matmul(objs, (sub.squeeze() / norm_s)).unsqueeze(1) # [N+1, 1]
        norm_all = th.norm(objs, dim=1, p=2, keepdim=True)
        out_y = th.sqrt(norm_all**2 - th.abs(out_x)**2)

        return th.cat((out_x, out_y), dim=1).detach().cpu()

