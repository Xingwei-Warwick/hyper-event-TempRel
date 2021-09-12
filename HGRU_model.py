import torch as th
from torch import nn
import geoopt as gt
import numpy as np
from transformers import AutoModel


'''
Hyperbolic Gated Recurrent Unit
Class HyperGRUCell and HyperGRU are created based on the TensorFlow version on (https://github.com/dalab/hyperbolic_nn)
'''


class HyperGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size, ball):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.ball = ball

        k = (1 / self.hidden_size) ** 0.5
        self.w_z = gt.ManifoldParameter(
            gt.ManifoldTensor(self.hidden_size, self.hidden_size).uniform_(-k, k)
        )
        self.w_r = gt.ManifoldParameter(
            gt.ManifoldTensor(self.hidden_size, self.hidden_size).uniform_(-k, k)
        )
        self.w_h = gt.ManifoldParameter(
            gt.ManifoldTensor(self.hidden_size, self.hidden_size).uniform_(-k, k)
        )
        self.u_z = gt.ManifoldParameter(
            gt.ManifoldTensor(self.hidden_size, self.input_size).uniform_(-k, k)
        )
        self.u_r = gt.ManifoldParameter(
            gt.ManifoldTensor(self.hidden_size, self.input_size).uniform_(-k, k)
        )
        self.u_h = gt.ManifoldParameter(
            gt.ManifoldTensor(self.hidden_size, self.input_size).uniform_(-k, k)
        )
        self.b_z = gt.ManifoldParameter(
            gt.ManifoldTensor(self.hidden_size, manifold=self.ball).zero_()
        )
        self.b_r = gt.ManifoldParameter(
            gt.ManifoldTensor(self.hidden_size, manifold=self.ball).zero_()
        )
        self.b_h = gt.ManifoldParameter(
            gt.ManifoldTensor(self.hidden_size, manifold=self.ball).zero_()
        )

    def transition(self, W, h, U, x, hyp_b):
        W_otimes_h = self.ball.mobius_matvec(W, h)
        U_otimes_x = self.ball.mobius_matvec(U, x)
        Wh_plus_Ux = self.ball.mobius_add(W_otimes_h, U_otimes_x)

        return self.ball.mobius_add(Wh_plus_Ux, hyp_b)

    def forward(self, hyp_x, hidden):
        z = self.transition(self.w_z, hidden, self.u_z, hyp_x, self.b_z)
        z = th.sigmoid(self.ball.logmap0(z))

        r = self.transition(self.w_r, hidden, self.u_r, hyp_x, self.b_r)
        r = th.sigmoid(self.ball.logmap0(r))

        r_point_h = self.ball.mobius_pointwise_mul(hidden, r)
        h_tilde = self.transition(self.w_h, r_point_h, self.u_h, hyp_x, self.b_h)

        minus_h_oplus_htilde = self.ball.mobius_add(-hidden, h_tilde)
        new_h = self.ball.mobius_add(
            hidden, self.ball.mobius_pointwise_mul(minus_h_oplus_htilde, z)
        )

        return new_h


class HyperGRU(nn.Module):
    def __init__(self, input_size, hidden_size, ball, default_dtype=th.float64):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.ball = ball
        self.default_dtype = default_dtype

        self.gru_cell = HyperGRUCell(self.input_size, self.hidden_size, ball=self.ball)

    def reset_parameters(self):
        self.gru_cell.reset_parameters()

    def init_gru_state(self, batch_size, hidden_size, cuda_device):
        return th.zeros(
            (batch_size, hidden_size), dtype=self.default_dtype, device=cuda_device
        )

    def forward(self, inputs):
        hidden = self.init_gru_state(inputs.shape[0], self.hidden_size, inputs.device)
        outputs = []
        for x in inputs.transpose(0, 1):
            hidden = self.gru_cell(x, hidden)
            outputs += [hidden]
        return th.stack(outputs).transpose(0, 1)


# Hyperbolic GRU for Event TempRel Extraction, without fine-tuning RoBERTa. Use static output directly
class EventTempRel_HGRU_static(nn.Module):
    def __init__(self, device, granularity=0.05, common_sense_emb_dim=64, bigramStats=1,
                dim_in=768, dim_hidden=128, dim_out=64,
                 num_class=4, non_lin='id', dropout=0., dtype=th.float64, **kwargs):
        super().__init__(**kwargs)
        self.dim_in = dim_in # dimsionality of input: Pre-trained LM's hidden state
        self.dim_out = dim_out
        self.dim_hidden = dim_hidden
        self.num_class = num_class
        self.non_lin = non_lin
        self.device = device
        self.dtype = dtype

        # Commonsense Knowledge related hyperparameters
        self.bigramStats_dim = bigramStats
        self.granularity = granularity
        self.common_sense_emb_dim = common_sense_emb_dim
        self.common_sense_emb = nn.Embedding(int(1.0/self.granularity)*self.bigramStats_dim, self.common_sense_emb_dim)

        self.ball = gt.Stereographic(-1) # declare a Poincare model using Geoopt's Stereographic class. The curvature is -1

        if dropout > 0.:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = None
        
        self.HyperGRU = HyperGRU(input_size=self.dim_in, hidden_size=self.dim_hidden,
                                ball=self.ball, default_dtype=self.dtype)

        self.W_ff_u = nn.Parameter(data=th.zeros(self.dim_out, self.dim_hidden))
        self.W_ff_v = nn.Parameter(data=th.zeros(self.dim_out, self.dim_hidden))
        nn.init.uniform_(self.W_ff_u, -1.0/(self.dim_hidden+self.dim_out), 1.0/(self.dim_hidden+self.dim_out))
        nn.init.uniform_(self.W_ff_v, -1.0/(self.dim_hidden+self.dim_out), 1.0/(self.dim_hidden+self.dim_out))

        self.b_ff = nn.Parameter(data=th.zeros(self.dim_out))    # zero initialize
        self.b_ff_d = nn.Parameter(data=th.zeros(self.dim_out))

        self.p_mlr = nn.Parameter(data=th.zeros(self.num_class, self.dim_out))    # should these be hyperbolic parameters?
        self.a_mlr = nn.Parameter(data=th.zeros(self.num_class, self.dim_out))    # these parameters are on tangent space
        nn.init.uniform_(self.a_mlr, -1.0/self.dim_out, 1.0/self.dim_out)
        nn.init.uniform_(self.p_mlr, -1.0/self.dim_out, 1.0/self.dim_out)

        self.W_ff_common = nn.Parameter(data=th.zeros(self.dim_out, self.bigramStats_dim*self.common_sense_emb_dim))
        nn.init.uniform_(self.W_ff_common, -1.0/(self.dim_out+self.bigramStats_dim*self.common_sense_emb_dim), 1.0/(self.dim_out+self.bigramStats_dim*self.common_sense_emb_dim))

        self.loss = nn.CrossEntropyLoss(reduction='none')
        self.hyper_para = [self.p_mlr, self.b_ff, self.b_ff_d, self.common_sense_emb.weight, self.W_ff_u, self.W_ff_v, self.W_ff_common] + list(self.HyperGRU.parameters())
        self.euclid_para = [self.a_mlr]

    def forward(self, sequence, mask1, mask2, common_ids):
        # common_ids [batch, 2]
        common_emb = self.common_sense_emb(common_ids) # [batch,2,common_dim]
        common_emb = common_emb.view(common_emb.size(0),-1)

        # sequence [batch, len, embeddings]
        projected = self.ball.expmap0(sequence)
        hidden = self.HyperGRU(projected)
        
        u = hidden * mask1 # boardcast
        v = hidden * mask2
        u = th.sum(u, dim=1)
        v = th.sum(v, dim=1)
        dsq = self.ball.dist(u, v).unsqueeze(1)

        # fully connected layers (concatenation)
        ffnn_u = self.ball.mobius_matvec(self.W_ff_u, u)
        ffnn_v = self.ball.mobius_matvec(self.W_ff_v, v)
        output_ffnn = self.ball.mobius_add(ffnn_u, ffnn_v)
        output_ffnn = self.ball.mobius_add(output_ffnn, self.b_ff)

        # extra feature: distance between u and v
        output_ffnn = self.ball.mobius_add(output_ffnn, self.ball.mobius_scalar_mul(dsq, self.b_ff_d))

        # CSE
        output_ffnn = self.ball.mobius_add(output_ffnn, self.ball.mobius_matvec(self.W_ff_common, common_emb))

        # non-linear
        output_ffnn = self._non_lin(self.ball.logmap0(output_ffnn))

        if self.dropout:
            output_ffnn = self.dropout(output_ffnn)

        output_ffnn = self.ball.expmap0(output_ffnn)
        logits = self._compute_mlr_logits(output_ffnn)

        return logits

    def _non_lin(self, vector):
        if self.non_lin == 'id':
            return vector
        elif self.non_lin == 'relu':
            return th.nn.functional.relu(vector)
        elif self.non_lin == 'tanh':
            return th.nn.functional.tanh(vector)
        elif self.non_lin == 'sigmoid':
            return th.nn.functional.sigmoid(vector)
    
    def _compute_mlr_logits(self, output_before):
        logits = []
        for cl in range(self.num_class):
            minus_p_plus_x = self.ball.mobius_add(-self.p_mlr[cl], output_before)    # [batch, hidden]
            norm_a = th.norm(self.a_mlr[cl])
            lambda_px = self._lambda(minus_p_plus_x)    # [batch, 1]
            px_dot_a = th.sum(minus_p_plus_x * nn.functional.normalize(self.a_mlr[cl].unsqueeze(0), p=2), dim=1)   # [batch, 1]
            logit = 2. * norm_a * th.asinh(px_dot_a * lambda_px)
            logits.append(logit)
        
        logits = th.stack(logits, axis=1)
        return logits
    
    @staticmethod
    def _lambda(vector):
        return 2. / (1-th.sum(vector * vector, dim=1))


# This class includes RoBERTa parameters, fine-tuning RoBERTa
class EventTempRel_HGRU(nn.Module):
    def __init__(self, device, granularity=0.05, common_sense_emb_dim=64, bigramStats=1,
                dim_in=768, dim_hidden=128, dim_out=64, model_name='roberta-base',
                 num_class=4, non_lin='id', dropout=0., dtype=th.float64, **kwargs):
        super().__init__(**kwargs)
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dim_hidden = dim_hidden
        self.num_class = num_class
        self.non_lin = non_lin
        self.device = device
        self.dtype = dtype

        self.model_name = model_name
        self.encoder_model = AutoModel.from_pretrained(self.model_name, return_dict=True)
        self.encoder_model.train()

        self.bigramStats_dim = bigramStats
        self.granularity = granularity
        self.common_sense_emb_dim = common_sense_emb_dim
        self.common_sense_emb = nn.Embedding(int(1.0/self.granularity)*self.bigramStats_dim,self.common_sense_emb_dim)

        self.ball = gt.Stereographic(-1) # declare a Poincare model using Geoopt's Stereographic class. The curvature is -1

        if dropout > 0.:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = None
        
        self.HyperGRU = HyperGRU(input_size=self.dim_in, hidden_size=self.dim_hidden,
                                ball=self.ball, default_dtype=self.dtype)

        self.W_ff_u = nn.Parameter(data=th.zeros(self.dim_out, self.dim_hidden))
        self.W_ff_v = nn.Parameter(data=th.zeros(self.dim_out, self.dim_hidden))
        nn.init.uniform_(self.W_ff_u, -1.0/(self.dim_hidden+self.dim_out), 1.0/(self.dim_hidden+self.dim_out))
        nn.init.uniform_(self.W_ff_v, -1.0/(self.dim_hidden+self.dim_out), 1.0/(self.dim_hidden+self.dim_out))

        self.b_ff = nn.Parameter(data=th.zeros(self.dim_out))    # zero initialize
        self.b_ff_d = nn.Parameter(data=th.zeros(self.dim_out))

        self.p_mlr = nn.Parameter(data=th.zeros(self.num_class, self.dim_out))    # should these be hyperbolic parameters?
        self.a_mlr = nn.Parameter(data=th.zeros(self.num_class, self.dim_out))    # these parameters are on tangent space
        nn.init.uniform_(self.a_mlr, -1.0/self.dim_out, 1.0/self.dim_out)
        nn.init.uniform_(self.p_mlr, -1.0/self.dim_out, 1.0/self.dim_out)

        self.W_ff_common = nn.Parameter(data=th.zeros(self.dim_out, self.bigramStats_dim*self.common_sense_emb_dim))
        nn.init.uniform_(self.W_ff_common, -1.0/(self.dim_out+self.bigramStats_dim*self.common_sense_emb_dim), 1.0/(self.dim_out+self.bigramStats_dim*self.common_sense_emb_dim))

        self.loss = nn.CrossEntropyLoss(reduction='none')
        self.hyper_para = [self.p_mlr, self.b_ff, self.b_ff_d, self.common_sense_emb.weight, self.W_ff_u, self.W_ff_v, self.W_ff_common] + list(self.HyperGRU.parameters())
        self.euclid_para = [self.a_mlr]
        self.plm_para = self.encoder_model.parameters()

    def forward(self, sequence, s_a_mask, mask1, mask2, common_ids):
        # common_ids [batch, 2]
        common_emb = self.common_sense_emb(common_ids) # [batch,2,common_dim]
        common_emb = common_emb.view(common_emb.size(0),-1)

        # sequence [batch, len, embeddings]
        encoded = self.encoder_model(sequence, s_a_mask).last_hidden_state
        projected = self.ball.expmap0(encoded)
        hidden = self.HyperGRU(projected)
        
        u = hidden * mask1 # boardcast
        v = hidden * mask2
        u = th.sum(u, dim=1)
        v = th.sum(v, dim=1)
        dsq = self.ball.dist(u, v).unsqueeze(1)

        # fully connected layers (concatenation)
        ffnn_u = self.ball.mobius_matvec(self.W_ff_u, u)
        ffnn_v = self.ball.mobius_matvec(self.W_ff_v, v)
        output_ffnn = self.ball.mobius_add(ffnn_u, ffnn_v)
        output_ffnn = self.ball.mobius_add(output_ffnn, self.b_ff)

        # extra feature: distance between u and v
        output_ffnn = self.ball.mobius_add(output_ffnn, self.ball.mobius_scalar_mul(dsq, self.b_ff_d))

        # CSE
        output_ffnn = self.ball.mobius_add(output_ffnn, self.ball.mobius_matvec(self.W_ff_common, common_emb))

        # non-linear
        output_ffnn = self._non_lin(self.ball.logmap0(output_ffnn))

        if self.dropout:
            output_ffnn = self.dropout(output_ffnn)

        output_ffnn = self.ball.expmap0(output_ffnn)
        logits = self._compute_mlr_logits(output_ffnn)

        return logits

    def _non_lin(self, vector):
        if self.non_lin == 'id':
            return vector
        elif self.non_lin == 'relu':
            return th.nn.functional.relu(vector)
        elif self.non_lin == 'tanh':
            return th.nn.functional.tanh(vector)
        elif self.non_lin == 'sigmoid':
            return th.nn.functional.sigmoid(vector)
    
    def _compute_mlr_logits(self, output_before):
        logits = []
        for cl in range(self.num_class):
            minus_p_plus_x = self.ball.mobius_add(-self.p_mlr[cl], output_before)    # [batch, hidden]
            norm_a = th.norm(self.a_mlr[cl])
            lambda_px = self._lambda(minus_p_plus_x)    # [batch, 1]
            px_dot_a = th.sum(minus_p_plus_x * nn.functional.normalize(self.a_mlr[cl].unsqueeze(0), p=2), dim=1)   # [batch, 1]
            logit = 2. * norm_a * th.asinh(px_dot_a * lambda_px)
            logits.append(logit)
        
        logits = th.stack(logits, axis=1)
        return logits
    
    @staticmethod
    def _lambda(vector):
        return 2. / (1-th.sum(vector * vector, dim=1))
