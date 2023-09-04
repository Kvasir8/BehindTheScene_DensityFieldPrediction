## d_model=num_features
# num_features = 2048  ## for resnet50 / for dim of feedforward network model for vanilla Transformer
# num_features = 512  ## for resnet34

import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import models.common.model.mlp as mlp

"""
# BTS model: The BTS model is used to predict the density field for each view of the input images. 
The predicted density fields are stored in a list.

# Stacking density fields: The predicted density fields from the BTS model are stacked along a new dimension, 
creating a tensor of shape (batch_size, num_views, height, width).

# Flattening and embedding: The density fields tensor is reshaped to (batch_size, num_views, height * width), e.g. features are stacked along the row, 
and then passed through an embedding layer that converts the density field values into a suitable format for the Transformer. 
The embedding layer is a linear layer that maps the input features to the desired dimension d_model.

# Transformer encoder: The embedded features are processed by a Transformer encoder, which consists of multiple layers
 of multi-head self-attention and feedforward sublayers. The Transformer encoder is designed to learn and capture
  relationships between the multiple views by attending to the most relevant parts of the input features. (density field as geometric consistency)
  The output of the Transformer encoder has the same shape as the input, (batch_size, num_views, d_model).

# Density field prediction: The transformed features are passed through a density field prediction layer, 
which is a sequential model containing a linear layer followed by a ReLU activation function. This layer predicts 
the accumulated density field for each pixel. The output shape is (batch_size, num_views, 1).

# Reshaping: The accumulated density field tensor is reshaped back to its original spatial dimensions (batch_size, height, width).
"""


## remark: hyper-params. e.g. 'nhead' could be tuned e.g. random- or grid search for future tuning strategy: hparams has less params in overfitting, and it should be normally trained when it comes to training normally e.g. dim_feedforward=2048. Hence it's required to make setting of overfitting param and normal setting param
class DensityFieldTransformer(nn.Module):
    def __init__(self, d_model=103, att_feat=32, nhead=1, num_layers=4, feat_pad=True, DFEnlayer=True,
                 nry=False, AE=False, do_=0., emb_enc=False, rb_=2048, ren_nc=64, B_=1):  ## dim_feedforward==input_feature Map_spatial_flattened_dim
        """ AE=mlp.ConvAutoEncoder | None
        :param d_model: (input features) Dimension of the token embeddings. In our case, it's the size of features (combined with positional encoding and feature map) to set size of input and output features for Transformer encoder layers, as well as the input for the final density field prediction layer. i.e. to specify the number of expected features in the input and output. Dimensionality of the input and output of the Transformer model. i.e. embedding dimension
        :param att_feat: attention_features, the dimension of the feedforward network model for embedding layer of the transformer (default=32)
        :param nhead: number of heads in the multi-head attention models (Note: att_feat(embed_dim) must be divisible by num_heads)
        :param num_layers: The number of sub-encoder-layers in the encoder. For GeoNeRF's arch, default: 4
        :param feat_pad: flag for feature to pad
        :param DFEnlayer: flag for replacing encoder layer with the IBRNet's encoder
        :param AE: flag for density field prediction for MLP layer of readout token with GeoNeRF's AutoEncoder layer for aggregation of view independent tokens (for experiment)
        :param do_: dropout ratio to randomly zero out the valid sampled_features' matrix
        :param rb_: ray_batch_size
        :param ren_nc: n_coarse == rendering number of coarse sampling for NeRF
        :param B_: batch size for a patch base sampling
        """
        super(DensityFieldTransformer, self).__init__()
        self.padding_flag,  self.emb_enc = feat_pad, emb_enc
        if emb_enc == "pwf":    self.emb_encoder = mlp.PoswiseFF_emb4enc(d_model, 2*att_feat, att_feat)
        elif emb_enc == "hpwf":
            self.emb_encoder = nn.Sequential( ## == mlp.PositionwiseFeedForward
                nn.Linear(d_model, 2 * att_feat, bias=True),
                nn.ELU(),
                nn.LayerNorm(d_in, eps=1e-6),
                nn.Linear(2 * att_feat, att_feat, bias=True)
            )
        elif emb_enc == "ff":   self.emb_encoder = nn.Sequential(nn.Linear(d_model, 2 * att_feat, bias=True), nn.ELU(), nn.Linear(2 * att_feat, att_feat, bias=True)) ## default: ReLU |  nn.LeakyReLU()
        else:   print("__unrecognized input for emb_enc")

        self.DFEnlayer, self.nry = DFEnlayer, nry
        self.att_feat = att_feat
        self.AE = AE
        self.dropout = nn.Dropout(do_)
        # self.ts_ = rb_ * ren_nc * B_  ## total num sampling (to decide input dimension for AE)
        self.rb_, self.B_ = rb_, B_
        self.n_coarse = ren_nc  ## Note: we assume patch size is 8x8, thus we have following ts_ as computation
        # self.ts_ = B_ * ren_nc * (8 * 8) * (rb_ // (8 * 8))  ## total num sampled points (to decide input dimension for AE)
        # self.S_ = 64  ## length of sequence for AE

        if self.DFEnlayer:
            self.transformer_enlayer = mlp.EncoderLayer(att_feat, att_feat, nhead, att_feat, att_feat)
            self.transformer_encoder = mlp.TrEnLayer(self.transformer_enlayer,num_layers)  ## TODO: replace MHA module with IBRNet network and complete integretable encoder part of transformer
        else:
            self.transformer_enlayer = TransformerEncoderLayer(att_feat, nhead, dim_feedforward=att_feat,batch_first=True)
            self.transformer_encoder = TransformerEncoder(self.transformer_enlayer, num_layers)

        if not self.nry: self.readout_token = nn.Parameter(torch.rand(1, 1, att_feat).to("cuda"), requires_grad=True)  ## ? # self.readout_token = torch.rand(1, 1, d_model).to("cuda") ## instead of dummy

        if self.AE: self.ConvAE = mlp.ConvAutoEncoder(self.att_feat, self.n_coarse)  ## [1, 2*self.att_feat, self.ts_] ## self.att_feat*2 ## self.ts_ ##(patch_size x ray_batch_size) self.att_feat, sampled_features.shape[0] or nv_+1 == 5 TODO: investigate more the model structure for validity in detail

        self.DF_pred_head = nn.Sequential(nn.Linear(self.att_feat,1))  ## Note: ReLU or Sigmoid would be detrimental for gradient flow at zero center activation function

    def forward(self, sampled_features, invalid_features, gfeat=None, iv_gfeat=None):  ### [n_, nv_, M, C1+C_pos_emb], [nv_==2, M==100000, C==1]
        ## invalid_features: invalid features to mask the features to let model learn without occluded points in the camera's view
        assert isinstance(invalid_features, torch.Tensor), f"__The {invalid_features} is not a torch.Tensor."
        assert invalid_features.dtype == torch.bool, f"The elements of the {invalid_features} are not boolean."
        # invalid_features = (invalid_features > 0.5)  ## round the each of values of 3D points simply by step function within the range of std_var [0,1]

        if self.dropout:
            invalid_features = 1 - self.dropout((1 - invalid_features.float()))  ## TODO: after dropping out, the values of elements are 2 somehow why?? ## randomly zero out the valid sampled_features' matrix. i.e. (1-invalid_features)

        # self.readout_token = nry.flatten(0,1)   ## if nry is enabled, but this doesnt work: TypeError: cannot assign 'torch.cuda.FloatTensor' as parameter 'readout_token' (torch.nn.Parameter or None expected)

        if self.emb_enc:    encoded_features = self.emb_encoder(sampled_features.flatten(0, -2)).reshape(sampled_features.shape[:-1] + (-1,))  ### [M*n==100000, nv_==6, 32]   ## Embedding to Transformer arch.
        else:               encoded_features = sampled_features.flatten(0, -2).reshape(sampled_features.shape[:-1] + (-1,))

        ## Process the embedded features with the Transformer
        if self.padding_flag:
            if self.nry:
                padded_features = torch.concat([gfeat.unsqueeze(1), encoded_features], dim=1)  ### (B*n_pts, nv_+1, 103) == ([100000, 2+1, 103]): padding along the column ## Note: needs to be fixed for nicer way
                padded_invalid  = torch.concat([iv_gfeat, invalid_features], dim=1, )
            elif not self.nry:
                padded_features = torch.concat([self.readout_token.expand(encoded_features.shape[0], -1, -1), encoded_features],dim=1)  ### (B*n_pts, nv_+1, 103) == ([100000, 2+1, 103]): padding along the column ## Note: needs to be fixed for nicer way
                padded_invalid  = torch.concat([torch.zeros(invalid_features.shape[0], 1, device="cuda"), invalid_features], dim=1, )
            else:                print("__unrecognizable nry condition")


            transformed_features = self.transformer_encoder(padded_features, src_key_padding_mask=padded_invalid)  ### masking dim(features) ([100000 * B, 1+nv_, 103]) with invalid padding [100000, 3])    ## self.transformer_enlayer for one encoder layer
            # transformed_features = self.transformer_enlayer(padded_features)  ### masking dim(features) ([100000 * B, 1+nv_, 103]) with invalid padding [100000, 3])
        else:
            invalid_features = invalid_features.squeeze(-1).permute(1, 0)
            transformed_features = self.transformer_encoder(encoded_features, src_key_padding_mask=invalid_features)  ### [100000, nv_==2, 103]

        if self.AE:
            # aggregated_features = transformed_features[:,0,:]  ### [M=100000 * nv_+1 ,att_feat==32]  ## first token refers to the readout token where it stores the feature information accumulated from the layers    # aggregated_features = self.attention(self.query.expand(transformed_features.shape[0], -1, -1), transformed_features, transformed_features, key_padding_mask=invalid_features)[0]
            aggregated_features = self.ConvAE(transformed_features[:, 0, :].view(-1, self.n_coarse, self.att_feat).transpose(1, 2) # n_rays, C, pts_per_ray
                ).transpose(1, 2).view(-1, self.att_feat) # n_pts, C
        else:
            # aggregated_features = self.ConvNet2AE(transformed_features[:, 0, :].view(-1, self.n_coarse, self.att_feat).transpose(1, 2)) ## .transpose(0,2)     ## .view(-1, self.att_feat)  # C_:=Channels TODO: feeding it into AE
            aggregated_features = transformed_features[:,0,:]  # n_pts, C  ## first token refers to the readout token where it stores the feature information accumulated from the layers    # aggregated_features = self.attention(self.query.expand(transformed_features.shape[0], -1, -1), transformed_features, transformed_features, key_padding_mask=invalid_features)[0]
        ## TODO: GeoNeRF: Identify readout token belongs to single ray: M should be divisable by nhead, so that it can feed into AE, Note: make sure sampled points are in valid in the mask. (camera frustum)
        ## !TODO: Q K^T V each element of which is a density field prediction for a corresponding 3D point.
        density_field = self.DF_pred_head(aggregated_features)  ## TODO: This should be 2 MLPs after AE's prediction. # .view(-1)  ### torch.Size([100000])
        # density_field = torch.nan_to_num(density_field, 0.0)
        # !!! BAD example below, see (https://pytorch.org/docs/stable/notes/autograd.html#in-place-correctness-checks) for more details
        # density_field[torch.all(invalid_features, dim=0)[:, 0], 0, 0] = 0

        # ! This might be an alternative to the padding.
        if False:
            final_output = torch.zeros_like(density_field)
            mask = torch.logical_not(torch.all(invalid_features, dim=0))[:, 0]
            final_output[mask, 0, 0] = density_field[mask, 0, 0]
            return final_output.mean(dim=1).squeeze(-1)

        # print("__dim(density_field): ", density_field.shape)
        return density_field
