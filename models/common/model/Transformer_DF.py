# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.optim import Adam
# from torchvision.models import resnet50
#
# from torch import optim
# from torch.autograd import Variable
#
# from models.common.backbones.image_encoder import ImageEncoder as bts


## d_model=num_features
num_features = 2048  ## for resnet50 / for dim of feddforward netowrk model for vanilla Transformer
# num_features = 512  ## for resnet34

import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import models.common.model.mlp as IBR   ## IBRNet

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
    def __init__(self, d_model=103, att_feat=103, nhead=1, num_layers=6, feature_pad=True, IBRNet=True):  ## dim_feedforward==input_feature Map_spatial_flattened_dim
        """
        :param d_model: (input features) Dimension of the token embeddings. In our case, it's the size of features (combined with positional encoding and feature map) to set size of input and output features for Transformer encoder layers, as well as the input for the final density field prediction layer. i.e. to specify the number of expected features in the input and output. Dimensionality of the input and output of the Transformer model. i.e. embedding dimension
        :param att_feat: attention_features, the dimension of the feedforward network model for embedding layer of the transformer (default=32)
        :param nhead: number of heads in the multi-head attention models (Note: att_feat(embed_dim) must be divisible by num_heads)
        :param num_layers: The number of sub-encoder-layers in the encoder. For standard Transformer arch, defualt: 6
        :param feature_pad: flag for feature to pad
        :param IBRNet: flag for replaceing encoder layer with the IBRNet's encoder
        """
        super(DensityFieldTransformer, self).__init__()
        self.padding_flag = feature_pad
        self.emb_encoder = nn.Sequential(nn.Linear(d_model, 2*att_feat, bias=True), nn.ReLU(), nn.Linear(2*att_feat, att_feat, bias=True))
        self.IBRNet = IBRNet

        ## DFTransformer encoder layers
        if self.IBRNet:
            self.transformer_enlayer = IBR.EncoderLayer(d_model, att_feat, nhead, att_feat, att_feat)
            self.transformer_encoder = IBR.TrEnLayer(self.transformer_enlayer, num_layers)    ## TODO: replace MHA module with IBRNet network and complete integratable encoder part of transformer
        else:
            self.transformer_enlayer = TransformerEncoderLayer(att_feat, nhead, dim_feedforward=att_feat, batch_first=True)
            self.transformer_encoder = TransformerEncoder(self.transformer_enlayer, num_layers)

        self.readout_token = nn.Parameter(torch.rand(1, 1, att_feat).to("cuda"), requires_grad=True)  ## ? # self.readout_token = torch.rand(1, 1, d_model).to("cuda") ## instead of dummy
        # self.readout_token = torch.rand(1, 1, att_feat).to("cuda")  ## ? # self.attention = nn.MultiheadAttention(d_model, nhead, batch_first=True)

        self.density_field_prediction = nn.Sequential(
            nn.Linear(att_feat, 1)
        )  ## Note: ReLU or Sigmoid would be detrimental for gradient flow at zero center activation function

    def forward(self, sampled_features, invalid_features):  ### [n_, nv_, M, C1+C_pos_emb], [nv_==2, M==100000, C==1]
        ## invalid_features: invalid features to mask the features to let model learn without occluded points in the camera's view
        assert isinstance(invalid_features, torch.Tensor), f"__The {invalid_features} is not a torch.Tensor."
        invalid_features = (invalid_features > 0.5)  ## round the each of values of 3D points simply by step function within the range of std_var [0,1]
        assert invalid_features.dtype == torch.bool, f"The elements of the {invalid_features} are not boolean."

        # embedded_features = self.in_embedding(sampled_features)  # Embedding to Transformer arch.
        encoded_features = self.emb_encoder(sampled_features.flatten(0, -2)).reshape(sampled_features.shape[:-1] + (-1,))   ### [M*n==100000, nv_==6, 32]

        ## Process the embedded features with the Transformer    ## TODO: interchangeable into the code snippet in models_bts.py to make comparison with vanilla vs modified (e.g. tranforemr or VAE, pos_enc, mlp, layers, change)
        if self.padding_flag:
            padded_features = torch.concat([self.readout_token.expand(encoded_features.shape[0], -1, -1), encoded_features], dim=1)  ### (B*n_pts, nv_+1, 103) == ([100000, 2+1, 103]): padding along the column ## Note: needs to be fixed for nicer way
            padded_invalid = torch.concat([torch.zeros(invalid_features.shape[0], 1, device="cuda"), invalid_features],dim=1,)  # invalid_features[...,0].permute(1,0) ### [M, num_features + one zero padding layer] == [6250, 96+1]
            # if self.IBRNet: transformed_features = self.transformer_enlayer(padded_features, slf_attn_mask=padded_invalid)  ### masking dim(features) ([100000 * B, 1+nv_, 103]) with invalid padding [100000, 3])
            if self.IBRNet: transformed_features = self.transformer_encoder(padded_features, src_key_padding_mask=padded_invalid)  ### masking dim(features) ([100000 * B, 1+nv_, 103]) with invalid padding [100000, 3])
            else: transformed_features = self.transformer_encoder(padded_features, src_key_padding_mask=padded_invalid)  ### masking dim(features) ([100000 * B, 1+nv_, 103]) with invalid padding [100000, 3])
            # transformed_features = self.transformer_enlayer(padded_features)  ### masking dim(features) ([100000 * B, 1+nv_, 103]) with invalid padding [100000, 3])
            invalid_features = padded_invalid
        else:
            invalid_features = invalid_features.squeeze(-1).permute(1, 0)
            transformed_features = self.transformer_encoder(encoded_features, src_key_padding_mask=invalid_features)  ### [100000, nv_==2, 103]

        aggregated_features = transformed_features[:,0,:]  # [M=100000, nv_+1 ,103]  ## first token refers to the readout token where it stores the feature information accumulated from the layers    # aggregated_features = self.attention(self.query.expand(transformed_features.shape[0], -1, -1), transformed_features, transformed_features, key_padding_mask=invalid_features)[0]
        # aggregated_features = transformed_features[0][:,0,:]  ## TODO: investigate matrices, the 2nd dimension [M=100000, nv_+1 ,3,3,3]

        ### MultiheadAtten( dim(Q)=(1,1,103), dim(K)=(n*n_pts,nv_,103), dim(V)=(n*n_pts,nv_,103) ) ### torch.Size([100000, 1, 103])

        # transformed_features = self.transformer_encoder(embedded_features, src_key_padding_mask=invalid_features[..., 0].permute(1, 0))
        # aggregated_features = self.attention(self.query.expand(transformed_features.shape[0], -1, -1), transformed_features, transformed_features, key_padding_mask=invalid_features[..., 0].permute(1, 0))[0]

        ## !TODO: Q K^T V each element of which is a density field prediction for a corresponding 3D point.
        density_field = self.density_field_prediction(aggregated_features)  # .view(-1)  ### torch.Size([100000])
        # density_field = torch.nan_to_num(density_field, 0.0)
        # !!! BAD example below, see (https://pytorch.org/docs/stable/notes/autograd.html#in-place-correctness-checks) for more details
        # density_field[torch.all(invalid_features, dim=0)[:, 0], 0, 0] = 0

        # ! This might be an alternative to the padding.
        if False:
            final_output = torch.zeros_like(density_field)
            mask = torch.logical_not(torch.all(invalid_features, dim=0))[:, 0]
            final_output[mask, 0, 0] = density_field[mask, 0, 0]
            return final_output.mean(dim=1).squeeze(-1)

        # density_field = self.transformer(sampled_features, src_key_padding_mask=invalid_features)   ## Masking invalid features
        # print("__dim(density_field): ", density_field.shape)
        return density_field
        # return density_field.mean(dim=1).squeeze(-1)    ## ! nn.Transformer documents to see the corresponding parameters
        # return density_field.reshape(self.n_, self.d_model)   ## reshaping of the accumulated_density_field tensor : (batch_size, num_density for number of points).


# class VAE(nn.Module):
#     """
#     encodes the input images into a compact latent representation.
#     This representation can then be used as input to the Transformer architecture,
#     which generates the density field prediction.
#     """
#     def __init__(self, ...):
#         #TODO Define encoder and decoder architecture
#
#     def encode(self, x):
#         #TODO Encode input image into a latent representation
#         return mu, log_var
#
#     def decode(self, z):
#         #TODO Decode latent representation into image features
#         return x_reconstructed
#
# class DensityFieldTransformerWithVAE(DensityFieldTransformer):
#     def __init__(self, d_model, nhead, num_layers):
#         super(DensityFieldTransformerWithVAE, self).__init__(d_model, nhead, num_layers)
#         self.vae = VAE(...)
#         del self.feature_extractor
#
#     def forward(self, x):
#         mu, log_var = self.vae.encode(x)
#         z = self.reparameterize(mu, log_var)
#         x_reconstructed = self.vae.decode(z)
#
#         #TODO Use the reconstructed image features as input to the Transformer architecture
#
# def combined_loss_function(output, target, depth_pred, latent_mean, latent_log_var):
#     """
#     :param output:
#     :param target:
#     :param depth_pred:
#     :param latent_mean:
#     :param latent_log_var:
#     :return:
#     """
#     lambda_p = 1.0
#     lambda_s = 0.1
#     lambda_r = 1.0
#     lambda_kl = 0.01
#
#     photometric_loss = compute_photometric_loss(output, target)
#     smoothness_loss = compute_smoothness_loss(depth_pred)
#
#     #(paper_equ.2) L_{NeRF_VAE(I,c,C;\theta,\phi)}
#     reconstruction_loss = compute_reconstruction_loss(output, target)
#     KL_divergence_loss = -0.5 * torch.sum(...) ## pseudo code
#
#     total_loss = (
#         lambda_p * photometric_loss
#         + lambda_s * smoothness_loss
#         + lambda_r * reconstruction_loss
#         + lambda_kl * KL_divergence_loss
#     )
#
#     return total_loss
#
#
#
#     @classmethod
#     def from_conf(cls, conf):
#         return cls(
#             conf.get_string("backbone"),
#             pretrained=conf.get_bool("pretrained", True),
#             latent_size=conf.get_int("latent_size", 128),
#         )
