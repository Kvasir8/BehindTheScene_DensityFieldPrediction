import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torchvision.models import resnet50

from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch import optim
from torch.autograd import Variable

from models.common.backbones.image_encoder import ImageEncoder as bts

## d_model=num_features
num_features = 2048  ## for resnet50 / for dim of feddforward netowrk model for vanilla Transformer
# num_features = 512  ## for resnet34

import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

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

## TODO: hyper-params. e.g. 'nhead' could be tuned e.g. random- or grid search for future tuning strategy: hparams has less params in overfitting, and it should be normally trained when it comes to training normally e.g. dim_feedforward=2048. Hence it's required to make setting of overfitting param and normal setting param
class DensityFieldTransformer(nn.Module):
    def __init__(
        self, in_features=103, attention_features=32, nhead=1, num_layers=6, feature_pad=True
    ):  ## dim_feedforward==input_feature Map_spatial_flattened_dim
        """
        :param d_model: Dimension of the token embeddings. In our case, it's the size of features (combined with positional encoding and feature map) to set size of input and output features for Transformer encoder layers, as well as the input for the final density field prediction layer. i.e. to specify the number of expected features in the input and output. Dimensionality of the input and output of the Transformer model. i.e. embedding dimension
        :param nhead: number of heads in the multi-head attention models (Note: embed_dim must be divisible by num_heads)
        :param num_layers: The number of sub-encoder-layers in the encoder. For standard Transformer arch, defualt: 6
        :param dim_feedforward: Dimension of the feedforward network model
        """
        super(DensityFieldTransformer, self).__init__()
        self.padding_flag = feature_pad
        # self.n_ = mlp_input.shape[0]
        # self.d_model = mlp_input.shape[-1]

        self.encoder = nn.Sequential(nn.Linear(in_features, 2*attention_features, bias=True), nn.ReLU(), nn.Linear(2*attention_features, attention_features, bias=True))

        ## Transformer encoder layers
        self.transformer_layer = TransformerEncoderLayer(
            attention_features, nhead, dim_feedforward=attention_features, batch_first=True
        )  ### (n, nv_==seq, features)
        self.transformer_encoder = TransformerEncoder(self.transformer_layer, num_layers)
        self.readout_token = nn.Parameter(torch.rand(1, 1, attention_features).to("cuda"), requires_grad=True)  ## ?
        # self.readout_token = torch.rand(1, 1, attention_features).to("cuda")  ## ?

        self.density_field_prediction = nn.Sequential(
            nn.Linear(attention_features, 1)
        )  ## Note: ReLU or Sigmoid would be detrimental for gradient flow at zero center activation function

    def forward(self, sampled_features, invalid_features):  ### [n_, nv_, M, C1+C_pos_emb], [nv_==2, M==100000, C==1]
        ## invalid_features: invalid features to mask the features to let model learn without occluded points in the camera's view

        assert isinstance(invalid_features, torch.Tensor), f"__The {invalid_features} is not a torch.Tensor."
        invalid_features = (
            invalid_features > 0.5
        )  ## round the each of values of 3D points simply by step function within the range of std_var [0,1]
        assert invalid_features.dtype == torch.bool, f"The elements of the {invalid_features} are not boolean."

        # embedded_features = self.in_embedding(sampled_features)  # Embedding to Transformer arch.
        encoded_features = self.encoder(sampled_features.flatten(0, -2)).reshape(sampled_features.shape[:-1] + (-1,))

        ## Process the embedded features with the Transformer    ## TODO: interchangeable into the code snippet in models_bts.py to make comparison with vanilla vs modified (e.g. tranforemr or VAE, pos_enc, mlp, layers, change)
        if self.padding_flag:
            padded_features = torch.concat(
                [self.readout_token.expand(encoded_features.shape[0], -1, -1), encoded_features], dim=1
            )  ### (B*n_pts, nv_+1, 103) == ([100000, 3, 103]): padding along the column ## Note: needs to be fixed for nicer way
            padded_invalid = torch.concat(
                [torch.zeros(invalid_features.shape[0], 1, device="cuda"), invalid_features],
                dim=1,
            )  ### [100000, 3]
            transformed_features = self.transformer_encoder(
                padded_features, src_key_padding_mask=padded_invalid
            )  ### masking dim(features) ([100000 * B, 1+nv_, 103]) with invalid padding [100000, 3])
            invalid_features = padded_invalid
        else:
            invalid_features = invalid_features.squeeze(-1).permute(1, 0)
            transformed_features = self.transformer_encoder(
                encoded_features, src_key_padding_mask=invalid_features
            )  ### [100000, nv_==2, 103]

        aggregated_features = transformed_features[
            :, 0, :
        ]  # 100000*B,103  ## ? key and values are roughly defined, which needs to be specified?
        ### MultiheadAtten( dim(Q)=(1,1,103), dim(K)=(n*n_pts,nv_,103), dim(V)=(n*n_pts,nv_,103) ) ### torch.Size([100000, 1, 103])

        # transformed_features = self.transformer_encoder(embedded_features, src_key_padding_mask=invalid_features[..., 0].permute(1, 0))
        # aggregated_features = self.attention(self.query.expand(transformed_features.shape[0], -1, -1), transformed_features, transformed_features, key_padding_mask=invalid_features[..., 0].permute(1, 0))[0]

        ## !TODO: Q K^T V each element of which is a density field prediction for a corresponding 3D point.
        density_field = self.density_field_prediction(aggregated_features)  ### torch.Size([100000])
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


# class DensityFieldTransformer1(nn.Module):
#     def __init__(self, input_img, backbone, bts_model, d_model, nhead = 16, num_layers):
#         """
#         :param input_img:
#         :param bts_model:
#         :param backbone Backbone network. Assumes it is resnet* e.g. resnet34 | resnet50
#         :param bts_model:
#         :param d_model: to set size of input and output features for Transformer encoder layers, as well as the input for the final density field prediction layer. i.e. to specify the number of expected features in the input and output. Dimensionality of the input and output of the Transformer model. i.e. embedding dimension
#         :param nhead: number of heads in the multi-head attention mechanisms
#         :param num_layers: the number of Transformer encoder layers in the neural network
#         """
#         super(DensityFieldTransformer1, self).__init__()
#
#         self.d_model = d_model
#
#         # Feature extraction (using a pre-trained CNN or other method)
#         resnet = models.resnet50(pretrained=True)
#         self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])  # remove last fully connected layer
#
#
#         C_, H_, W_ = input_img.shape
#         self.bts_model = bts_model
#
#         ## Embedding layer to convert image features to a suitable format for the Transformer: The extracted features are then flattened and embedded into a suitable format for the Transformer encoder.
#         self.embedding = nn.Linear(C_ * H_ * W_, d_model)   ##TODO: This is wrong => we need one pixel from feature map and concatenate multiview into matrix form then embed
#
#         self.transformer_layer = TransformerEncoderLayer(d_model, nhead)
#         self.transformer_encoder = TransformerEncoder(self.transformer_layer, num_layers)
#
#         self.density_field_prediction = nn.Sequential(
#             nn.Linear(d_model, 1),
#             nn.ReLU()
#         )
#
#     def forward(self, x_multiview):
#         """
#         :param x_multiview: (batch_size, num_views, 3, 224, 224)
#         :return:
#         """
#         # Extract features from the input image
#         features = self.feature_extractor(x)            # apply the remaining layers to the input image x to extract the features
#         features = features.view(features.size(0), -1)  # to flatten the 2D feature map into a 1D feature vector that can be input to the embedding layer.
#
#         # Extract features for each view using BTS model
#         density_fields = []
#         for view_idx in range(x_multiview.shape[1]):
#             x_single_view = x_multiview[:, view_idx]
#             density_field_single_view = self.bts_model(x_single_view)
#             density_fields.append(density_field_single_view)
#
#         density_fields = torch.stack(density_fields, dim=1)  # (batch_size, num_views, H, W)
#
#         # Flatten and embed the features
#         batch_size, num_views, height, width = density_fields.shape
#         features = density_fields.view(batch_size, num_views, -1)
#         embedded_features = self.embedding(features)
#
#         # Process the embedded features with the Transformer
#         transformed_features = self.transformer_encoder(embedded_features)
#
#         # Predict the accumulated density field for each pixel
#         accumulated_density_field = self.density_field_prediction(transformed_features)
#         accumulated_density_field = accumulated_density_field.view(batch_size, height, width)
#
#         return accumulated_density_field ## TODO: predict multiview_signma in this line from (self.sample_color, models_bts.py)
#
#
# class DensityFieldTransformer2(nn.Module):
#     def __init__(self, backbone="resnet34", d_model=num_features, nhead=16, num_layers=6)
#         """
#         :param backbone Backbone network. Assumes it is resnet* e.g. resnet34 | resnet50
#         :param d_model: to set size of input and output features for Transformer encoder layers, as well as the input for the final density field prediction layer. i.e. to specify the number of expected features in the input and output. Dimensionality of the input and output of the Transformer model. i.e. embedding dimension
#         :param nhead: number of heads in the multi-head attention mechanisms, default=8~16 c.f. Pytorch page
#         :param num_layers: the number of Transformer encoder layers in the neural network. For vanilla-Transformer, the number of layers is 6.
#         """
#         super(DensityFieldTransformer2, self).__init__()
#         self.d_model = d_model  ## This can be experimental value
#
#         ## Feature extraction
#         self.model = getattr(torchvision.models, backbone)(pretrained=pretrained)
#         # resnet = resnet50(pretrained=True)
#
#         ## Remove the last fully connected layer to get features from the penultimate layer
#         self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])   # last layer [-1] := slice that selects all elements of the list except for the last one
#
#         ## Embedding layer to convert image features to a suitable format for the Transformer
#         self.in_embedding = nn.Linear(num_features, d_model)    ## TODO: pixel from feature map from multi-views are concatenated
#
#         ## Transformer encoder layers
#         self.transformer_layer = TransformerEncoderLayer(d_model, nhead, batch_first=True)  # in- and output tensors: (B, num_views == seq, feature)
#         self.transformer_encoder = TransformerEncoder(self.transformer_layer, num_layers)
#
#         ## Output density field prediction layer    ## TODO: dim: (batch_size, num_pixels, 1)
#         self.density_field_prediction = nn.Sequential( nn.Linear(d_model, 1), nn.ReLU() )   ## predicts a single value for the density field per pixel (density value per pixel in the image)
#
#     # def encoder(self, images, poses):
#     #     self.poses = poses # (B,N,3,4), B:=Batch size, N:=#_views
#     #     self.feature_maps = self.feature_extractor(images) # (B,N,H,W,C1)
#
#     def forward(self, xyz=None, features, geo_info): ### TODO: if necessary, replace code with the sample_features @ models_bts.py
#         ## Extract features from the input image e.g. features = self.feature_extractor(x)
#         # uv, depth = projection(to_camera_coordinates(xyz, self.poses))  ## camera projection from pt.   # dim(xyz): (B,M), M:=#_pts.
#
#         # image_features = F.grid_sample(self.feature_maps, uv)  # (B,N,M,C1) ## TODO: bilinear interpolate the feature map to correspond to color pixel?
#
#         ## Concatenate image features and positional encoding
#         # features = torch.cat(image_features, positional_encoding(uv, depth)) # (B,N,M,C1+C3) ## == pos_enc == self.code_xyz
#         # single_view_bts(features[:, 0, :, :])   ## number of view in default is 1   ## orign BTS
#         ## Process the embedded features with the Transformer    ## TODO: interchangeable into the code snippet in models_bts.py to make comparison with vanilla vs modified (e.g. tranforemr or VAE, pos_enc, mlp, layers, change)
#         transformed_features = self.transformer_encoder(features) # (B,1,M,C2)
#
#         ## Predict the density field for each pixel
#         density = self.density_field_prediction((transformed_features, geo_info)) # (B,M) # geometrical_information maybe optional
#         return density  ## TODO: predict multiview_signma in this line from (self.sample_color, models_bts.py)
#
#         """etc"""
#         # # Flatten and embed the features (for whole image pixels which are not correct way for transformer arch to pass)
#         # batch_size, num_pixels = features.shape[:2]
#         # features = features.view(batch_size, num_pixels, -1)    ## uninterested feature maps taken into account
#         # embedded_features = self.embedding(features)
#         #
#         # # Process the embedded features with the Transformer
#         # transformed_features = self.transformer_encoder(embedded_features)
#         #
#         # # Predict the density field for each pixel
#         # density_field = self.density_field_prediction(transformed_features)
#         # return density_field
#
#
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
