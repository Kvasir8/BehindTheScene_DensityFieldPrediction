"""
Main model implementation
"""

import torch
import torch.autograd.profiler as profiler
import torch.nn.functional as F
from torch import nn

from models.common.backbones.backbone_util import make_backbone
from models.common.model.code import PositionalEncoding
from models.common.model.mlp_util import make_mlp

EPS = 1e-3

from models.common.model.Transformer_DF import DensityFieldTransformer


class MVBTSNet(torch.nn.Module):
    def __init__(self, conf):
        super().__init__()  ### inherits the initialization behavior from its parent class
        self.DFT = DensityFieldTransformer( conf.get("d_model"),conf.get("att_feat"),conf.get("nhead"),
            conf.get("num_layers"), conf.get("feature_pad"), conf.get("DFEnlayer"), conf.get("AE"))
        self.DFT_flag = conf.get("DFT_flag", True)
        self.nv_ = conf.get("nv_", "num_multiviews")
        self.test_sample = conf.get("test_sample", False)
        self.d_min, self.d_max = conf.get("z_near"), conf.get("z_far")
        self.learn_empty, self.empty_empty, self.inv_z = conf.get("learn_empty", True), conf.get("empty_empty", False), conf.get("inv_z", True)
        self.color_interpolation, self.code_mode = conf.get("color_interpolation", "bilinear"), conf.get("code_mode", "z")
        if self.code_mode not in ["z", "distance"]: raise NotImplementedError(f"Unknown mode for positional encoding: {self.code_mode}")

        self.encoder = make_backbone(conf["encoder"])
        self.code_xyz = PositionalEncoding.from_conf(conf["code"], d_in=3)
        self.flip_augmentation = conf.get("flip_augmentation", False)
        self.return_sample_depth = conf.get("return_sample_depth", False)
        self.sample_color = conf.get("sample_color", True)

        d_in = self.encoder.latent_size + self.code_xyz.d_out
        d_out = 1 if self.sample_color else 4
        ## If sample_color is set to False, then d_out is set to 4 to represent the RGBA color values
        ## (red, green, blue, alpha) of the reconstructed scene. If sample_color is set to True, then d_out is set to 1
        ## to represent the estimated depth value of the reconstructed scene.

        self._d_in, self._d_out = d_in, d_out
        self.mlp_coarse, self.mlp_fine = make_mlp(conf["mlp_coarse"], d_in, d_out=d_out), make_mlp(conf["mlp_fine"], d_in, d_out=d_out, allow_empty=True)

        if self.learn_empty: self.empty_feature = nn.Parameter(torch.randn((self.encoder.latent_size,), requires_grad=True))
        ## factor to multiply the output of the corresponding MLP in the forward, which helps to control the range of the output values from the MLP
        self._scale = 0

    def set_scale(self, scale): self._scale = scale
    def get_scale(self): return self._scale
    def compute_grid_transforms(self, *args, **kwargs): pass

    def encode(self, images, Ks, poses_c2w, ids_encoder=None, ids_render=None, images_alt=None, combine_ids=None):  ### ids_encoder:=which img beginning
        poses_w2c = torch.inverse(poses_c2w)    ## ! called from trainer_overfit.py to tell all images =>

        if ids_encoder is None:
            images_encoder = images
            Ks_encoder = Ks
            poses_w2c_encoder = poses_w2c
            ids_encoder = list(range(len(images)))
        else:
            images_encoder = images[:, ids_encoder]
            Ks_encoder = Ks[:, ids_encoder]
            poses_w2c_encoder = poses_w2c[:, ids_encoder]

        if images_alt is not None:
            images = images_alt
        else:
            images = images * .5 + .5

        if ids_render is None:
            images_render = images
            Ks_render = Ks
            poses_w2c_render = poses_w2c
            ids_render = list(range(len(images)))
        else:
            images_render = images[:, ids_render]
            Ks_render = Ks[:, ids_render]
            poses_w2c_render = poses_w2c[:, ids_render]

        if combine_ids is not None:
            combine_ids = list(list(group) for group in combine_ids)
            get_combined = set(sum(combine_ids, []))
            for i in range(images.shape[1]):
                if i not in get_combined:
                    combine_ids.append((i,))
            remap_encoder = {v: i for i, v in enumerate(ids_encoder)}
            remap_render = {v: i for i, v in enumerate(ids_render)}
            comb_encoder = [[remap_encoder[i] for i in group if i in ids_encoder] for group in combine_ids]
            comb_render = [[remap_render[i] for i in group if i in ids_render] for group in combine_ids]
            comb_encoder = [group for group in comb_encoder if len(group) > 0]
            comb_render = [group for group in comb_render if len(group) > 0]
        else:
            comb_encoder = None
            comb_render = None

        n_, nv_, c_, h_, w_ = images_encoder.shape   ### torch.Size([n, nv_, 3, 192, 640]) 3:=RGB
        c_l = self.encoder.latent_size

        if self.flip_augmentation and self.training: ## data augmentation for color TODO: analyze data augmentation
            do_flip = (torch.rand(1) > .5).item()
        else:
            do_flip = False

        if do_flip:
            images_encoder = torch.flip(images_encoder, dims=(-1, ))

        image_latents_ms = self.encoder(images_encoder.view(n_ * nv_, c_, h_, w_))

        if do_flip:
            image_latents_ms = [torch.flip(il, dims=(-1, )) for il in image_latents_ms]

        _, _, h_, w_ = image_latents_ms[0].shape
        image_latents_ms = [F.interpolate(image_latents, (h_, w_)).view(n_, nv_, c_l, h_, w_) for image_latents in image_latents_ms]    ### UserWarning: nn.functional.upsample is deprecated. Use nn.functional.interpolate instead

        self.grid_f_features = image_latents_ms
        self.grid_f_Ks = Ks_encoder
        self.grid_f_poses_w2c = poses_w2c_encoder
        self.grid_f_combine = comb_encoder

        self.grid_c_imgs = images_render
        self.grid_c_Ks = Ks_render
        self.grid_c_poses_w2c = poses_w2c_render
        self.grid_c_combine = comb_render

    def sample_features(self, xyz, use_single_featuremap=True): ## 2nd arg: to control whether multiple feature maps should be combined into a single feature map or not. If True, the function will average the sampled features from multiple feature maps along the view dimension (nv) before returning the result. This can be useful when you want to combine information from multiple views or feature maps into a single representation.
        n_, n_pts, _ = xyz.shape ## Get the shape of the input point cloud and the feature grid (n, pts, spatial_coordinate == 3)
        n_, nv_, c_, h_, w_ = self.grid_f_features[self._scale].shape       ### torch.Size([1, 4, 64, 192, 640])
        # if not use_single_featuremap:   nv_ = self.nv_
        xyz = xyz.unsqueeze(1)  # (n, 1, pts, 3)    ## Add a singleton dimension to the input point cloud to match grid_f_poses_w2c shape
        ones = torch.ones_like(xyz[..., :1])        ## Create a tensor of ones to add a fourth dimension to the point cloud for homogeneous coordinates
        xyz = torch.cat((xyz, ones), dim=-1)        ## Concatenate the tensor of ones with the point cloud to create homogeneous coordinates
        xyz_projected = ((self.grid_f_poses_w2c[:, :nv_, :3, :]) @ xyz.permute(0, 1, 3, 2))  ## Apply the camera poses to the point cloud to get the projected points and calculate the distance
        distance = torch.norm(xyz_projected, dim=-2).unsqueeze(-1)  ### [1, 2, 100000, 1]
        xyz_projected = (self.grid_f_Ks[:, :nv_] @ xyz_projected).permute(0, 1, 3, 2)        ## Apply the intrinsic camera parameters to the projected points to get pixel coordinates
        xy = xyz_projected[:, :, :, [0, 1]]         ## Extract the x,y coordinates and depth value from the projected points
        z_ = xyz_projected[:, :, :, 2:3]

        xy = xy / z_.clamp_min(EPS) ## Normalize the x,y coordinates by the depth value and check for invalid points    => image coord -> pixel coord
        invalid = (z_ <= EPS) | (xy[:, :, :, :1] < -1) | (xy[:, :, :, :1] > 1) | (xy[:, :, :, 1:2] < -1) | (xy[:, :, :, 1:2] > 1)
        '''given a vector p = (x, y, z) this is the difference of normalizing either:z ||p|| = sqrt(x^2 + y^2 + z^2). So you either give the network (x, y, z_normalized) or (x, y, ||p||_normalized) as input. It is just different parameterizations of the same point.'''
        if self.code_mode == "z":  ## Depending on the code mode, normalize the depth value or distance value to the [-1, 1] range and concatenate with the xy coordinates
            # Get z into [-1, 1] range  ## Normalizing the z coordinates leads to a consistent positional encoding of the viewing information. In line 172 the viewing information (xyz_projected) is given to a positional encoder before it is appended to the overall feature vector
            if self.inv_z:
                z_ = (1 / z_.clamp_min(EPS) - 1 / self.d_max) / (1 / self.d_min - 1 / self.d_max)
            else:
                z_ = (z_ - self.d_min) / (self.d_max - self.d_min)
            z_ = 2 * z_ - 1
            xyz_projected = torch.cat((xy, z_), dim=-1)      ## concatenates the normalized x, y, and z coordinates
        elif self.code_mode == "distance":
            if self.inv_z:
                distance = (1 / distance.clamp_min(EPS) - 1 / self.d_max) / (1 / self.d_min - 1 / self.d_max)
            else:
                distance = (distance - self.d_min) / (self.d_max - self.d_min)
            distance = 2 * distance - 1
            xyz_projected = torch.cat((xy, distance), dim=-1)   ## Apply the positional encoder to the concatenated xy and depth/distance coordinates (it enables the model to capture more complex spatial dependencies without a significant increase in model complexity or training data)
        xyz_code = self.code_xyz(xyz_projected.view(n_ * nv_ * n_pts, -1)).view(n_, nv_, n_pts, -1).permute(0, 2, 1, 3)   ## ! positional encoding dimension to check (concatenate)

        feature_map = self.grid_f_features[self._scale][:, :nv_] ## Extract the feature map corresponding to the current scale and view. i.e. extracting the features for the first nv_ views c.f. encoder in pipeline
        # These samples are from different scales
        if self.learn_empty:    ## "empty space" can refer to areas in a scene where there is no object, or it could also refer to areas that are not observed or are beyond the range of the sensor. This allows the model to have a distinct learned representation for "empty" space, which can be beneficial in tasks like 3D reconstruction where understanding both the objects in a scene and the empty space between them is important.
            empty_feature_expanded = self.empty_feature.view(1, 1, 1, c_).expand(n_, nv_, n_pts, c_)    ## trainable parameter, initialized with random features
        ## feature_map (2, 4, 64, 128, 128): n_ = 2, nv_ = 4 views, c_ = 64 channels in the feature map, and the height and width of the feature map are h = 128 and w = 128
        ## !TODO: for multiviews for F.grid_sample : xy.view(n_ * nv_, 1, -1, 2) To debug how xy looks like in order to integrate for multiview (by looking over doc in Pytorch regarding how to sample all frames)
        sampled_features = F.grid_sample(feature_map.view(n_ * nv_, c_, h_, w_), xy.view(n_ * nv_, 1, -1, 2), mode="bilinear", padding_mode="border", align_corners=False).view(n_, nv_, c_, n_pts).permute(0, 3, 1, 2)   ## Sample features using grid sampling and interpolate them using bilinear interpolation
        ## dim(sampled_features): (n_, nv_, n_pts, c_)
        if self.learn_empty:    ## Replace invalid features in the sampled features tensor with the corresponding features from the expanded empty feature
            sampled_features[invalid.expand(-1, -1, -1, c_)] = empty_feature_expanded[invalid.expand(-1, -1, -1, c_)] ## broadcasting and make it fit to feature map
        ## dim(xyz): (B,M), M:=#_pts.
        if self.test_sample:
            # For testing: Initialize the features e.g. 4 views, 100000 points per view, and 103 features per point
            sampled_features = torch.rand((4, 100000, 103))  # [num_views, num_points, num_features]
            # Reshape the features to match the input shape that the transformer expects: [num_points, num_views, num_features]
            sampled_features = sampled_features.permute(1, 0, 2)
        else: sampled_features = torch.cat((sampled_features, xyz_code), dim=-1)  ## Concatenate the sampled features and the encoded xyz coordinates, and then it will be passed to MLP
        ### dim(sampled_features): (n_, nv_, M, C1+C_pos_emb)
        sampled_features = sampled_features    # .squeeze(1) ### torch.Size([4, 100000, 103 == feats+pos_emb]) : dim(sampled_features): (nv, M, C1+C_pos_emb)
        # If there are multiple frames with predictions, reduce them.
        # TODO: Technically, this implementations should be improved if we use multiple frames.
        # The reduction should only happen after we perform the unprojection.

        ## Run Density Field Transformer Network to accumulate multi-views
        if False: ## self.DFT_flag ## TODO: integrate the code from DFT
            ## Process the embedded features with the Transformer    ## TODO: interchangeable into the code snippet in models_bts.py to make comparison with vanilla vs modified (e.g. tranforemr or VAE, pos_enc, mlp, layers, change)
            # input_featureMap_spatial_flattened_dim = len(sampled_features)
            # self.DFT = DensityFieldTransformer(input_featureMap_spatial_flattened_dim, )
            ## Accumulate multiviews
            # transformed_features = self.DFT.transformer_encoder(sampled_features)  # (B,1,M,C2)

            density_predictions = self.DFT(sampled_features, invalid)  ### dim(density): (B,1,M,C2)

            # for idx, DF_img in enumerate(sampled_features): # (n, nv, n_pts, c + xyz_code[...,-1])
            #     features = self.encoder(DF_img)
            #     # Extract features for DensityFieldTransformer
            #     if idx == 0:
            #         x_multiview = features.unsqueeze(1)
            #     else:
            #         x_multiview = torch.cat((x_multiview, features.unsqueeze(1)), dim=1)
            #
            # # Pass the accumulated multiviews to DensityFieldTransformer
            # sampled_features = self.DFT(x_multiview)

        '''allows the algorithm to select the best features among different views or groups of views, based on the invalid flags. It provides an additional level of flexibility for combining features from different views in a more controlled manner.'''
        # if self.grid_f_combine is not None: ## => there are specific groups of frames/views that need to be combined.
        #     invalid_groups = []             ## features that are out of camera's frustum or are out of range of positional encoding, [-1,1]
        #     sampled_features_groups = []
        #
        #     for group in self.grid_f_combine:
        #         if len(group) == 1:
        #             invalid_groups.append(invalid[:, group])
        #             sampled_features_groups.append(sampled_features[:, group])
        #
        #         invalid_to_combine = invalid[:, group]
        #         features_to_combine = sampled_features[:, group]    ## the code tries to combine the features from different views within the group
        #
        #         indices = torch.min(invalid_to_combine, dim=1, keepdim=True)[1] ## These indices indicate the best features among the different views, as they have the lowest(torch.min) invalid flags.
        #         invalid_picked = torch.gather(invalid_to_combine, dim=1, index=indices)     ## best invalid flags are also extracted
        #         features_picked = torch.gather(features_to_combine, dim=1, index=indices.expand(-1, -1, -1, features_to_combine.shape[-1])) ## similarly, best features are then extracted
        #         ## Once all groups have been processed:
        #         invalid_groups.append(invalid_picked)
        #         sampled_features_groups.append(features_picked)
        #
        #     invalid = torch.cat(invalid_groups, dim=1)
        #     sampled_features = torch.cat(sampled_features_groups, dim=1)

        # if use_single_featuremap:   ## ! compute the mean of the sampled features across the view dimension and check if any of the features are invalid.
        #     sampled_features = sampled_features.mean(dim=1) ### torch.Size([1, 100000, 103]) : mean(dim=1)==squeeze
        #     invalid = torch.any(invalid, dim=1) ## sampled_features are averaged into the same dim with single featuremap   ##         return sampled_features, invalid[..., 0].permute(0, 2, 1)    ## !! The output of the function is a tuple containing the sampled features and a boolean tensor indicating the invalid features

        return sampled_features, invalid[..., 0].permute(0, 2, 1)    ## !! The output of the function is a tuple containing the sampled features and a boolean tensor indicating the invalid features

    def sample_colors(self, xyz):
        n_, n_pts, _ = xyz.shape                     ## n := batch size, n_pts := #_points in world coord.
        n_, nv_, c_, h_, w_ = self.grid_c_imgs.shape     ## nv_ := #_views
        xyz = xyz.unsqueeze(1)                      # (n, 1, pts, 3)
        ones = torch.ones_like(xyz[..., :1])        ## create a tensor of ones with the same shape as the first two dimensions of (xyz), up to the third dimension, and add a trailing singleton dimension (shape: (n, 1)) e.g. (n, 1, pts, 1)
        xyz = torch.cat((xyz, ones), dim=-1)        ## concatenates the tensor of ones with xyz along the last dimension (i.e., the dimension representing the coordinates of each point).
        xyz_projected = ((self.grid_c_poses_w2c[:, :, :3, :]) @ xyz.permute(0, 1, 3, 2))    ## multiply the camera-to-world transformation matrices with the concatenated tensor (xyz) to get the projected coordinates of the points in the camera coordinate system (shape: (n, nv, 3, n_pts))
        distance = torch.norm(xyz_projected, dim=-2).unsqueeze(-1)  ## compute the Euclidean norm of the projected coordinates along the last dimension and add a trailing singleton dimension (shape: (n, nv, 1, n_pts))
        xyz_projected = (self.grid_c_Ks @ xyz_projected).permute(0, 1, 3, 2)    ## multiply the intrinsic camera matrices with the projected coordinates to get the pixel coordinates (shape: (n, nv, n_pts, 3) - 3rd dimension: x, y, and z coordinates in the pixel space)
        xy = xyz_projected[:, :, :, [0, 1]]         ## select only the x and y coordinates of the pixel coordinates (shape: (n, nv, n_pts, 2))
        z_ = xyz_projected[:, :, :, 2:3]             ## select only the z coordinate of the pixel coordinates (shape: (n, nv, n_pts, 1))

        # This scales the x-axis into the right range.
        xy = xy / z_.clamp_min(EPS)
        invalid = (z_ <= EPS) | (xy[:, :, :, :1] < -1) | (xy[:, :, :, :1] > 1) | (xy[:, :, :, 1:2] < -1) | (xy[:, :, :, 1:2] > 1)    ## Invalid points are points outside the image or points with invalid depth. This creates a boolean tensor of shape (n, nv, 1, n_pts), where each element is True if the corresponding point is invalid.

        sampled_colors = F.grid_sample(self.grid_c_imgs.view(n_ * nv_, c_, h_, w_), xy.view(n_ * nv_, 1, -1, 2), mode=self.color_interpolation, padding_mode="border", align_corners=False).view(n_, nv_, c_, n_pts).permute(0, 1, 3, 2)  ## Sample colors from the grid using the projected world coordinates.

        assert not torch.any(torch.isnan(sampled_colors))   ## Check that there are no NaN values in the sampled colors tensor.

        if self.grid_c_combine is not None:     ## If self.grid_c_combine is not None, combine colors from multiple points in the same group.
            invalid_groups = []
            sampled_colors_groups = []

            for group in self.grid_c_combine:   ## group:=list of indices that correspond to a subset of the total set of points in the point cloud. These subsets are combined to create a single image of the entire point cloud from multiple views.
                if len(group) == 1:     ## If the group contains only one point, append the corresponding invalid tensor and sampled colors tensor to the respective lists.
                    invalid_groups.append(invalid[:, group])
                    sampled_colors_groups.append(sampled_colors[:, group])
                    continue

                invalid_to_combine = invalid[:, group]        ## Otherwise, combine colors from the group by picking the color of the first valid point in the group.
                colors_to_combine = sampled_colors[:, group]

                indices = torch.min(invalid_to_combine, dim=1, keepdim=True)[1]         ## Get the index of the first valid point in the group.
                invalid_picked = torch.gather(invalid_to_combine, dim=1, index=indices) ## Pick the invalid tensor and sampled colors tensor corresponding to the first valid point in the group.
                colors_picked = torch.gather(colors_to_combine, dim=1, index=indices.expand(-1, -1, -1, colors_to_combine.shape[-1]))

                invalid_groups.append(invalid_picked)        ## Append the picked invalid tensor and sampled colors tensor to the respective lists.
                sampled_colors_groups.append(colors_picked)

            invalid = torch.cat(invalid_groups, dim=1)    ## Concatenate the invalid tensors and sampled colors tensors along the second dimension.
            sampled_colors = torch.cat(sampled_colors_groups, dim=1)

        if self.return_sample_depth:    ## If self.return_sample_depth is True, concatenate the sample depth to the sampled colors tensor.
            distance = distance.view(n_, nv_, n_pts, 1)
            sampled_colors = torch.cat((sampled_colors, distance), dim=-1)  ## cat along the last elem (c.f. paper pipeline)

        return sampled_colors, invalid  ## Return the sampled colors tensor and the invalid tensor.

    def forward(self, xyz, coarse=True, viewdirs=None, far=False, only_density=False):  ##? what are "viewdirs" and "far" for?
        """
        Predict (r, g, b, sigma) at world space points xyz.
        Please call encode first!
        :param xyz (B, 3) / [nv_==4, M==8192, 3]
        B is batch of points (in rays)
        :return (B, 4) r g b sigma
        """
        '''context manager that helps to measure the execution time of the code block inside it. i.e. used to profile the execution time of the forward pass of the model during inference for performance analysis and optimization purposes. ## to analyze the performance of the code block, helping developers identify bottlenecks and optimize their code.'''
        with profiler.record_function("model_inference"):   ## create object with the name "model_inference". ## stop the timer when exiting the block
            n_, n_pts, _ = xyz.shape         ## n:=Batch_size
            nv_ = self.grid_c_imgs.shape[1] ## 4 == (stereo 2 + side fish eye cam 2)

            if self.grid_c_combine is not None:
                nv_ = len(self.grid_c_combine)

            # Sampled features all has shape: scales [n, n_pts, c + xyz_code]   ## c + xyz_code := combined dimensionality of the features and the positional encoding c.f. (paper) Fig.2
            sampled_features, invalid_features = self.sample_features(xyz, use_single_featuremap=False)  # (B, n_pts, n_v, 103), (B, n_pts, n_v)
            # sampled_features = sampled_features.reshape(n * n_pts, -1)  ## n_pts := number of points per "ray"
            ### sampled_features == torch.Size([1*batch_size, 4, 100000, 103])  ## 100,000 points in world coordinate
            # mlp_input = sampled_features.view(1, n*n_pts, self.grid_f_features[0].shape[1], -1) ### dim(mlp_input)==torch.Size([1, 100000, 4, 103])==([one batch==1 for convection, B*100000, 4, 103]) ## origin : (n, n_pts, -1) == (Batch_size, number of 3D points, 103)
            mlp_input = sampled_features  ## Transformer will receive a single sequence of B*100,000 tokens, each token being a 103-dimensional vector
            # print("__dim(mlp_intput): ", mlp_input.shape)  ## Transformer will receive a single sequence of B*100,000 tokens, each token being a 103-dimensional vector

            # Camera frustum culling stuff, currently disabled
            combine_index = None
            dim_size = None


            # Run main NeRF network
            if self.DFT_flag:   ## interchangeable into the code snippet in models_bts.py to make comparison with vanilla vs modified (e.g. tranforemr or VAE, pos_enc, mlp, layers, change)
                mlp_output = self.DFT(mlp_input.flatten(0,1), invalid_features.flatten(0,1)) ## Transformer to learn inter-view dependencies ## squeeze to unbatch to pass them to Transformer ## mlp_input.view(1, -1, 4, sampled_features.size()[-1])
                if torch.any(torch.isnan(mlp_output)):
                    print("nan_existed: ", torch.any(torch.isnan(mlp_output)))

            elif coarse or self.mlp_fine is None:
                mlp_output = self.mlp_coarse(
                    mlp_input[..., 0, :],
                    combine_inner_dims=(n_pts,),
                    combine_index=combine_index,
                    dim_size=dim_size,
                )
            else:
                mlp_output = self.mlp_fine(
                    mlp_input,
                    combine_inner_dims=(n_pts,),
                    combine_index=combine_index,
                    dim_size=dim_size,
                )

            mlp_output = mlp_output.reshape(n_, n_pts, self._d_out)  # (n_, pts, c) -> (n_, n_pts, c)

            if self.sample_color:
                sigma = mlp_output[..., :1] ## TODO: vs multiview_signma c.f. 265 nerf.py for single_view vs multi_view_sigma
                sigma = F.softplus(sigma)
                rgb, invalid_colors = self.sample_colors(xyz)  # (n, nv_, pts, 3)
            else: ## RGB colors and invalid colors are computed directly from the mlp_output tensor. i.e. w/o calling sample_colors(xyz)
                sigma = mlp_output[..., :1]
                sigma = F.relu(sigma)
                rgb = mlp_output[..., 1:4].reshape(n, 1, n_pts, 3)
                rgb = F.sigmoid(rgb)
                invalid_colors = invalid_features.unsqueeze(-2)
                nv_ = 1

            if self.empty_empty:    ## method sets the sigma values of the invalid features to 0 for invalidity.
                sigma[torch.all(invalid_features, dim=-1)] = 0  # sigma[invalid_features[..., 0]] = 0
            # TODO: Think about this!
            # Since we don't train the colors directly, lets use softplus instead of relu
            '''Combine RGB colors and invalid colors'''
            if not only_density:
                _, _, _, c_ = rgb.shape
                rgb = rgb.permute(0, 2, 1, 3).reshape(n_, n_pts, nv_ * c_)         # (n, pts, nv * 3)
                invalid_colors = invalid_colors.permute(0, 2, 1, 3).reshape(n_, n_pts, nv_)

                invalid = invalid_colors | torch.all(invalid_features, dim=-1)[..., None]  # invalid = invalid_colors | torch.all(invalid_features, dim=1).expand(-1,-1,invalid_colors.shape[-1])       # # invalid = invalid_colors | invalid_features  # Invalid features gets broadcasted to (n, n_pts, nv)
                invalid = invalid.to(rgb.dtype)
            else:   ## If only_density is True, the method only returns the volume density (sigma) without computing the RGB colors.
                rgb = torch.zeros((n_, n_pts, nv_ * 3), device=sigma.device)
                invalid = invalid_features.to(sigma.dtype)
        return rgb, invalid, sigma
        # return rgb, torch.prod(invalid, dim=-1), sigma



class BTSNet(torch.nn.Module):
    def __init__(self, conf):
        super().__init__()
        
        self.d_min = conf.get("z_near")
        self.d_max = conf.get("z_far")

        self.learn_empty = conf.get("learn_empty", True)
        self.empty_empty = conf.get("empty_empty", False)
        self.inv_z = conf.get("inv_z", True)

        self.color_interpolation = conf.get("color_interpolation", "bilinear")
        self.code_mode = conf.get("code_mode", "z")
        if self.code_mode not in ["z", "distance"]:
            raise NotImplementedError(f"Unknown mode for positional encoding: {self.code_mode}")

        self.encoder = make_backbone(conf["encoder"])

        self.code_xyz = PositionalEncoding.from_conf(conf["code"], d_in=3)

        self.flip_augmentation = conf.get("flip_augmentation", False)

        self.return_sample_depth = conf.get("return_sample_depth", False)

        self.sample_color = conf.get("sample_color", True)

        d_in = self.encoder.latent_size + self.code_xyz.d_out
        d_out = 1 if self.sample_color else 4

        self._d_in = d_in
        self._d_out = d_out

        self.mlp_coarse = make_mlp(conf["mlp_coarse"], d_in, d_out=d_out)
        self.mlp_fine = make_mlp(conf["mlp_fine"], d_in, d_out=d_out, allow_empty=True)

        if self.learn_empty:
            self.empty_feature = nn.Parameter(torch.randn((self.encoder.latent_size,), requires_grad=True))
        ## factor to multiply the output of the corresponding MLP in the forward, which helps to control the range of the output values from the MLP
        self._scale = 0

    def set_scale(self, scale):
        self._scale = scale

    def get_scale(self):
        return self._scale

    def compute_grid_transforms(self, *args, **kwargs):
        pass

    def encode(self, images, Ks, poses_c2w, ids_encoder=None, ids_render=None, images_alt=None, combine_ids=None):
        poses_w2c = torch.inverse(poses_c2w)

        if ids_encoder is None:
            images_encoder = images
            Ks_encoder = Ks
            poses_w2c_encoder = poses_w2c
            ids_encoder = list(range(len(images)))
        else:
            images_encoder = images[:, ids_encoder]
            Ks_encoder = Ks[:, ids_encoder]
            poses_w2c_encoder = poses_w2c[:, ids_encoder]

        if images_alt is not None:
            images = images_alt
        else:
            images = images * .5 + .5

        if ids_render is None:
            images_render = images
            Ks_render = Ks
            poses_w2c_render = poses_w2c
            ids_render = list(range(len(images)))
        else:
            images_render = images[:, ids_render]
            Ks_render = Ks[:, ids_render]
            poses_w2c_render = poses_w2c[:, ids_render]

        if combine_ids is not None:
            combine_ids = list(list(group) for group in combine_ids)
            get_combined = set(sum(combine_ids, []))
            for i in range(images.shape[1]):
                if i not in get_combined:
                    combine_ids.append((i,))
            remap_encoder = {v: i for i, v in enumerate(ids_encoder)}
            remap_render = {v: i for i, v in enumerate(ids_render)}
            comb_encoder = [[remap_encoder[i] for i in group if i in ids_encoder] for group in combine_ids]
            comb_render = [[remap_render[i] for i in group if i in ids_render] for group in combine_ids]
            comb_encoder = [group for group in comb_encoder if len(group) > 0]
            comb_render = [group for group in comb_render if len(group) > 0]
        else:
            comb_encoder = None
            comb_render = None

        n, nv, c, h, w = images_encoder.shape
        c_l = self.encoder.latent_size

        if self.flip_augmentation and self.training:
            do_flip = (torch.rand(1) > .5).item()
        else:
            do_flip = False

        if do_flip:
            images_encoder = torch.flip(images_encoder, dims=(-1, ))

        image_latents_ms = self.encoder(images_encoder.view(n * nv, c, h, w))

        if do_flip:
            image_latents_ms = [torch.flip(il, dims=(-1, )) for il in image_latents_ms]

        _, _, h_, w_ = image_latents_ms[0].shape
        image_latents_ms = [F.upsample(image_latents, (h_, w_)).view(n, nv, c_l, h_, w_) for image_latents in image_latents_ms]

        self.grid_f_features = image_latents_ms
        self.grid_f_Ks = Ks_encoder
        self.grid_f_poses_w2c = poses_w2c_encoder
        self.grid_f_combine = comb_encoder

        self.grid_c_imgs = images_render    ## used to handle the combination of groups of sampled colors and their corresponding invalidity masks
        self.grid_c_Ks = Ks_render
        self.grid_c_poses_w2c = poses_w2c_render
        self.grid_c_combine = comb_render

    def sample_features(self, xyz, use_single_featuremap=True): ## 2nd arg: to control whether multiple feature maps should be combined into a single feature map or not. If True, the function will average the sampled features from multiple feature maps along the view dimension (nv) before returning the result. This can be useful when you want to combine information from multiple views or feature maps into a single representation.
        n, n_pts, _ = xyz.shape ## Get the shape of the input point cloud and the feature grid (n, pts, 3)
        n, nv, c, h, w = self.grid_f_features[self._scale].shape

        # if not use_single_featuremap:   nv = self.nv

        xyz = xyz.unsqueeze(1)  # (n, 1, pts, 3)    ## Add a singleton dimension to the input point cloud to match grid_f_poses_w2c shape
        ones = torch.ones_like(xyz[..., :1])        ## Create a tensor of ones to add a fourth dimension to the point cloud for homogeneous coordinates
        xyz = torch.cat((xyz, ones), dim=-1)        ## Concatenate the tensor of ones with the point cloud to create homogeneous coordinates
        xyz_projected = ((self.grid_f_poses_w2c[:, :nv, :3, :]) @ xyz.permute(0, 1, 3, 2))  ## Apply the camera poses to the point cloud to get the projected points and calculate the distance
        distance = torch.norm(xyz_projected, dim=-2).unsqueeze(-1)
        xyz_projected = (self.grid_f_Ks[:, :nv] @ xyz_projected).permute(0, 1, 3, 2)    ## Apply the intrinsic camera parameters to the projected points to get pixel coordinates
        xy = xyz_projected[:, :, :, [0, 1]]         ## Extract the x,y coordinates and depth value from the projected points
        z = xyz_projected[:, :, :, 2:3]

        xy = xy / z.clamp_min(EPS)                  ## Normalize the x,y coordinates by the depth value and check for invalid points    => image coord -> pixel coord
        invalid = (z <= EPS) | (xy[:, :, :, :1] < -1) | (xy[:, :, :, :1] > 1) | (xy[:, :, :, 1:2] < -1) | (xy[:, :, :, 1:2] > 1)
        '''given a vector p = (x, y, z) this is the difference of normalizing either:z ||p|| = sqrt(x^2 + y^2 + z^2). So you either give the network (x, y, z_normalized) or (x, y, ||p||_normalized) as input. It is just different parameterizations of the same point.'''
        if self.code_mode == "z": ## Depending on the code mode, normalize the depth value or distance value to the [-1, 1] range and concatenate with the xy coordinates
            # Get z into [-1, 1] range  ## Normalizing the z coordinates leads to a consistent positional encoding of the viewing information. In line 172 the viewing information (xyz_projected) is given to a positional encoder before it is appended to the overall feature vector
            if self.inv_z:
                z = (1 / z.clamp_min(EPS) - 1 / self.d_max) / (1 / self.d_min - 1 / self.d_max)
            else:
                z = (z - self.d_min) / (self.d_max - self.d_min)
            z = 2 * z - 1
            xyz_projected = torch.cat((xy, z), dim=-1)      ## concatenates the normalized x, y, and z coordinates
        elif self.code_mode == "distance":
            if self.inv_z:
                distance = (1 / distance.clamp_min(EPS) - 1 / self.d_max) / (1 / self.d_min - 1 / self.d_max)
            else:
                distance = (distance - self.d_min) / (self.d_max - self.d_min)
            distance = 2 * distance - 1
            xyz_projected = torch.cat((xy, distance), dim=-1)   ## Apply the positional encoder to the concatenated xy and depth/distance coordinates (it enables the model to capture more complex spatial dependencies without a significant increase in model complexity or training data)
        xyz_code = self.code_xyz(xyz_projected.view(n * nv * n_pts, -1)).view(n, nv, n_pts, -1)

        feature_map = self.grid_f_features[self._scale][:, :nv] ## Extract the feature map corresponding to the current scale and view. i.e. extracting the features for the first nv views c.f. encoder in pipeline
        # These samples are from different scales
        if self.learn_empty:    ## "empty space" can refer to areas in a scene where there is no object, or it could also refer to areas that are not observed or are beyond the range of the sensor. This allows the model to have a distinct learned representation for "empty" space, which can be beneficial in tasks like 3D reconstruction where understanding both the objects in a scene and the empty space between them is important.
            empty_feature_expanded = self.empty_feature.view(1, 1, 1, c).expand(n, nv, n_pts, c)    ## trainable parameter, initialized with random features
        ## feature_map (2, 4, 64, 128, 128): n = 2, nv = 4 views, c = 64 channels in the feature map, and the height and width of the feature map are h = 128 and w = 128
        ## TODO!: for multiviews for F.grid_sample : xy.view(n * nv, 1, -1, 2) To debug how xy looks like in order to integrate for multiview (by looking over doc in Pytorch regarding how to sample all frames)
        sampled_features = F.grid_sample(feature_map.view(n * nv, c, h, w), xy.view(n * nv, 1, -1, 2), mode="bilinear", padding_mode="border", align_corners=False).view(n, nv, c, n_pts).permute(0, 1, 3, 2)   ## Sample features using grid sampling and interpolate them using bilinear interpolation
        ## dim(sampled_features): (n, nv, n_pts, c)
        if self.learn_empty:    ## Replace invalid features in the sampled features tensor with the corresponding features from the expanded empty feature
            sampled_features[invalid.expand(-1, -1, -1, c)] = empty_feature_expanded[invalid.expand(-1, -1, -1, c)] ## broadcasting and make it fit to feature map
        ## dim(xyz): (B,M), M:=#_pts.
        sampled_features = torch.cat((sampled_features, xyz_code), dim=-1)  ## Concatenate the sampled features and the encoded xyz coordinates, and then it will be passed to MLP
        ## dim(sampled_features): (n, nv, M, C1+C_pos_emb)

        # If there are multiple frames with predictions, reduce them.
        # TODO: Technically, this implementations should be improved if we use multiple frames.
        # The reduction should only happen after we perform the unprojection.

        '''allows the algorithm to select the best features among different views or groups of views, based on the invalid flags. It provides an additional level of flexibility for combining features from different views in a more controlled manner.'''
        if self.grid_f_combine is not None: ## => there are specific groups of frames/views that need to be combined.
            invalid_groups = []             ## features that are out of camera's frustum or are out of range of positional encoding, [-1,1]
            sampled_features_groups = []

            for group in self.grid_f_combine:
                if len(group) == 1:
                    invalid_groups.append(invalid[:, group])
                    sampled_features_groups.append(sampled_features[:, group])

                invalid_to_combine = invalid[:, group]
                features_to_combine = sampled_features[:, group]    ## the code tries to combine the features from different views within the group

                indices = torch.min(invalid_to_combine, dim=1, keepdim=True)[1] ## These indices indicate the best features among the different views, as they have the lowest(torch.min) invalid flags.
                invalid_picked = torch.gather(invalid_to_combine, dim=1, index=indices)     ## best invalid flags are also extracted
                features_picked = torch.gather(features_to_combine, dim=1, index=indices.expand(-1, -1, -1, features_to_combine.shape[-1])) ## similarly, best features are then extracted
                ## Once all groups have been processed:
                invalid_groups.append(invalid_picked)
                sampled_features_groups.append(features_picked)

            invalid = torch.cat(invalid_groups, dim=1)
            sampled_features = torch.cat(sampled_features_groups, dim=1)
        ## !
        if use_single_featuremap:   ## compute the mean of the sampled features across the view dimension and check if any of the features are invalid.
            sampled_features = sampled_features.mean(dim=1) ### (mean dim=1)== squeeze
            invalid = torch.any(invalid, dim=1) ### sampled_features are averaged into the same dim with single featuremap

        return sampled_features, invalid    ##!! The output of the function is a tuple containing the sampled features and a boolean tensor indicating the invalid features

    def sample_colors(self, xyz):   ##? where does z come from? we're working on image domain with predicted depth from density field computed?
        n, n_pts, _ = xyz.shape                     ## n := batch size, n_pts := #_points in world coord.
        n, nv, c, h, w = self.grid_c_imgs.shape     ## nv := #_views
        xyz = xyz.unsqueeze(1)                      # (n, 1, pts, 3)
        ones = torch.ones_like(xyz[..., :1])        ## create a tensor of ones with the same shape as the first two dimensions of (xyz), up to the third dimension, and add a trailing singleton dimension (shape: (n, 1)) e.g. (n, 1, pts, 1)
        xyz = torch.cat((xyz, ones), dim=-1)        ## concatenates the tensor of ones with xyz along the last dimension (i.e., the dimension representing the coordinates of each point).
        xyz_projected = ((self.grid_c_poses_w2c[:, :, :3, :]) @ xyz.permute(0, 1, 3, 2))    ## multiply the camera-to-world transformation matrices with the concatenated tensor (xyz) to get the projected coordinates of the points in the camera coordinate system (shape: (n, nv, 3, n_pts))
        distance = torch.norm(xyz_projected, dim=-2).unsqueeze(-1)  ## compute the Euclidean norm of the projected coordinates along the last dimension and add a trailing singleton dimension (shape: (n, nv, 1, n_pts))
        xyz_projected = (self.grid_c_Ks @ xyz_projected).permute(0, 1, 3, 2)    ## multiply the intrinsic camera matrices with the projected coordinates to get the pixel coordinates (shape: (n, nv, n_pts, 3) - 3rd dimension: x, y, and z coordinates in the pixel space)
        xy = xyz_projected[:, :, :, [0, 1]]         ## select only the x and y coordinates of the pixel coordinates (shape: (n, nv, n_pts, 2))
        z = xyz_projected[:, :, :, 2:3]             ## select only the z coordinate of the pixel coordinates (shape: (n, nv, n_pts, 1))

        # This scales the x-axis into the right range.
        xy = xy / z.clamp_min(EPS)
        invalid = (z <= EPS) | (xy[:, :, :, :1] < -1) | (xy[:, :, :, :1] > 1) | (xy[:, :, :, 1:2] < -1) | (xy[:, :, :, 1:2] > 1)    ## Invalid points are points outside the image or points with invalid depth. This creates a boolean tensor of shape (n, nv, 1, n_pts), where each element is True if the corresponding point is invalid.

        sampled_colors = F.grid_sample(self.grid_c_imgs.view(n * nv, c, h, w), xy.view(n * nv, 1, -1, 2), mode=self.color_interpolation, padding_mode="border", align_corners=False).view(n, nv, c, n_pts).permute(0, 1, 3, 2)  ## Sample colors from the grid using the projected world coordinates.

        assert not torch.any(torch.isnan(sampled_colors))   ## Check that there are no NaN values in the sampled colors tensor.

        if self.grid_c_combine is not None:     ## If self.grid_c_combine is not None, combine colors from multiple points in the same group.
            invalid_groups = []
            sampled_colors_groups = []

            for group in self.grid_c_combine:   ## group:=list of indices that correspond to a subset of the total set of points in the point cloud. These subsets are combined to create a single image of the entire point cloud from multiple views.
                if len(group) == 1:     ## If the group contains only one point, append the corresponding invalid tensor and sampled colors tensor to the respective lists.
                    invalid_groups.append(invalid[:, group])
                    sampled_colors_groups.append(sampled_colors[:, group])
                    continue

                invalid_to_combine = invalid[:, group]        ## Otherwise, combine colors from the group by picking the color of the first valid point in the group.
                colors_to_combine = sampled_colors[:, group]

                indices = torch.min(invalid_to_combine, dim=1, keepdim=True)[1]         ## Get the index of the first valid point in the group.
                invalid_picked = torch.gather(invalid_to_combine, dim=1, index=indices) ## Pick the invalid tensor and sampled colors tensor corresponding to the first valid point in the group.
                colors_picked = torch.gather(colors_to_combine, dim=1, index=indices.expand(-1, -1, -1, colors_to_combine.shape[-1]))

                invalid_groups.append(invalid_picked)        ## Append the picked invalid tensor and sampled colors tensor to the respective lists.
                sampled_colors_groups.append(colors_picked)

            invalid = torch.cat(invalid_groups, dim=1)    ## Concatenate the invalid tensors and sampled colors tensors along the second dimension.
            sampled_colors = torch.cat(sampled_colors_groups, dim=1)

        if self.return_sample_depth:    ## If self.return_sample_depth is True, concatenate the sample depth to the sampled colors tensor.
            distance = distance.view(n, nv, n_pts, 1)
            sampled_colors = torch.cat((sampled_colors, distance), dim=-1)  ## cat along the last elem (c.f. paper pipeline)

        return sampled_colors, invalid  ## Return the sampled colors tensor and the invalid tensor.

    def forward(self, xyz, coarse=True, viewdirs=None, far=False, only_density=False):  ##? what are "viewdirs" and "far" for?
        """
        Predict (r, g, b, sigma) at world space points xyz.
        Please call encode first!
        :param xyz (B, 3)
        B is batch of points (in rays)
        :return (B, 4) r g b sigma
        """
        '''context manager that helps to measure the execution time of the code block inside it. i.e. used to profile the execution time of the forward pass of the model during inference for performance analysis and optimization purposes. ## to analyze the performance of the code block, helping developers identify bottlenecks and optimize their code.'''
        with profiler.record_function("model_inference"):   ## create object with the name "model_inference". ## stop the timer when exiting the block
            n, n_pts, _ = xyz.shape
            nv = self.grid_c_imgs.shape[1]

            if self.grid_c_combine is not None:
                nv = len(self.grid_c_combine)

            # Sampled features all has shape: scales [n, n_pts, c + xyz_code]   ## c + xyz_code := combined dimensionality of the features and the positional encoding c.f. (paper) Fig.2
            sampled_features, invalid_features = self.sample_features(xyz, use_single_featuremap=not only_density)  # invalid features (n, n_pts, 1) if only_density is False, then use_single_featuremap is true
            sampled_features = sampled_features.reshape(n * n_pts, -1)  ## n_pts := number of points per "ray"

            mlp_input = sampled_features.view(n, n_pts, -1)

            # Camera frustum culling stuff, currently disabled
            combine_index = None
            dim_size = None

            # Run main NeRF network
            if coarse or self.mlp_fine is None:
                mlp_output = self.mlp_coarse(
                    mlp_input,
                    combine_inner_dims=(n_pts,),
                    combine_index=combine_index,
                    dim_size=dim_size,
                )
            else:
                mlp_output = self.mlp_fine(
                    mlp_input,
                    combine_inner_dims=(n_pts,),
                    combine_index=combine_index,
                    dim_size=dim_size,
                )

            mlp_output = mlp_output.reshape(n, n_pts, self._d_out)  # (n, pts, c) -> (n, n_pts, c)

            if self.sample_color:
                sigma = mlp_output[..., :1] ## TODO: vs multiview_signma c.f. 265 nerf.py for single_view vs multi_view_sigma
                sigma = F.softplus(sigma)
                rgb, invalid_colors = self.sample_colors(xyz)  # (n, nv, pts, 3)
            else: ## RGB colors and invalid colors are computed directly from the mlp_output tensor. i.e. w/o calling sample_colors(xyz)
                sigma = mlp_output[..., :1]
                sigma = F.relu(sigma)
                rgb = mlp_output[..., 1:4].reshape(n, 1, n_pts, 3)
                rgb = F.sigmoid(rgb)
                invalid_colors = invalid_features.unsqueeze(-2)
                nv = 1

            if self.empty_empty:    ## method sets the sigma values of the invalid features to 0 for invalidity.
                sigma[invalid_features[..., 0]] = 0
            # TODO: Think about this!
            # Since we don't train the colors directly, lets use softplus instead of relu
            '''Combine RGB colors and invalid colors'''
            if not only_density:
                _, _, _, c = rgb.shape
                rgb = rgb.permute(0, 2, 1, 3).reshape(n, n_pts, nv * c)         # (n, pts, nv * 3)
                invalid_colors = invalid_colors.permute(0, 2, 1, 3).reshape(n, n_pts, nv)

                invalid = invalid_colors | invalid_features                 # Invalid features gets broadcasted to (n, n_pts, nv)
                invalid = invalid.to(rgb.dtype)
            else:   ## If only_density is True, the method only returns the volume density (sigma) without computing the RGB colors.
                rgb = torch.zeros((n, n_pts, nv * 3), device=sigma.device)
                invalid = invalid_features.to(sigma.dtype)
        return rgb, invalid, sigma
