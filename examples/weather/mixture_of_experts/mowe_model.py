# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Tuple, Union, Optional, Literal
import torch

from physicsnemo.experimental.models.dit import DiT


class MoWE(DiT):
    """
    Warning
    -----------
    This model uses experimental DiT architecture which may have changes in the future.

    Mixture of Weather Experts (MoWE) - This is a wrapper around the DiT model

    Parameters
    -----------
    input_size (Union[int, Tuple[int, int]]):
        Height and width of the input images.
    in_channels (int):
        The number of input channels..
    n_models (int):
        The number of models (experts) used in the Mixture of Weather Experts (MoWE) architecture.
    patch_size (Union[int, Tuple[int, int]], optional):
        The size of each image patch along height and width. Defaults to (8,8).
    out_channels (Union[None, int], optional):
        The number of output channels. If None, it is `in_channels`. Defaults to None,
        which means the output will have the same number of channels as the input.
    hidden_size (int, optional):
        The dimensionality of the transformer embeddings. Defaults to 384.
    depth (int, optional):
        The number of transformer blocks. Defaults to 12.
    num_heads (int, optional):
        The number of attention heads. Defaults to 8.
    mlp_ratio (float, optional):
        The ratio of the MLP hidden dimension to the embedding dimension. Defaults to 4.0.
    attention_backbone (str, optional):
        If 'timm' uses Attention from timm. If 'transformer_engine', uses MultiheadAttention from transformer_engine. Defaults to 'transformer_engine'.
    layernorm_backbone (str, optional):
        If 'apex', uses FusedLayerNorm from apex. If 'torch', uses LayerNorm from torch.nn. Defaults to 'apex'.
    noise_dim (int, optional):
        Dimensionality of noise. If None, the model is deterministic. Defaults to None.
    embedding_type (str, optional):
        The type of positional embedding ('sin-cos' or 'learnable'). Defaults to 'sin-cos'.
    pos_embedding_dim (int, optional):
        The dimensionality of the positional embedding. Defaults to 1.
    return_probabilities (bool, optional):
        If True, the model returns probabilities for each expert and an optional bias term. If False,
        it returns the the output directly. Defaults to True.
    bias: (bool, optional),
        If True, the model returns a bias term along with the probabilities. If False, it does not return a bias term. Defaults to True.

    Forward
    -------
    x (torch.Tensor):
        (B, n_models, in_channels, H, W) tensor of spatial inputs.
    t (torch.Tensor):
        (B,) tensor of lead times.
    noise (Optional[torch.Tensor]):
        (B, ens_size, d) tensor of random noises. The ens_size is the number of
          ensemble members to be generated for the outputs based on the random noise.

    Outputs
    -------
    if return_probabilities is True:
        probabilities (torch.Tensor):
            (B, n_models, out_channels, H, W) tensor of probabilities for each expert
        bias (torch.Tensor):
            (B, out_channels, H, W) tensor of bias values. (if bias is True)
    if return_probabilities is False:
        output (torch.Tensor):
            The output tensor of shape (B, out_channels, H, W).

    Note
    -------
    If noise is provided, the output will have an extra 'ensemble' dimension after batch dim B.


    Example
    --------
    >>> model = MoWE(
    ...     input_size=(32,64),
    ...     patch_size=4,
    ...     in_channels=5,
    ...     out_channels=5,
    ...     n_models=3,
    ...     condition_dim=8,
    ...     return_probabilities=True,
    ...
    ... )
    >>> x = torch.randn(2, 5, 32, 64)             # [B, n, C, H, W]
    >>> t = torch.randint(0, 1000, (2,))          # [B]
    >>> noise = torch.randn(2, 4, 8)              # [B, ens, d]
    >>> probabilities, bias = model(x, t, noise)
    >>> probabilities.size()
    torch.Size([2, 4, 3, 5, 32, 64])              # [B, ens, n, C, H, W]
    >>> bias.size()
    torch.Size([2, 4, 5, 32, 64])                 # [B, ens, C, H, W]
    """

    def __init__(
        self,
        input_size: Union[int, Tuple[int, int]],
        in_channels: int,
        n_models: int,
        patch_size: Union[int, Tuple[int, int]] = (8, 8),
        out_channels: Optional[int] = None,
        hidden_size: int = 384,
        depth: int = 12,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        attention_backbone: Literal[
            "timm", "transformer_engine"
        ] = "transformer_engine",
        layernorm_backbone: Literal["apex", "torch"] = "torch",
        noise_dim: Optional[int] = None,
        pos_embedding_dim: int = 1,
        return_probabilities: bool = True,
        bias: bool = True,
    ):
        self.n_models = n_models
        self.in_channels = self.n_models * in_channels
        if out_channels is None:
            out_channels = in_channels
        if return_probabilities:
            if bias:
                self.out_channels = (self.n_models + 1) * out_channels
            else:
                self.out_channels = self.n_models * out_channels
        else:
            if bias:
                raise ValueError("Bias must be False if return_probabilities is False.")
            self.out_channels = out_channels

        super().__init__(
            input_size=input_size,
            in_channels=self.in_channels,
            patch_size=patch_size,
            out_channels=self.out_channels,
            hidden_size=hidden_size,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            attention_backbone=attention_backbone,
            layernorm_backbone=layernorm_backbone,
            condition_dim=noise_dim,
            pos_embedding_dim=pos_embedding_dim,
        )
        self.return_probabilities = return_probabilities
        self.bias = bias
        self.noise_dim = noise_dim

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Performs the forward pass of the MoWE model.
        """

        # Handle stacked model inputs by merging the model and channel dimensions
        b, n_models, ch, h, w = x.shape
        x = x.view(b, n_models * ch, h, w)
        if self.noise_dim:
            n_ens = noise.size(1)
            # combine batch size and n_ens dim
            x = x.unsqueeze(1).expand(b, n_ens, *x.shape[1:])
            x = x.reshape(b * n_ens, *x.shape[2:])
            t = t.unsqueeze(1).expand(b, n_ens).reshape(-1) if t is not None else None
            noise = noise.reshape(b * n_ens, self.noise_dim)
            pos_embed = self.pos_embed.repeat(b * n_ens, 1, 1, 1)
        else:
            pos_embed = self.pos_embed.repeat(b, 1, 1, 1)

        # Same as DiT forward
        x = torch.cat([x, pos_embed], dim=1)
        x_emb = self.x_embedder(x)
        x = x_emb.flatten(2).transpose(1, 2)
        t = self.t_embedder(t)
        # Handle noise
        if self.cond_embedder is not None:
            if noise is None:
                # Fallback to using only timestep embedding if noise is not provided
                c = t
            else:
                noise_embedding = self.cond_embedder(noise)
                c = t + noise_embedding
        else:
            if noise is not None:
                raise ValueError("Noise is provided but noise_dim is None.")
            c = t
        for block in self.blocks:
            x = block(x, c)
        x = self.proj_layer(x, c)
        x = x.reshape(x.shape[0], x.shape[-1], self.h_patches, self.w_patches)
        x = self.patch_recovery(x)

        # MoWE specific reshaping
        if self.noise_dim:
            x = x.view(b, n_ens, *x.shape[1:])

        if self.return_probabilities:
            # Reshape output to separate each expert probablities and apply softmax
            if self.bias:
                if self.noise_dim:
                    x = x.view(b, n_ens, n_models + 1, ch, h, w)
                    probabilities = torch.softmax(x[:, :, :-1], dim=2)
                    bias = x[:, :, -1]
                else:
                    x = x.view(b, n_models + 1, ch, h, w)
                    probabilities = torch.softmax(x[:, :-1], dim=1)
                    bias = x[:, -1]
                return probabilities, bias
            else:
                if self.noise_dim:
                    x = x.view(b, n_ens, n_models, ch, h, w)
                    probabilities = torch.softmax(x, dim=2)
                else:
                    x = x.view(b, n_models, ch, h, w)
                    probabilities = torch.softmax(x, dim=1)
                return probabilities
        return x
