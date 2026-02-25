"""
JAX implementation of JiT based on the official pytorch code https://github.com/LTH14/JiT/blob/main/model_jit.py
The improvements (RoPE, qk-norm, and cls tokens) are used.
"""

import jax
import jax.numpy as jnp
import jax.random as jrandom
from flax import nnx
from utils.rope_pos_embed import VisionRotaryEmbeddingFast


def _modulate(x, shift, scale):
    """
    Shift and scale for adaLN
    """
    return x * (1 + jnp.expand_dims(scale, axis=1)) + jnp.expand_dims(shift, axis=1)


#################################################################################
#                                 Improved Transformer                          #
#################################################################################


def scaled_dot_product_attention(query, key, value):
    """
    Returns softmax(QK^T / sqrt(d_k)) V, where d_k is the dimension of the queries (i.e. query.shape[-1])

    Args:
        query: shape (B, self.num_heads, N, C // self.num_heads)
        key: shape (B, self.num_heads, N, C // self.num_heads)
        value: shape (B, self.num_heads, N, C // self.num_heads)
    Return:
        shape (B, self.num_heads, N, C // self.num_heads)
    """

    scale_factor = 1 / jnp.sqrt(query.shape[-1])
    attn_weight = (query @ key.swapaxes(-2, -1)) * scale_factor
    attn_weight = jax.nn.softmax(
        attn_weight, axis=-1
    )  # shape (B, self.num_heads, N, N)
    return attn_weight @ value


class Attention(nnx.Module):
    """
    Attention using qk-norm and RoPE
    """

    def __init__(self, dim, num_heads=12, qkv_bias=True, *, rngs):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.q_norm = RMSNorm(head_dim)
        self.k_norm = RMSNorm(head_dim)
        self.qkv = nnx.Linear(
            dim,
            dim * 3,
            use_bias=qkv_bias,
            rngs=rngs,
            kernel_init=nnx.nn.initializers.xavier_uniform(),
            bias_init=nnx.nn.initializers.zeros_init(),
        )
        self.proj = nnx.Linear(
            dim,
            dim,
            use_bias=True,
            rngs=rngs,
            kernel_init=nnx.nn.initializers.xavier_uniform(),
            bias_init=nnx.nn.initializers.zeros_init(),
        )

    def __call__(self, x, rope):
        B, N, C = x.shape
        qkv = jnp.transpose(
            self.qkv(x).reshape((B, N, 3, self.num_heads, C // self.num_heads)),
            (2, 0, 3, 1, 4),
        )
        # self.qkv(x): (B, N, 3*C)
        # after reshape: (B, N, 3, self.num_heads, C // self.num_heads)
        # after permute: (3, B, self.num_heads, N, C // self.num_heads)

        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # Each has shape (B, self.num_heads, N, C // self.num_heads)

        # Apply qk-norm
        q = self.q_norm(q)
        k = self.k_norm(k)

        # RoPE
        q = rope(q)
        k = rope(k)

        x = scaled_dot_product_attention(q, k, v)
        x = x.swapaxes(1, 2).reshape((B, N, C))
        x = self.proj(x)
        return x


class RMSNorm(nnx.Module):
    # Implementation from Lightning-JiT: https://github.com/hustvl/LightningDiT/blob/2725fed42a14898744433809949834e26957bcdd/models/rmsnorm.py
    def __init__(self, dim, eps=1e-6):
        """
        Initialize the RMSNorm normalization layer.

        Args:
            dim (int): The dimension of the input tensor.
            eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.
        """

        super().__init__()
        self.eps = eps
        self.weight = nnx.Param(jnp.ones(dim))

    def __call__(self, x):
        """
        Applies RMS normalization to the input array x of hidden states.
        RMS(x) = sqrt(1/n * sum x_i^2)
        RMSNorm(x) = x / sqrt(RMS(x)^2 + eps) * g

        Args:
            x: jnp array with shape (batch_size, seq_len, hidden_size)
        """
        variance = jnp.mean(x**2, axis=-1, keepdims=True)
        x = x * jax.lax.rsqrt(variance + self.eps)
        return x * self.weight


class SwiGLUFFN(nnx.Module):
    def __init__(self, dim, hidden_dim, bias=True, *, rngs):
        """
        SwiGLU FFN for the improved Transformer

        Args:
            dim: Input and output dimension.
            hidden_dim: Base hidden dimension.
            rngs: The nnx.Rngs object for parameter initialization.
        """
        super().__init__()
        hidden_dim = int(
            hidden_dim * 2 / 3
        )  # Makes the parameter count equal to that of usual FFN
        self.w12 = nnx.Linear(
            dim,
            2 * hidden_dim,
            use_bias=bias,
            rngs=rngs,
            kernel_init=nnx.nn.initializers.xavier_uniform(),
            bias_init=nnx.nn.initializers.zeros_init(),
        )
        self.w3 = nnx.Linear(
            hidden_dim,
            dim,
            use_bias=bias,
            rngs=rngs,
            kernel_init=nnx.nn.initializers.xavier_uniform(),
            bias_init=nnx.nn.initializers.zeros_init(),
        )

    def __call__(self, x):
        x12 = self.w12(x)
        x1, x2 = jnp.split(x12, 2, axis=-1)
        hidden = jax.nn.silu(x1) * x2
        return self.w3(hidden)


#################################################################################
#       Embedding Layers for Timesteps, Class Labels, and Patches               #
#################################################################################


class TimestepEmbedder(nnx.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256, rngs=nnx.Rngs(0)):
        super().__init__()
        self.mlp = nnx.Sequential(
            nnx.Linear(
                frequency_embedding_size,
                hidden_size,
                kernel_init=nnx.nn.initializers.normal(stddev=0.02),
                bias_init=nnx.nn.initializers.zeros_init(),
                rngs=rngs,
            ),
            nnx.silu,
            nnx.Linear(
                hidden_size,
                hidden_size,
                kernel_init=nnx.nn.initializers.normal(stddev=0.02),
                bias_init=nnx.nn.initializers.zeros_init(),
                rngs=rngs,
            ),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D JAX array of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, dim) JAX array of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = jnp.exp(
            -jnp.log(max_period)
            * jnp.arange(start=0, stop=half, dtype=jnp.float32)
            / half
        )
        args = jnp.expand_dims(t, 1) * jnp.expand_dims(freqs, 0)  # (N, half)
        embedding = jnp.concatenate(
            [jnp.cos(args), jnp.sin(args)], axis=-1
        )  # (N, 2*half)
        if dim % 2:
            embedding = jnp.concatenate(
                [embedding, jnp.zeros((embedding.shape[0], 1))], axis=-1
            )
        return embedding

    def __call__(self, t):
        t_freq = self.timestep_embedding(
            t, self.frequency_embedding_size
        )  # (N, self.frequency_embedding size)
        t_emb = self.mlp(t_freq)  # (N, D), where D = hidden embedding size
        return t_emb


class LabelEmbedder(nnx.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """

    def __init__(self, num_classes, hidden_size, dropout_prob, key):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        key1, key2 = jrandom.split(key)

        self.embedding_table = nnx.Embed(
            num_embeddings=num_classes + use_cfg_embedding,
            features=hidden_size,
            rngs=nnx.Rngs(params=key1),
            embedding_init=nnx.nn.initializers.normal(stddev=0.02),
        )
        self.drop_key = key2
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels):
        """
        Drops labels to enable classifier-free guidance.
        """
        self.drop_key, subkey = jrandom.split(self.drop_key)
        drop_ids = (
            jrandom.uniform(subkey, shape=(labels.shape[0],), dtype=jnp.float32)
            < self.dropout_prob
        )
        labels = jnp.where(drop_ids, self.num_classes, labels)
        return labels

    def __call__(self, labels, train):
        use_dropout = self.dropout_prob > 0
        if train and use_dropout:
            labels = self.token_drop(labels)
        embeddings = self.embedding_table(labels)
        return embeddings


class PatchEmbedder(nnx.Module):
    """
    Linearly embed each patch in the input. Includes a bottleneck:
    two consecutive linear layers with a small hidden width (bottleneck_width)
    """

    def __init__(self, in_channels, hidden_size, patch_size, bottleneck_width, rngs):
        super().__init__()
        self.proj1 = nnx.Conv(
            in_features=in_channels,
            out_features=bottleneck_width,
            kernel_size=(patch_size, patch_size),
            strides=(patch_size, patch_size),
            padding="VALID",
            use_bias=False,
            rngs=rngs,
            kernel_init=nnx.nn.initializers.xavier_uniform(
                in_axis=[0, 1, 2], out_axis=3
            ),
            # weight matrix shape is (patch_size, patch_size, in_channels, hidden_size)
        )
        self.proj2 = nnx.Conv(
            in_features=bottleneck_width,
            out_features=hidden_size,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="VALID",
            use_bias=True,
            rngs=rngs,
            kernel_init=nnx.nn.initializers.xavier_uniform(
                in_axis=[0, 1, 2], out_axis=3
            ),
            bias_init=nnx.nn.initializers.zeros_init(),
        )
        self.hidden_size = hidden_size

    def __call__(self, x):
        """
        x: (N, H, W, C) tensor of spatial inputs (images or latent representations of images)

        N: batch size
        C: number of channels
        H, W: height, width of image
        """
        n = x.shape[0]
        x = self.proj2(self.proj1(x))
        # self.proj1(x): (N, H/patch_size, W/patch_size, bottleneck_width)
        # self.proj2(self.proj1(x)): (N, H/patch_size, W/patch_size, hidden_size)
        return jnp.reshape(x, (n, -1, self.hidden_size))


#################################################################################
#                                 Core JiT Model                                #
#################################################################################


class JiTBlock(nnx.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, *, rngs):
        super().__init__()

        self.norm1 = RMSNorm(hidden_size, eps=1e-6)
        self.attn = Attention(
            hidden_size, num_heads=num_heads, qkv_bias=True, rngs=rngs
        )
        self.norm2 = RMSNorm(hidden_size, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = SwiGLUFFN(hidden_size, mlp_hidden_dim, rngs=rngs)

        self.adaLN_modulation = nnx.Sequential(
            nnx.silu,
            nnx.Linear(
                hidden_size,
                6 * hidden_size,
                rngs=rngs,
                kernel_init=nnx.nn.initializers.zeros_init(),
                bias_init=nnx.nn.initializers.zeros_init(),
            ),
        )

    def __call__(self, x, c, feat_rope=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = jnp.split(
            self.adaLN_modulation(c), 6, axis=1
        )  # equivalent to using axis=-1 because c has shape (N, D)
        x = x + jnp.expand_dims(gate_msa, axis=1) * self.attn(
            _modulate(self.norm1(x), shift_msa, scale_msa), rope=feat_rope
        )
        x = x + jnp.expand_dims(gate_mlp, axis=1) * self.mlp(
            _modulate(self.norm2(x), shift_mlp, scale_mlp)
        )
        return x


class FinalLayer(nnx.Module):
    """
    The final layer of JiT.
    """

    def __init__(self, hidden_size, patch_size, out_channels, rngs):
        super().__init__()
        self.norm_final = RMSNorm(hidden_size, eps=1e-6)
        self.linear = nnx.Linear(
            hidden_size,
            patch_size * patch_size * out_channels,
            rngs=rngs,
            kernel_init=nnx.nn.initializers.zeros_init(),
            bias_init=nnx.nn.initializers.zeros_init(),
        )
        self.adaLN_modulation = nnx.Sequential(
            nnx.silu,
            nnx.Linear(
                hidden_size,
                2 * hidden_size,
                rngs=rngs,
                kernel_init=nnx.nn.initializers.zeros_init(),
                bias_init=nnx.nn.initializers.zeros_init(),
            ),
        )

    def __call__(self, x, c):
        shift, scale = jnp.split(self.adaLN_modulation(c), 2, axis=1)
        x = _modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class JiT(nnx.Module):
    """
    Just Image Transformer
    """

    def __init__(
        self,
        input_size=256,
        patch_size=16,
        in_channels=3,
        hidden_size=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,  # ratio of mlp hidden dim to embedding dim
        in_context_len=32,  # num of in-context cls tokens
        in_context_blocks=8,  # which block index the cls tokens are added in
        class_dropout_prob=0.1,  # probability of dropping class label for CFG
        num_classes=1000,
        bottleneck_width=128,
        key=jrandom.key(0),
        training=True,
        cfg_scale=2.9,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.training = training
        self.num_classes = num_classes
        self.cfg_scale = cfg_scale
        self.in_context_len = in_context_len
        self.in_context_blocks = in_context_blocks

        key, subkey_x = jrandom.split(key)

        # Patch embeddings
        self.x_embedder = PatchEmbedder(
            in_channels,
            hidden_size,
            patch_size,
            bottleneck_width,
            nnx.Rngs(params=subkey_x),
        )

        # Timestep and label embedders
        key, subkey_t = jrandom.split(key)
        self.t_embedder = TimestepEmbedder(hidden_size, 256, nnx.Rngs(params=subkey_t))
        key, subkey_y = jrandom.split(key)
        self.y_embedder = LabelEmbedder(
            num_classes, hidden_size, class_dropout_prob, subkey_y
        )

        # Positional embeddings: fixed sin-cos embedding
        self.pos_embed = jnp.expand_dims(
            jnp.asarray(
                get_2d_sincos_pos_embed(
                    embed_dim=hidden_size, grid_size=input_size // patch_size
                ),
                dtype=jnp.float32,
            ),
            axis=0,
        )
        # (1, num_patches, hidden_size), where num_patches = (input_size // patch_size) ** 2

        # RoPE
        half_head_dim = hidden_size // num_heads // 2
        hw_seq_len = input_size // patch_size
        self.feat_rope = VisionRotaryEmbeddingFast(
            dim=half_head_dim, pt_seq_len=hw_seq_len, num_cls_token=0
        )
        self.feat_rope_incontext = VisionRotaryEmbeddingFast(
            dim=half_head_dim, pt_seq_len=hw_seq_len, num_cls_token=self.in_context_len
        )

        # In-context cls tokens
        if self.in_context_len > 0:
            key, subkey_cls = jrandom.split(key)
            self.in_context_posemb = nnx.Param(
                0.02
                * jrandom.normal(
                    subkey_cls, shape=(1, self.in_context_len, hidden_size)
                )
            )

        # JiT blocks
        self.blocks = [None] * depth
        for i in range(depth):
            key, subkey_block = jrandom.split(key)
            self.blocks[i] = JiTBlock(
                hidden_size, num_heads, mlp_ratio, rngs=nnx.Rngs(params=subkey_block)
            )

        key, subkey_final = jrandom.split(key)
        self.final_layer = FinalLayer(
            hidden_size, patch_size, self.out_channels, nnx.Rngs(params=subkey_final)
        )

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        n = x.shape[0]
        c = self.out_channels
        p = self.patch_size
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = jnp.reshape(
            x, (n, h, w, p, p, c)
        )  # arranges the patches in a grid (h by w) with each patch having shape (p, p, c)
        x = jnp.einsum("nhwpqc->nhpwqc", x)  # reorders to (n, h, p, w, p, c)
        return jnp.reshape(x, (n, h * p, w * p, c))

    def __call__(self, x, t, y):
        """
        Forward pass of JiT.
        x: (N, H, W, C) array of images
        t: (N,) array of diffusion timesteps
        y: (N,) array of class labels
        """
        x = (
            self.x_embedder(x) + self.pos_embed
        )  # (N, T, D), where T = H * W / patch_size ** 2
        # self.x_embedder has shape (N, T, D)
        # self.pos_embed has shape (1, T, D)

        # Create conditioning embedding c by summing timestep and label embeddings:
        t = self.t_embedder(t)  # (N, D)
        y = self.y_embedder(y, self.training)  # (N, D)
        c = t + y  # (N, D)

        # in context cls token
        if self.in_context_len > 0:
            in_context_tokens = jnp.tile(
                y[:, None, :], (1, self.in_context_len, 1)
            )  # (N, L, D) where L = self.in_context_len
            in_context_tokens += (
                self.in_context_posemb
            )  # self.in_context_posemb has shape (1, L, D)

        for i, block in enumerate(self.blocks):
            if i == self.in_context_blocks and self.in_context_len > 0:
                # Prepend cls tokens
                x = jnp.concatenate([in_context_tokens, x], axis=1)  # (N, L+T, D)
            x = block(
                x,
                c,
                (
                    self.feat_rope
                    if i < self.in_context_blocks
                    else self.feat_rope_incontext
                ),
            )
        x = x[:, self.in_context_len :]  # Remove cls tokens

        x = self.final_layer(x, c)  # (N, T, patch_size ** 2 * out_channels)

        x = self.unpatchify(x)  # (N, H, W, out_channels)
        return x

    def forward_x_to_v(self, x, t, y, min_clip_val=5e-2):
        """
        Converts the model's x-prediction output into velocity.
        """
        return (self.__call__(x, t, y) - x) / jnp.clip(
            (1 - t)[:, None, None, None], min=min_clip_val
        )

    def forward_x_to_v_with_cfg(self, x, t, y, min_clip_val=5e-2):
        """
        Converts the model's x-prediction output into velocity and applies classifier-free guidance (CFG).
        """
        cond_output = self.__call__(x, t, y)

        N = y.shape[0]
        y_null = jnp.array([self.num_classes] * N)
        uncond_output = self.__call__(x, t, y_null)

        cfg_output = uncond_output + self.cfg_scale * (cond_output - uncond_output)
        return (cfg_output - x) / jnp.clip(
            (1 - t)[:, None, None, None], min=min_clip_val
        )


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = jnp.arange(grid_size, dtype=jnp.float32)
    grid_w = jnp.arange(grid_size, dtype=jnp.float32)
    grid = jnp.meshgrid(grid_w, grid_h)  # here w goes first
    grid = jnp.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = jnp.concatenate(
            [jnp.zeros([extra_tokens, embed_dim]), pos_embed], axis=0
        )
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = jnp.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = jnp.arange(embed_dim // 2, dtype=jnp.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = jnp.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = jnp.sin(out)  # (M, D/2)
    emb_cos = jnp.cos(out)  # (M, D/2)

    emb = jnp.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


#################################################################################
#                                   JiT Configs                                  #
#################################################################################


def JiT_B_16(**kwargs):
    return JiT(
        depth=12,
        hidden_size=768,
        num_heads=12,
        bottleneck_width=128,
        in_context_len=32,
        in_context_blocks=4,
        patch_size=16,
        **kwargs
    )


def JiT_L_16(**kwargs):
    return JiT(
        depth=24,
        hidden_size=1024,
        num_heads=16,
        bottleneck_width=128,
        in_context_len=32,
        in_context_blocks=8,
        patch_size=16,
        **kwargs
    )


def get_JiT_model(model_name):
    JiT_models = {"JiT-B/16": JiT_B_16, "JiT-L/16": JiT_L_16}
    return JiT_models[model_name]
