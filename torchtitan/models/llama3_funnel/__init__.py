# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.components.loss import build_cross_entropy_loss
from torchtitan.components.lr_scheduler import build_lr_schedulers
from torchtitan.components.optimizer import build_optimizers
from torchtitan.components.tokenizer import build_hf_tokenizer
from torchtitan.components.validate import build_validator
from torchtitan.datasets.hf_datasets import build_hf_dataloader
from torchtitan.protocols.train_spec import register_train_spec, TrainSpec

from .infra.parallelize import parallelize_llama
from .infra.pipeline import pipeline_llama
from .model.args import TransformerModelArgs
from .model.model import Transformer
from .model.state_dict_adapter import Llama3StateDictAdapter

# # Copyright (c) Meta Platforms, Inc. and affiliates.
# # All rights reserved.
# #
# # This source code is licensed under the BSD-style license found in the
# # LICENSE file in the root directory of this source tree.
# #
# # Copyright (c) Meta Platforms, Inc. All Rights Reserved.

# from torchtitan.components.loss import build_cross_entropy_loss
# from torchtitan.components.lr_scheduler import build_lr_schedulers
# from torchtitan.components.optimizer import build_optimizers
# from torchtitan.datasets.hf_datasets import build_hf_dataloader
# from torchtitan.datasets.tokenizer.tiktoken import build_tiktoken_tokenizer
# from torchtitan.protocols.train_spec import register_train_spec, TrainSpec

# from ..llama3.infra.parallelize import parallelize_llama
# from ..llama3.infra.pipeline import pipeline_llama
# from .model.args import TransformerModelArgs
# from .model.model import Transformer

__all__ = [
    "parallelize_llama",
    "pipeline_llama",
    "TransformerModelArgs",
    "Transformer",
    "llama3_funnel_configs",
]


llama3_configs = {
    "debugmodel": TransformerModelArgs(
        dim=256, n_layers=6, n_heads=16, rope_theta=500000
    ),
    "debugmodel_flex_attn": TransformerModelArgs(
        dim=256,
        n_layers=6,
        n_heads=16,
        rope_theta=500000,
        use_flex_attn=True,
        attn_mask_type="block_causal",
    ),
    "test_100m": TransformerModelArgs(
        dim=1024,
        n_layers=4,
        n_heads=16,
        n_kv_heads=16,
        ffn_dim_multiplier=2.0, #Mutiply by 2 to get real intermediate_size
        multiple_of=1024,
        rope_theta=10000,
    ),
    "test_1b": TransformerModelArgs(
        dim=2048,
        n_layers=8,
        n_heads=32,
        n_kv_heads=32,
        ffn_dim_multiplier=2.0, #Mutiply by 2 to get real intermediate_size
        multiple_of=1024,
        rope_theta=10000,
    ),
    "test_1b_funnel_p2": TransformerModelArgs(
        dim=2048,
        n_layers=8,
        n_heads=32,
        n_kv_heads=32,
        ffn_dim_multiplier=2.0, #Mutiply by 2 to get real intermediate_size
        multiple_of=1024,
        rope_theta=10000,
        pool_layer_start=2,
        pool_layer_end=6,
        pool_size=2,
        pool_stagger=True,
    ),
    # "test_1b_funnel_p4": TransformerModelArgs(
    #     dim=2048,
    #     n_layers=8,
    #     n_heads=32,
    #     n_kv_heads=32,
    #     ffn_dim_multiplier=2.0, #Mutiply by 2 to get real intermediate_size
    #     multiple_of=1024,
    #     rope_theta=10000,
    #     pool_layer_start=2,
    #     pool_layer_end=6,
    #     pool_size=4,
    #     pool_stagger=False,
    #     pool_predict_overlap=True,
    #     pool_ffn=False,
    #     pool_expand_ffn=False,
    #     pool_local_attn=512,
    # ),
    "test_1b_funnel_p4": TransformerModelArgs(
        dim=2048,
        n_layers=8,
        n_heads=32,
        n_kv_heads=32,
        ffn_dim_multiplier=2.0, #Mutiply by 2 to get real intermediate_size
        multiple_of=1024,
        rope_theta=10000,
        pool_layer_start=2,
        pool_layer_end=6,
        pool_interleaved=None,
        pool_size=4,
        pool_stagger=False,
        pool_predict_overlap=True,
        pool_ffn=False,
        pool_expand_ffn=False,
        pool_local_attn=None,
        use_flex_attn=False,
    ),
    "test_40b": TransformerModelArgs(
        dim=5120,
        n_layers=61,
        n_heads=64,
        n_kv_heads=64,
        ffn_dim_multiplier=2.666,
        multiple_of=1024,
        rope_theta=10000,
    ),
    "8B": TransformerModelArgs(
        dim=4096,
        n_layers=32,
        n_heads=32,
        n_kv_heads=8,
        ffn_dim_multiplier=1.3,
        multiple_of=1024,
        rope_theta=500000,
    ),
    "70B": TransformerModelArgs(
        dim=8192,
        n_layers=80,
        n_heads=64,
        n_kv_heads=8,
        ffn_dim_multiplier=1.3,
        multiple_of=4096,
        rope_theta=500000,
    ),
    "405B": TransformerModelArgs(
        dim=16384,
        n_layers=126,
        n_heads=128,
        n_kv_heads=8,
        ffn_dim_multiplier=1.2,
        multiple_of=4096,
        rope_theta=500000,
    ),
}


register_train_spec(
    TrainSpec(
        name="llama3_funnel",
        model_cls=Transformer,
        model_args=llama3_configs,
        parallelize_fn=parallelize_llama,
        pipelining_fn=pipeline_llama,
        build_optimizers_fn=build_optimizers,
        build_lr_schedulers_fn=build_lr_schedulers,
        build_dataloader_fn=build_hf_dataloader,
        build_tokenizer_fn=build_hf_tokenizer,
        build_loss_fn=build_cross_entropy_loss,
        build_validator_fn=build_validator,
        state_dict_adapter=Llama3StateDictAdapter,
    )
)
