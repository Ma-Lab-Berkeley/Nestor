"""FSDP2 integration tests."""

from __future__ import annotations

import tempfile
from pathlib import Path

import torch

from nestor.backbones import NestorConfig
from nestor.configs.adapters import TextAdapterConfig
from nestor.configs.distributed import DistributedConfig
from nestor.heads import LanguageModelingHeadConfig
from nestor.models import Model, SequenceModelConfig
from nestor.train.checkpoint import load_checkpoint, save_checkpoint
from nestor.utils.distributed import (
    cleanup_distributed,
    init_distributed,
    is_fsdp2_model,
    wrap_model_for_distributed,
)


def _create_test_model_config() -> SequenceModelConfig:
    """Create a tiny sequence model config for FSDP2 tests."""

    return SequenceModelConfig(
        adapter_config=TextAdapterConfig(embed_dim=16, vocab_size=32, max_length=8),
        backbone_config=NestorConfig(
            input_dim=16,
            num_blocks=1,
            feature_dims=[16],
            router_num_experts_per_group=[1],
            pooling_strides=[1],
            num_groups=[1],
            block_num_layers=1,
            block_num_heads=2,
            block_head_dims=8,
            block_mlp_hidden_dims=[32],
            block_mlp_activations="gelu",
            router_num_experts_per_group_per_token=1,
            router_score_mlp_hidden_dims=[16],
            router_score_mlp_activations="gelu",
            router_noise_mlp_hidden_dims=[16],
            router_noise_mlp_activations="gelu",
        ),
        head_config=LanguageModelingHeadConfig(vocab_size=32),
    )


class FSDP2CheckpointTest:
    """Tests for FSDP2 wrapping and checkpoint round-tripping."""

    def test_fsdp2_wrap_and_checkpoint_roundtrip(self) -> None:
        """FSDP2 checkpoints should round-trip into both plain and FSDP2 models."""

        with tempfile.TemporaryDirectory() as tmpdir:
            shared_file = Path(tmpdir) / "dist_init"
            checkpoint_path = Path(tmpdir) / "model.pt"

            init_distributed(
                DistributedConfig(
                    enabled=True,
                    backend="gloo",
                    init_method=f"file://{shared_file}",
                    world_size=1,
                    rank=0,
                    local_rank=0,
                )
            )
            try:
                config = _create_test_model_config()
                inputs = torch.randint(0, 32, (2, 8))

                fsdp_model = wrap_model_for_distributed(
                    Model.from_config(config),
                    DistributedConfig(
                        enabled=True,
                        backend="gloo",
                        init_method=f"file://{shared_file}",
                        world_size=1,
                        rank=0,
                        local_rank=0,
                    ),
                    device=torch.device("cpu"),
                )
                assert is_fsdp2_model(fsdp_model)

                optimizer = torch.optim.AdamW(fsdp_model.parameters(), lr=1e-3)
                train_output = fsdp_model(inputs)
                loss = torch.nn.functional.cross_entropy(
                    train_output.pred[:, :-1].reshape(-1, 32), inputs[:, 1:].reshape(-1)
                )
                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    reference_logits = fsdp_model(inputs).pred.detach().cpu()

                save_checkpoint(
                    checkpoint_path,
                    model=fsdp_model,
                    optimizer=optimizer,
                    step=3,
                    epoch=1,
                )

                plain_model = Model.from_config(config)
                checkpoint = load_checkpoint(
                    checkpoint_path,
                    model=plain_model,
                    device=torch.device("cpu"),
                )
                with torch.no_grad():
                    plain_logits = plain_model(inputs).pred.detach().cpu()
                assert torch.allclose(plain_logits, reference_logits, atol=1e-5)
                assert checkpoint["step"] == 3
                assert checkpoint["epoch"] == 1

                reloaded_fsdp_model = wrap_model_for_distributed(
                    Model.from_config(config),
                    DistributedConfig(
                        enabled=True,
                        backend="gloo",
                        init_method=f"file://{shared_file}",
                        world_size=1,
                        rank=0,
                        local_rank=0,
                    ),
                    device=torch.device("cpu"),
                )
                reloaded_optimizer = torch.optim.AdamW(
                    reloaded_fsdp_model.parameters(),
                    lr=1e-3,
                )
                load_checkpoint(
                    checkpoint_path,
                    model=reloaded_fsdp_model,
                    optimizer=reloaded_optimizer,
                    device=torch.device("cpu"),
                )
                with torch.no_grad():
                    reloaded_logits = reloaded_fsdp_model(inputs).pred.detach().cpu()
                assert torch.allclose(reloaded_logits, reference_logits, atol=1e-5)
            finally:
                cleanup_distributed()
