"""Nestor: a nested sparse transformer, with the dense and MoE baselines it is compared to.

Modules:
    configs: Config dataclasses for models, data, optimization, and evaluation.
    backbones: The Nestor backbone (column attention, hybrid expert pools,
        depth reads) and its base abstractions.
    adapters / heads / models: Token embedding, language-modeling head, and the
        adapter-backbone-head model that ties them together.
    data: The memory-mapped token dataset.
    scaling: muP-tagged layers and the Muon + AdamW optimizer.
    train: Trainer, optimizer builder, losses, checkpointing.
    eval: Held-out evaluation suite and checkpoint loading.
    experiments: The registered pretraining runs of the paper.
"""

__version__ = "0.1.0"
