"""Dataset and dataloader configuration."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Literal


@dataclasses.dataclass(kw_only=True)
class DatasetConfig:
    """Base configuration for datasets.

    Attributes:
        dataset_type (str): Dataset type identifier.
        split (Literal["train", "val", "test"]): Dataset split to use.
    """

    dataset_type: str = "base"
    split: Literal["train", "val", "test"] = "train"


@dataclasses.dataclass(kw_only=True)
class DclmDatasetConfig(DatasetConfig):
    """Configuration for a pre-tokenized DCLM (or any flat-token) corpus.

    Points at a single flat binary file of token ids produced by
    ``scripts/preprocess_dclm.py``: the cl100k_base BPE ids of the DCLM-baseline
    shards, end-of-text separated, concatenated into one ``numpy`` array on
    disk. The dataset memory-maps the file (so a 10B+ token corpus never enters
    RAM) and serves contiguous ``block_size``-token windows for next-token
    prediction.

    Attributes:
        dataset_type (Literal["dclm"]): Dataset type identifier.
        data_path (Path): Path to the flat token ``.bin`` file. A sidecar
            ``<data_path>.meta.json`` (written by the preprocessor) supplies the
            dtype and token count; if absent they are inferred from
            ``token_dtype`` and the file size.
        token_dtype (Literal["uint16", "uint32"]): On-disk integer width. cl100k
            has ~100k tokens, so ``uint32`` is required (``uint16`` only fits a
            <=65536 vocabulary).
        block_size (int): Number of tokens per packed training window (context).
        tokenizer (str): tiktoken encoding name the ids were produced with; must
            match the text adapter so the vocabulary lines up.
        token_remap (dict[int, int]): Optional vectorized id replacements applied
            after reading a window, used to align equivalent special tokens.
    """

    dataset_type: Literal["dclm"] = "dclm"
    data_path: Path = dataclasses.field(default_factory=lambda: Path("./data/dclm/dclm.bin"))
    token_dtype: Literal["uint16", "uint32"] = "uint32"
    block_size: int = 4096
    tokenizer: str = "cl100k_base"
    token_remap: dict[int, int] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(kw_only=True)
class DataLoaderConfig:
    """Configuration for PyTorch DataLoader.

    Attributes:
        batch_size (int): Global batch size. In distributed runs, this is
            divided evenly across ranks to get the per-rank batch size.
        num_workers (int): Number of data loading workers.
        pin_memory (bool): Whether to pin memory for faster GPU transfer.
        drop_last (bool): Whether to drop the last incomplete batch.
        shuffle (bool): Whether to shuffle data (overridden by
            ``DistributedSampler`` when distributed).
        prefetch_factor (int | None): Number of batches to prefetch per worker.
        persistent_workers (bool): Keep worker processes alive between epochs.
    """

    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True
    drop_last: bool = True
    shuffle: bool = True
    prefetch_factor: int | None = 2
    persistent_workers: bool = False
