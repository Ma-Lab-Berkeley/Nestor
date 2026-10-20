"""Dataset implementations."""

from __future__ import annotations

from collections.abc import Sized
from pathlib import Path
from typing import cast

import torch
from torch.utils.data import Dataset as TorchDataset

from nestor.configs.data import DatasetConfig, DclmDatasetConfig


class Dataset(TorchDataset):
    """Base dataset class.

    Attributes:
        config (DatasetConfig): Dataset configuration.
        dataset (TorchDataset | None): Underlying dataset, when wrapped.
    """

    def __init__(self, config: DatasetConfig):
        """Initialize the dataset.

        Args:
            config (DatasetConfig): Dataset configuration.
        """

        self.config = config
        self.dataset: TorchDataset | None = None

    def __len__(self) -> int:
        """Return the number of samples."""

        if self.dataset is None:
            raise RuntimeError("Dataset not initialized")
        return len(cast(Sized, self.dataset))

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get a sample by index.

        Args:
            index (int): Sample index.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: ``(inputs, targets)``.
        """

        if self.dataset is None:
            raise RuntimeError("Dataset not initialized")
        return self.dataset[index]

    @staticmethod
    def from_config(config: DatasetConfig) -> "Dataset":
        """Create a dataset from configuration.

        Args:
            config (DatasetConfig): Dataset configuration.

        Returns:
            Dataset: The constructed dataset.
        """

        if isinstance(config, DclmDatasetConfig):
            return DclmTokenDataset(config)
        raise ValueError(f"Unknown dataset type: {config.dataset_type}")


class DclmTokenDataset(Dataset):
    """Memory-mapped flat-token corpus for large-scale causal-LM pretraining.

    Memory-maps a single flat ``.bin`` of token ids (produced offline by
    ``scripts/preprocess_dclm.py`` over the DCLM-baseline ``jsonl.zst`` shards)
    and serves contiguous ``block_size``-token windows. A 10B+ token corpus
    (tens of GB on disk) therefore never enters process memory; each worker
    only pages in the windows it reads.

    Windows are non-overlapping (``num_windows = num_tokens // block_size``),
    which matches Megatron-style packing. Each item is a
    ``(block, block.clone())`` pair of token ids; the causal LM loss applies the
    next-token shift.

    Attributes:
        config (DclmDatasetConfig): Dataset configuration.
        num_tokens (int): Total tokens in the memory-mapped file.
        block_size (int): Tokens per window.
    """

    config: DclmDatasetConfig

    def __init__(self, config: DclmDatasetConfig):
        """Initialize the memory-mapped token dataset.

        Args:
            config (DclmDatasetConfig): Dataset configuration.
        """

        import json

        import numpy as np

        super().__init__(config)
        data_path = Path(config.data_path)
        if not data_path.is_file():
            raise FileNotFoundError(
                f"DCLM token file not found: {data_path}. Run "
                "scripts/preprocess_dclm.py to produce it."
            )

        # Prefer the sidecar metadata (authoritative dtype + token count) written
        # by the preprocessor; fall back to the configured dtype + file size.
        dtype_name = config.token_dtype
        meta_path = data_path.with_suffix(data_path.suffix + ".meta.json")
        meta: dict[str, object] = {}
        if meta_path.is_file():
            meta = json.loads(meta_path.read_text())
            dtype_name = str(meta.get("token_dtype", dtype_name))

        self._dtype = np.dtype(dtype_name)
        # mmap_mode="r" keeps the array on disk; workers share the page cache.
        self._tokens = np.memmap(data_path, dtype=self._dtype, mode="r")
        self.num_tokens = int(self._tokens.shape[0])

        meta_tokens = meta.get("num_tokens")
        if isinstance(meta_tokens, int) and meta_tokens != self.num_tokens:
            raise ValueError(
                f"DCLM meta token count {meta_tokens} != file size implies "
                f"{self.num_tokens}; the .bin or its .meta.json is stale."
            )

        self.block_size = config.block_size
        self._num_windows = self.num_tokens // self.block_size
        if self._num_windows == 0:
            raise ValueError(
                f"Corpus has {self.num_tokens} tokens, fewer than one block of "
                f"{self.block_size}. Preprocess more shards or reduce block_size."
            )

    def __len__(self) -> int:
        return self._num_windows

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get a contiguous token window by index.

        Args:
            index (int): Window index in ``[0, num_windows)``.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: ``(input_ids, target_ids)``, each a
                1-D ``LongTensor`` of length ``block_size``. They are identical;
                the causal LM loss applies the next-token shift.
        """

        import numpy as np

        start = index * self.block_size
        # Copy out of the read-only mmap and widen to int64 for embedding lookup.
        window = np.array(
            self._tokens[start : start + self.block_size],
            dtype=np.int64,
            copy=True,
        )
        for source, destination in self.config.token_remap.items():
            window[window == source] = destination
        block = torch.from_numpy(window)
        return block, block.clone()
