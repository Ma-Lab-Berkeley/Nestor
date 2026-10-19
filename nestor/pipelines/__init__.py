"""Pipelines for training and evaluation."""

from nestor.pipelines.base import BasePipeline
from nestor.pipelines.eval import EvalPipeline
from nestor.pipelines.train import TrainPipeline

__all__ = ["BasePipeline", "EvalPipeline", "TrainPipeline"]
