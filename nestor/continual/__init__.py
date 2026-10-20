"""Continued training of the pretrained models: long-context extension and domain adaptation.

Two protocols, both continued pretraining from a trained checkpoint with a
fresh optimizer and a restarted schedule:

* :mod:`nestor.continual.extension` -- the length ladder: every pretrained
  model extended from 4k to 8k / 16k / 32k / 64k on long DCLM documents, the
  baselines at equal tokens and Nestor at equal tokens and at equal FLOPs
  (its attention rebate spent on new or on repeated documents).
* :mod:`nestor.continual.adaptation` -- domain adaptation and forgetting: the
  32k-extended 500M Nestor, 500M MoE and 1B dense models trained on books,
  mathematics, code and German, without and with 25 % replay, the sparse
  models with frozen routers.

:mod:`nestor.continual.flops` prices the models at any context length; the
FLOP ratio of Nestor to the MoE control sets the equal-FLOP token budgets.
"""

from nestor.continual.adaptation import CORPORA, AdaptationRun
from nestor.continual.extension import LENGTHS, SETTINGS, ExtensionRun
from nestor.continual.flops import body_flops, flop_ratio

__all__ = [
    "CORPORA",
    "LENGTHS",
    "SETTINGS",
    "AdaptationRun",
    "ExtensionRun",
    "body_flops",
    "flop_ratio",
]
