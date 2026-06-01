"""Lazy model exports.

Important: do not import Geneformer at package import time.
The host-unaware NT baseline should be able to import only nucleotide_transformer.
"""

from __future__ import annotations

__all__ = [
    "NucleotideTransformerConfig",
    "NucleotideTransformerEncoder",
    "NucleotideTransformerWrapper",
    "GeneformerConfig",
    "GeneformerEncoder",
    "GeneformerWrapper",
    "HostUnawareConfig",
    "HostUnawarePredictor",
    "HostUnawareModel",
]


def __getattr__(name: str):
    if name in {
        "NucleotideTransformerConfig",
        "NucleotideTransformerEncoder",
        "NucleotideTransformerWrapper",
    }:
        from .nucleotide_transformer import (
            NucleotideTransformerConfig,
            NucleotideTransformerEncoder,
            NucleotideTransformerWrapper,
        )

        return {
            "NucleotideTransformerConfig": NucleotideTransformerConfig,
            "NucleotideTransformerEncoder": NucleotideTransformerEncoder,
            "NucleotideTransformerWrapper": NucleotideTransformerWrapper,
        }[name]

    if name in {
        "GeneformerConfig",
        "GeneformerEncoder",
        "GeneformerWrapper",
    }:
        from .geneformer import GeneformerConfig, GeneformerEncoder, GeneformerWrapper

        return {
            "GeneformerConfig": GeneformerConfig,
            "GeneformerEncoder": GeneformerEncoder,
            "GeneformerWrapper": GeneformerWrapper,
        }[name]

    if name in {
        "HostUnawareConfig",
        "HostUnawarePredictor",
        "HostUnawareModel",
    }:
        from .host_unaware import (
            HostUnawareConfig,
            HostUnawareModel,
            HostUnawarePredictor,
        )

        return {
            "HostUnawareConfig": HostUnawareConfig,
            "HostUnawarePredictor": HostUnawarePredictor,
            "HostUnawareModel": HostUnawareModel,
        }[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
