"""Compatibility wrapper for ATS/CDM/2DM toy game generation."""

from .experiments.ats_generator import SUPPORTED_KINDS, generate_tiny_game

__all__ = ["SUPPORTED_KINDS", "generate_tiny_game"]

