#!/usr/bin/env python3
"""
Simple test program demonstrating ExecutorTorch multi-method export with shared state.

This example shows:
1. A model with mutable state (a cache buffer)
2. Multiple methods that read/write the shared state
3. Native ExecutorTorch multi-method export with share_mutable_buffers=True
4. Runtime execution showing state persists across method calls
"""

import torch
from executorch.exir import to_edge, ExecutorchBackendConfig
from executorch.exir.passes import MemoryPlanningPass
from executorch.exir.passes.init_mutable_pass import InitializedMutableBufferPass
from torch.export import export


class SimpleStatefulModel(torch.nn.Module):
    """
    A simple model with a mutable cache buffer shared across methods.

    Methods:
    - forward: Adds input to cache and returns the result
    - set_cache: Sets the cache to a new value
    - get_cache: Returns the current cache value
    """

    def __init__(self, cache_size: int = 4):
        super().__init__()
        # Register a mutable buffer that will be shared across methods
        self.register_buffer("cache", torch.zeros(cache_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add input to cache and return doubled result."""
        self.cache.add_(x)
        return self.cache * 2

    def set_cache(self, new_cache: torch.Tensor) -> torch.Tensor:
        """Set cache to a new value."""
        self.cache.copy_(new_cache)
        return self.cache.clone()

    def get_cache(self) -> torch.Tensor:
        """Get current cache value."""
        return self.cache.clone()

