import numpy as np
import cupy as cp
from cupyx import jit

from .kernels import init_kernel, count_kernel

class Counter():
    def __init__(self, keys, capacity: int = 0, capacity_factor: int = 1.75):
        assert isinstance(keys, (np.ndarray, cp.ndarray)), "Invalid key type" 
        assert keys.dtype == np.uint64, "Keys must be of type uint64"
        keys = cp.asanyarray(keys, dtype=np.uint64)
        if len(keys.shape) > 1:
            keys = keys.reshape(-1)

        # Dynamically determine hashtable capacity if not provided
        if capacity == 0:
            capacity = int(keys.size * capacity_factor) 
        assert capacity > keys.size, "Capacity must be greater than size of keyset"

        self._capacity = capacity
        self._size = keys.size

        self._thread_block_size = 512

        self._kEmpty = 0xFFFFFFFFFFFFFFFF
        self._keys = cp.full(capacity, self._kEmpty, dtype=np.uint64)
        self._values = cp.full(capacity, 0xFFFFFFFF, dtype=np.uint32)

        _sz = keys.size
        grid_size = int(_sz / self._thread_block_size + (_sz % self._thread_block_size > 0))

        init_kernel[grid_size, self._thread_block_size](
                self._keys, self._values, keys, _sz, self._capacity)

    def count(self, keys):
        assert isinstance(keys, (np.ndarray, cp.ndarray)), "Invalid key type" 
        assert keys.dtype == np.uint64, "Keys must be of type uint64"
        keys = cp.asanyarray(keys, dtype=np.uint64)
        if len(keys.shape) > 1:
            keys = keys.reshape(-1)
        
        _sz = keys.size
        grid_size = int(_sz / self._thread_block_size + (_sz % self._thread_block_size > 0))

        count_kernel[grid_size, self._thread_block_size](
                self._keys, self._values, keys, _sz, self._capacity)

    def __repr__(self):
        s = f"Counter({self._keys[:40]}, {self._values[:40]}, size={self._size}, capacity={self._capacity})"
        return s
                    
    def __str__(self):
        return self.__repr__()

