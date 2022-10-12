import time
import argparse

import numpy as np
import cupy as cp

import npstructures as nps
import bionumpy as bnp

from cupycounter import Counter as CupyCounter 

def get_arguments():
    parser = argparse.ArgumentParser("Script checking that counts computed by cucounter.Counter and npstructures.Counter are equal.")
    parser.add_argument("-backend", choices=["numpy", "cupy"], required=True)
    parser.add_argument("-counter_size", type=int, required=True)
    parser.add_argument("-chunk_size", type=int, required=True)
    parser.add_argument("-cupycounter_capacity", type=int, default=0)
    return parser.parse_args()


fasta_filename = "data/fa/testreads20m.fa"
keys_filename = "data/npy/uniquekmers.npy"
args = get_arguments()

def check(fasta_filename, keys_filename, xp, counter_size, chunk_size, cupycounter_capacity):
    keys = np.load(keys_filename)[:counter_size]
    keys = xp.asanyarray(keys)

    nps_counter = nps.Counter(keys=keys)
    cupy_counter = CupyCounter(keys=keys, capacity=cupycounter_capacity)

    c = 0
    for chunk in bnp.open(fasta_filename, chunk_size=chunk_size):
        kmers = bnp.kmers.fast_hash(chunk.sequence, 31, bnp.encodings.ACTGEncoding)

        nps_counter.count(kmers.ravel())
        cupy_counter.count(kmers.ravel())

        c += 1
        print(f"Counting kmer chunks ... {c}\r", end="")
    print(f"Counting kmer chunks ... {c}")

    nps_counts = nps_counter[keys.ravel()]
    cupy_counts = cupy_counter[keys.ravel()]

    assert isinstance(nps_counts, xp.ndarray)
    assert isinstance(cupy_counts, xp.ndarray)
    xp.testing.assert_array_equal(nps_counts, cupy_counts)
    print("Assert passed")
    print(cupy_counts.size)
    print(nps_counts.size)
    print(cupy_counts[:100])

if __name__ == "__main__":
    if args.backend == "cupy":
        nps.set_backend(cp)
        bnp.set_backend(cp)

    array_module = np if args.backend == "numpy" else cp

    print(f"Backend array module : {array_module.__name__}")
    print(f"Counter size         : {args.counter_size}")
    print(f"Chunk size           : {args.chunk_size}")
    print(f"cupycounter capacity : {args.cupycounter_capacity}")

    check(fasta_filename, keys_filename, array_module, args.counter_size, args.chunk_size, args.cupycounter_capacity)
