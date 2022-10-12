import time
import importlib
import shutil
import pickle
import argparse

import numpy as np
import cupy as cp

import npstructures as nps
import bionumpy as bnp

from cupycounter import Counter as CupyCounter


def get_arguments():
    parser = argparse.ArgumentParser(description="Benchmarking script for reading a fasta file and achieving a frequency count for kmers")
    parser.add_argument("-backend", choices=["numpy", "cupy"], required=True)
    parser.add_argument("-counter", choices=["nps", "cupy"], required=True)
    parser.add_argument("-counter_size", type=int, required=True)
    parser.add_argument("-chunk_size", type=int, required=True)
    parser.add_argument("-cupycounter_capacity", type=int, default=0)
    return parser.parse_args()


args = get_arguments()

fasta_filename = "data/fa/testreads20m.fa"
keys_filename = "data/npy/uniquekmers.npy"


def pipeline(
        fasta_filename, keys_filename, 
        xp, counter_type, counter_size, chunk_size,
        cupycounter_capacity):
    shsize = shutil.get_terminal_size().columns
    print(">> INFO")
    print(f"BACKEND_ARRAY_MODULE       : {xp.__name__}")
    print(f"COUNTER_TYPE               : {'Counter' if counter_type == nps.Counter else 'CuCounter'}")
    print(f"COUNTER_SIZE               : {counter_size}")
    print(f"CHUNK_SIZE                 : {chunk_size}")
    print(f"CUPYCOUNTER_CAPACITY       : {cupycounter_capacity}")

    keys = np.load(keys_filename)[:counter_size]
    keys = xp.asanyarray(keys)

    t = time.time()
    if counter_type == CupyCounter:
        counter = counter_type(keys, cupycounter_capacity)
    else:
        counter = counter_type(keys)
    counter_init_t = time.time() - t

    chunk_creation_t = 0
    chunk_hashing_t = 0
    chunk_counting_t = 0

    t_ = time.time()

    num_chunks = 0
    for chunk in bnp.open(fasta_filename, chunk_size=chunk_size):
        t = time.time()
        kmers = bnp.kmers.fast_hash(chunk.sequence, 31, bnp.encodings.ACTGEncoding)
        chunk_hashing_t += time.time() - t

        t = time.time()
        counter.count(kmers.ravel())
        chunk_counting_t += time.time() - t

        num_chunks+=1
        print(f"PROCESSING CHUNK: {num_chunks}", end="\r")
    print(f"PROCESSING CHUNK: {num_chunks}")

    total_t = time.time() - t_
    chunk_creation_t = total_t - (chunk_hashing_t + chunk_counting_t)

    print(">> TIMES")
    print(f"COUNTER_INIT_TIME      : {round(counter_init_t, 3)} seconds")
    print(f"CHUNK_CREATION_TIME    : {round(chunk_creation_t, 3)} seconds")
    print(f"CHUNK_HASHING_TIME     : {round(chunk_hashing_t, 3)} seconds")
    print(f"CHUNK_COUNTING_TIME    : {round(chunk_counting_t, 3)} seconds")
    print(f"TOTAL_FA2COUNTS_TIME   : {round(total_t, 3)} seconds")


if __name__ == "__main__":
    if args.backend == "cupy":
        nps.set_backend(cp)
        bnp.set_backend(cp)

    array_module = np if args.backend == "numpy" else cp
    counter_type = nps.Counter if args.counter == "nps" else CupyCounter
    
    time_data = pipeline(
            fasta_filename=fasta_filename, 
            keys_filename=keys_filename, 
            xp=array_module, 
            counter_type=counter_type, 
            counter_size=args.counter_size, 
            chunk_size=args.chunk_size,
            cupycounter_capacity=args.cupycounter_capacity)

