from cupyx import jit

@jit.rawkernel()
def init_kernel(table_keys, table_values, keys, size, capacity):
    tid = jit.blockIdx.x * jit.blockDim.x + jit.threadIdx.x

    if (tid < size):
        key = keys[tid]
        h = key % capacity

        exit = False
        while not exit:
            jit.atomic_cas(table_keys, h, 0xFFFFFFFFFFFFFFFF, key)

            if table_keys[h] == 0xFFFFFFFFFFFFFFFF or table_keys[h] == key:
                table_values[h] = 0
                exit = True
                
            h = (h + 1) % capacity

@jit.rawkernel()
def count_kernel(table_keys, table_values, keys, size, capacity):
    tid = jit.blockIdx.x * jit.blockDim.x + jit.threadIdx.x

    if (tid < size):
        key = keys[tid]
        h = key % capacity

        exit = False
        while not exit:
            cur = table_keys[h]

            if cur == 0xFFFFFFFFFFFFFFFF:
                exit = True
            elif cur == key:
                jit.atomic_add(table_values, h, 1)
                exit = True

            h = (h + 1) % capacity

