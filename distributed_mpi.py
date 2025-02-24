"""run.py:"""
#!/usr/bin/env python
import os
import sys
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


lote = 50000
entradas = 100
neuronios = 50

def produto(x,y):
    return x @ y

vmap_produto = torch.func.vmap(produto, in_dims=0)

def neuronio_linear(x,w,b):
    return vmap_produto(x.repeat(neuronios, 1), w) + b

vmap_neuronio = torch.func.vmap(neuronio_linear, in_dims=0)


def run(rank, size):

    particao = int(lote // size)

    print(f"Partição: {particao}")

    print(f"Rank {rank} - World Size: {size}")

    X = torch.empty(particao, entradas)
    W = torch.empty(neuronios, entradas)
    B = torch.empty(neuronios)
    y = torch.empty(particao, neuronios)

    dist.barrier(async_op=False)

    if rank == 0:

        Xo = torch.rand(lote, entradas)
        Wo = torch.rand(neuronios, entradas)
        Bo = torch.zeros(neuronios)

        lista_X = [Xo[i*particao:(i+1)*particao] for i in range(size)]

        dist.scatter(tensor=X, scatter_list=lista_X, src=0, async_op=False)

        print(f"Rank {rank} - Scatter SEND X")

        dist.broadcast(Wo, src=0, async_op=False)

        print(f"Rank {rank} - Broadcast SEND W")

        dist.broadcast(Bo, src=0, async_op=False)

        print(f"Rank {rank} - Broadcast SEND B")

        lista_y = [torch.empty(particao, neuronios) for k in range(size)]

        dist.gather(y, lista_y, async_op=False)

        print(f"Rank {rank} - Gather RECV y")

    else:
        dist.scatter(tensor=X, src=0, async_op=False)

        print(f"Rank {rank} - Scatter RECV X")

        dist.broadcast(W, src=0, async_op=False)

        print(f"Rank {rank} - broadcast RECV W")

        dist.broadcast(B, src=0, async_op=False)

        print(f"Rank {rank} - broadcast RECV B")

        linear_vmap = lambda X,W,B: vmap_neuronio(X, W.repeat(particao,1,1), B.repeat(particao,1))

        y = linear_vmap(X, W, B)

        dist.gather(y, dst=0, async_op=False)

        print(f"Rank {rank} - Gather y")

def init_process(rank, size, fn, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size, init_method="env://?use_libuv=False",)
    fn(rank, size)


if __name__ == "__main__":
    world_size = 10
    processes = []
    if "google.colab" in sys.modules:
        print("Running in Google Colab")
        mp.get_context("spawn")
    else:
        mp.set_start_method("spawn")
    for rank in range(world_size):
        p = mp.Process(target=init_process, args=(rank, world_size, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()