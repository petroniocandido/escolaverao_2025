"""run.py:"""
#!/usr/bin/env python
import os
import sys
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

#################################################
# OPERAÇÃO LINEAR EM LOTE VETORIZADA COM VMAP
#################################################

lote = 50000
entradas = 100
neuronios = 50

def produto(x,y):
    return x @ y

vmap_produto = torch.func.vmap(produto, in_dims=0)

def neuronio_linear(x,w,b):
    return vmap_produto(x.repeat(neuronios, 1), w) + b

vmap_neuronio = torch.func.vmap(neuronio_linear, in_dims=0)



#################################################
# ESSA É A FUNÇÃO QUE SERÁ EXECUTADA EM 
# TODOS OS NÓS DO CLUSTER (EM CADA CORE/GPU)
# rank: Identificador do processo. O rank 0 é 
#       o processo-coordenador
# size: Número de processos em execução
#################################################

def run(rank, size):  

    particao = int(lote // size)      # Calcula o tamanho da partição de X para cada processo executor

    print(f"Partição: {particao}")

    print(f"Rank {rank} - World Size: {size}")

    # Inicializa tensores vazios para receber os dados do processo coordenador
    
    X = torch.empty(particao, entradas)
    W = torch.empty(neuronios, entradas)
    B = torch.empty(neuronios)
    y = torch.empty(particao, neuronios)

    # Sincroniza todos os processos
    
    dist.barrier(async_op=False)

    ############################################################
    # Bloco executado apenas pelo processo coordenador
    ############################################################
    if rank == 0:

        # Carrega os dados origianais a serem distribuídos entre os processos executores
        Xo = torch.rand(lote, entradas)
        Wo = torch.rand(neuronios, entradas)
        Bo = torch.zeros(neuronios)

        # Divide o tensor X em partições a serem enviados aos processos executores
        lista_X = [Xo[i*particao:(i+1)*particao] for i in range(size)]

        # Distribui as partições de X entre processos executores
        # Deve haver uma função scatter para receber esses dados no bloco dos processos executores
        dist.scatter(tensor=X, scatter_list=lista_X, src=0, async_op=False)

        print(f"Rank {rank} - Scatter SEND X")

        # Espalha W entre processos executores
        # Deve haver uma função broadcast para receber esses dados no bloco dos processos executores
        dist.broadcast(Wo, src=0, async_op=False)

        print(f"Rank {rank} - Broadcast SEND W")

        # Espalha B entre processos executores
        # Deve haver uma função broadcast para receber esses dados no bloco dos processos executores
        dist.broadcast(Bo, src=0, async_op=False)

        print(f"Rank {rank} - Broadcast SEND B")

        # Prepara uma lista de tensores vazios para receber os dados dos processos executores
        lista_y = [torch.empty(particao, neuronios) for k in range(size)]

        
        # Recolhe y dos processos executores
        # Deve haver uma função gather para enviar esses dados no bloco dos processos executores
        dist.gather(y, lista_y, async_op=False)

        print(f"Rank {rank} - Gather RECV y")

    ############################################################
    # Bloco dos processos executores
    ############################################################
    else:

        # Coleta uma partição de X vinda do processo coordenador
        # Deve haver uma função scatter para enviar esses dados no bloco do processo coordenador
        dist.scatter(tensor=X, src=0, async_op=False)

        print(f"Rank {rank} - Scatter RECV X")

        # Coleta W vindo do processo coordenador
        # Deve haver uma função broadcast para enviar esses dados no bloco do processo coordenador
        dist.broadcast(W, src=0, async_op=False)

        print(f"Rank {rank} - broadcast RECV W")

        # Coleta B vindo do processo coordenador
        # Deve haver uma função broadcast para enviar esses dados no bloco do processo coordenador
        dist.broadcast(B, src=0, async_op=False)

        print(f"Rank {rank} - broadcast RECV B")

        # Executa a função linear y = W*X + B em lotes
        
        linear_vmap = lambda X,W,B: vmap_neuronio(X, W.repeat(particao,1,1), B.repeat(particao,1))

        y = linear_vmap(X, W, B)

        # Envia y vindo do processo coordenador
        # Deve haver uma função gather para receber esses dados no bloco do processo coordenador
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
