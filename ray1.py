import os
import tempfile

import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.models import resnet18
from torchvision.datasets import FashionMNIST
from torchvision.transforms import ToTensor, Normalize, Compose


from data import get_dataset, get_dataloader
from model import Lenet5, AlexNet, VGG16

import ray.train.torch
from ray import train
from ray.train import ScalingConfig, RunConfig
from ray.train.torch import TorchTrainer, get_device, TorchConfig

DEFAULT_PATH = "./"
#DEFAULT_PATH = "D:\\Dropbox\\Projetos\\ppgmcs\\MinicursoProgramacaoDistribuidaGPU\\"
#DEFAULT_PATH = "/mnt/c/Users/petro/Dropbox/Projetos/ppgmcs/MinicursoProgramacaoDistribuidaGPU/"
LOGS = DEFAULT_PATH + "logs/"
MODELS = DEFAULT_PATH + "models/model.pt"

ray.init(
    address='172.22.23.237:6379',
    #runtime_env={"env_vars": {"USE_LIBUV": "0"}}
    )

print(ray.cluster_resources())

# This function can either take in zero arguments or a single Dict argument which is set by defining train_loop_config
def worker_func(config):

    #####################
    # MODELO
    #####################

    # Inicialização
    model = Lenet5()

    if torch.cuda.is_available():
        model = model.to(torch.device(0))
    
    # Preparar o modelo para o Ray
    model = ray.train.torch.prepare_model(model)

    #####################
    # DADOS
    #####################

    # Carregar Dataset

    train_set,_ = get_dataset() # load your dataset

    # Configurar DataLoader

    train_loader = DataLoader(train_set, batch_size=config["batch"], shuffle=True)

    # Preparar Dataloader.
    train_loader = ray.train.torch.prepare_data_loader(train_loader)

    #####################
    # LOOP DE TREINAMENTO
    #####################
    
    criterion = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=config["lr"])

    # Loop
    for epoch in range(config["epochs"]):
        if ray.train.get_context().get_world_size() > 1:
            train_loader.sampler.set_epoch(epoch)

        model.train()

        for images, labels in train_loader:
            # This is done by `prepare_data_loader`!
            # images, labels = images.to("cuda"), labels.to("cuda")
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()

        with torch.no_grad():
            for images_val, labels_val in train_loader:
                outputs_val = model(images_val)
                loss_val = criterion(outputs_val, labels_val)
                

        #####################
        # SALVAR MÉTRICAS
        #####################

        metrics = {"loss": loss.item(), "loss_val": loss_val.item(), "epoch": epoch}
        
        checkpoint = ray.train.Checkpoint.from_directory(LOGS)

        ray.train.report(metrics,checkpoint=checkpoint)

        if ray.train.get_context().get_world_rank() == 0:
            print(metrics)

        #####################
        # SALVAR MODELO
        #####################

        torch.save(model.module.state_dict(),MODELS)

##############################
# ÁREA DE TRABALHO
##############################

# Configurar um storage_path é obrigatório para clusters
# Deve ser uma pasta compartilhada (um bucket s3 do AWS, uma pasta NFS, etc)
run_config = RunConfig(storage_path=DEFAULT_PATH, name="minicurso_ray")


##############################
# ESCALONAMENTO E RECURSOS
##############################

print("GPUs:" + str(torch.cuda.device_count()))

use_gpu = torch.cuda.is_available()

scaling_config = ScalingConfig(
    num_workers = 2, 
    use_gpu=use_gpu,
    resources_per_worker={
        "CPU": 1, 
        "GPU": 1,
        "memory" : 1e9,
        },
    trainer_resources={       # Evita que o Ray ocupe uma CPU só para gerenciamento
        "CPU": 1,
    },
    placement_strategy="SPREAD",    # Para execução distribuída
    )

##############################
# TREINAMENTO DISTRIBUÍDO
##############################


# Configurar

config = {
    "lr": 1e-4, 
    "epochs": 10,
    "batch" : 512
}

trainer = TorchTrainer(
    worker_func,                                 # Método a ser executado nos workers
    train_loop_config=config,                   # Parâmetros do método 
    scaling_config=scaling_config,              # Configurações de escalonamento e recursos
    run_config=run_config,                      # 
    torch_config=TorchConfig(backend="gloo"),  #The configuration for setting up the PyTorch Distributed backend. 
)

# Executar

result = trainer.fit()

# result.metrics    :   The metrics reported during training.
# metrics_dataframe :   Metrics as Pandas Dataframe
# result.checkpoint :   The latest checkpoint reported during training.
# result.path       :   The path where logs are stored.
# result.error      :   The exception that was raised, if training failed.

##############################
# MODELO FINAL
##############################

