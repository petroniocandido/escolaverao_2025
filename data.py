import torch
from torchvision import datasets, transforms

converte_tensor = transforms.Compose([transforms.ToTensor()])

def get_dataset(diretorio = "."):
    treino = datasets.CIFAR10(diretorio, train=True, download=True, transform = converte_tensor)
    teste = datasets.CIFAR10(diretorio, train=False, download=True, transform = converte_tensor)
    return treino, teste


def get_dataloader(batch = 256, diretorio = "."):
    treino, teste = get_dataset(diretorio)
    treino_loader = torch.utils.data.DataLoader(treino, batch_size = batch)
    teste_loader = torch.utils.data.DataLoader(teste, batch_size = batch)
    return treino_loader, teste_loader