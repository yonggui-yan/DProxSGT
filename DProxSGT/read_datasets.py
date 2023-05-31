import numpy as np
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms

def read_datasets(dataset_name, data_dir=None, device='cpu'):
    if dataset_name in ["CIFAR10", "FashionMNIST"]:
        pass
    else:
        print('New dataset, readdatasets need adjustment')
        return None, None
        

    if data_dir==None:
        data_dir = './data/' + dataset_name + '/'
        
    if dataset_name == "FashionMNIST":
        train_dataset = torchvision.datasets.FashionMNIST(args.data_dir, train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))
                   
        test_dataset = torchvision.datasets.FashionMNIST(args.data_dir, train=False, download=True,
                    transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))
        return  train_dataset, test_dataset
 
    if dataset_name == "cifar10":
    
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),
        ])

        train_dataset = torchvision.datasets.CIFAR10(root=args.data_dir, train=True, download=True, transform=transform_train)
        test_dataset  = torchvision.datasets.CIFAR10(root=args.data_dir, train=False, download=True, transform=transform_test)
        
        return train_dataset, test_dataset
     
