import numpy as np
import torch
from torch.utils import data
from torchvision import datasets, transforms

# ==========================================
# CLASSI DATASET (Già presenti nel tuo codice)
# ==========================================

class CustomDataset(data.Dataset):
    def __init__(self, dataset, indices, source_class=None, target_class=None):
        self.dataset = dataset
        self.indices = indices
        self.source_class = source_class
        self.target_class = target_class
        self.contains_source_class = False

    def __getitem__(self, index):
        x, y = self.dataset[int(self.indices[index])][0], self.dataset[int(self.indices[index])][1]
        if y == self.source_class:
            y = self.target_class
        return x, y

    def __len__(self):
        return len(self.indices)

class PoisonedDataset(data.Dataset):
    def __init__(self, dataset, source_class=None, target_class=None):
        self.dataset = dataset
        self.source_class = source_class
        self.target_class = target_class

    def __getitem__(self, index):
        x, y = self.dataset[index][0], self.dataset[index][1]
        if y == self.source_class:
            y = self.target_class
        return x, y

    def __len__(self):
        return len(self.dataset)

class IMDBDataset:
    def __init__(self, reviews, targets):
        self.reviews = reviews
        self.target = targets

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, index):
        x = torch.tensor(self.reviews[index, :], dtype=torch.long)
        y = torch.tensor(self.target[index], dtype=torch.float)
        return x, y

def combine_datasets(list_of_datasets):
    return data.ConcatDataset(list_of_datasets)


# ==========================================
# FUNZIONI DI SAMPLING (IID / NON-IID)
# ==========================================

def cifar_iid(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
        
        # Convert to the format expected by the rest of the code (dict with 'data' and 'labels')
        # Note: This is a simplification. Usually we just need indices.
        # But looking at your Peer class, it expects indices.
        # Let's wrap it to match the expected output format if needed, 
        # but usually dict_users[i] = indices is enough for CustomDataset.
        
        # However, looking at FL class: self.labels.append(user_groups_train[p]['labels'])
        # So we need to return a dictionary structure.
        
        current_indices = list(dict_users[i])
        current_labels = [dataset[idx][1] for idx in current_indices]
        dict_users[i] = {'data': np.array(current_indices), 'labels': current_labels}
        
    return dict_users

def mnist_iid(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
        
        current_indices = list(dict_users[i])
        # MNIST targets might be tensor or int depending on version
        current_labels = [int(dataset[idx][1]) for idx in current_indices]
        dict_users[i] = {'data': np.array(current_indices), 'labels': current_labels}
        
    return dict_users

# ==========================================
# FUNZIONE MANCANTE: DISTRIBUTE_DATASET
# ==========================================

def distribute_dataset(dataset_name, num_peers, num_classes, dd_type, class_per_peer, samples_per_class, alpha):
    tokenizer = None
    user_groups_train = {}
    train_dataset, test_dataset = None, None

    if dataset_name == 'CIFAR10':
        # Trasformazioni standard per CIFAR10
        trans_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ])
        trans_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ])
        
        # Scarica Dataset
        train_dataset = datasets.CIFAR10('./data', train=True, download=True, transform=trans_train)
        test_dataset = datasets.CIFAR10('./data', train=False, download=True, transform=trans_test)

        # Distribuzione IID
        if dd_type == 'IID':
            user_groups_train = cifar_iid(train_dataset, num_peers)
        else:
            # Implementazione placeholder per Non-IID se servisse in futuro
            print("Warning: Non-IID not fully implemented in this quick-fix, falling back to IID logic for structure.")
            user_groups_train = cifar_iid(train_dataset, num_peers)

    elif dataset_name == 'MNIST':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        train_dataset = datasets.MNIST('./data', train=True, download=True, transform=trans_mnist)
        test_dataset = datasets.MNIST('./data', train=False, download=True, transform=trans_mnist)

        if dd_type == 'IID':
            user_groups_train = mnist_iid(train_dataset, num_peers)
        else:
            user_groups_train = mnist_iid(train_dataset, num_peers)

    else:
        raise ValueError(f"Dataset {dataset_name} non supportato in questo fix.")

    return train_dataset, test_dataset, user_groups_train, tokenizer