from .data  import construct_data_loaders, construct_datasets
import torch
from .adjacent_pair_dataset import AdjacentZPairDataset

def construct_data_loaders(data_config, batch_size, split='train'):
    dataset = AdjacentZPairDataset(
        root_dir=data_config['dataroot'],
        z_spacing=data_config.get('z_spacing', 1)
    )
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
