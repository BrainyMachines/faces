import torch
from torchvision.datasets.folder import ImageFolder, default_loader


class ImageFolderWithFilenames(ImageFolder):
    """Extends ImageFolder to return filename."""
    
    def __init__(self, root, transform=None, target_transform=None,
                 loader=default_loader):
        super(ImageFolderWithFilenames, self).__init__(root,
                                          transform=transform,
                                          target_transform=target_transform,
                                          loader=loader
                                          )
    
    def __getitem__(self, index):
        img, targets = super(ImageFolderWithFilenames, self).__getitem__(index)
        return img, targets, self.imgs[index][0]

    def __len__(self):
        return super(ImageFolderWithFilenames, self).__len__()