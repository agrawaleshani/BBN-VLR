from torch_geometric.datasets import ModelNet
import torch_geometric.transforms as T
import torchvision

class IMBALANCEMODELNET10():
    def __init__(self):
        self.cls_num = 10
    
    
    def get_num_classes(self):
        return self.cls_num
    
    


if __name__ == '__main__':
    pre_transform = T.NormalizeScale()
    transform = T.SamplePoints(2048)         
    trainset = IMBALANCEMODELNET10(root='ModelNet10', train=True,
                                 transform=transform, pre_transform=pre_transform)
    train_dataset = ModelNet(root="ModelNet10", name='10', train=True, transform=transform, pre_transform=pre_transform)
    val_dataset = ModelNet(root="ModelNet10", name='10', train=False, transform=transform, pre_transform=pre_transform)

    trainloader = iter(trainset)
    data, label = next(trainloader)
    import pdb; pdb.set_trace()
