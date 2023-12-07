import torch
import torch.nn as nn
import torch.nn.functional as F
from .SetModel import SetTransformer, MaxWelling, DeepSet

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    
    def forward(self, x):
        return x

class Projector(nn.Module):
    def __init__(self, dim_in=768, dim_out=768):
        super(Projector, self).__init__()
        self.lin = nn.Linear(dim_in, dim_out)
        self.bn = nn.BatchNorm1d(dim_out)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.lin(x)))

class LinearBatchNorm(nn.Module):
    """Implements BatchNorm1d by BatchNorm2d, for SyncBN purpose"""
    def __init__(self, dim, affine=True):
        super(LinearBatchNorm, self).__init__()
        self.dim = dim
        self.bn = nn.BatchNorm2d(dim, affine=affine)

    def forward(self, x):
        x = x.view(-1, self.dim, 1, 1)
        x = self.bn(x)
        x = x.view(-1, self.dim)
        return x

class SupConResNet(nn.Module):
    """backbone + projection head"""
    def __init__(self, name='dinov2_vitb14', head='mlp', feat_dim=768):
        super(SupConResNet, self).__init__()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        #model, preprocess = clip.load(name, device)
        model = torch.hub.load('facebookresearch/dinov2', name)
        dim_in = 768 
        
        for param in model.parameters():
            param.requires_grad = False
        
        self.backbone = model
        self.finetune_layer = Projector(dim_in, dim_in)

        if head == 'linear':
            self.head = nn.Linear(dim_in, feat_dim)
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, feat_dim)
            )
        else:
            raise NotImplementedError(
                'head not supported: {}'.format(head))
    
    def encoder(self, x):
        with torch.no_grad():
            feat = self.backbone(x)
        #feat = feat.to(torch.float32)
        feat = self.finetune_layer(feat)
        return feat

    def forward(self, x):
        feat = self.encoder(x)
        feat = F.normalize(self.head(feat), dim=1)
        return feat

class SupConResNetWSI(nn.Module):
    """backbone + projection head"""
    def __init__(self, name='dinov2_vitb14', head='mlp', feat_dim=768, num_classes=10, set_model='SetTransformer'):
        super(SupConResNetWSI, self).__init__()
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = torch.hub.load('facebookresearch/dinov2', name)
        dim_in = 768 
        
        for param in model.parameters():
            param.requires_grad = False
        
        self.backbone = model
        self.finetune_layer = Projector(dim_in, dim_in)

        if head == 'linear':
            self.head = nn.Linear(dim_in, feat_dim)
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, feat_dim)
            )
        else:
            raise NotImplementedError(
                'head not supported: {}'.format(head))
        
        self.set_model_name = set_model
        if set_model == 'SetTransformer':
            self.set_model = SetTransformer(dim_input=dim_in, num_outputs=1, dim_output=dim_in)
        elif set_model == 'MaxWelling':
            self.set_model = MaxWelling(dim_in, dim_in)
        elif set_model == 'DeepSet':
            self.set_model = DeepSet(dim_in, dim_in)
        else:
            raise NotImplementedError(
                'set model not supported: {}'.format(set_model))
    
    def encoder(self, x):
        with torch.no_grad():
            inst_feat = self.backbone(x)
        #inst_feat = inst_feat.to(torch.float32)
        inst_feat = self.finetune_layer(inst_feat)
        return inst_feat


    def forward(self, x):
        batch_size = x.size(0)
        x = x.view((-1, *(x.size()[2:])))
        inst_feat = self.encoder(x)
        inst_feat = inst_feat.view((batch_size, -1, inst_feat.size()[1]))

        set_feat = self.set_model(inst_feat)
        if self.set_model_name == 'SetTransformer':
            set_feat = set_feat.view(set_feat.size(0), -1)
        feat = F.normalize(self.head(set_feat), dim=1)
        return feat

class SupCEResNet(nn.Module):
    """encoder + classifier"""
    def __init__(self, name='dinov2_vitb14', num_classes=10):
        super(SupCEResNet, self).__init__()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = torch.hub.load('facebookresearch/dinov2', name)
        dim_in = 768 
        
        for param in model.parameters():
            param.requires_grad = False
        
        self.backbone = model
        self.finetune_layer = Projector(dim_in, dim_in)

        self.fc = nn.Linear(dim_in, num_classes)

    def encoder(self, x):
        with torch.no_grad():
            x = self.backbone(x)
        x = self.finetune_layer(x)
        return x

    def forward(self, x):
        x = self.encoder(x)
        out = self.fc(x)
        return out

class SupCEResNetWSI(nn.Module):
    """encoder + classifier"""
    def __init__(self, name='dinov2_vitb14', num_classes=10, set_model='SetTransformer'):
        super(SupCEResNetWSI, self).__init__()
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = torch.hub.load('facebookresearch/dinov2', name)
        dim_in = 768
        
        for param in model.parameters():
            param.requires_grad = False
        
        self.backbone = model
        self.finetune_layer = Projector(dim_in, dim_in)

        self.set_model_name = set_model
        if set_model == 'SetTransformer':
            self.set_model = SetTransformer(dim_input=dim_in, num_outputs=1, dim_output=dim_in)  
        elif set_model == 'MaxWelling':
            self.set_model = MaxWelling(dim_in, dim_in)
        elif set_model == 'DeepSet':
            self.set_model = DeepSet(dim_in, dim_in)
        else:
            raise NotImplementedError(
                'set model not supported: {}'.format(set_model))
        self.fc = nn.Linear(dim_in, num_classes)

    def encoder(self, x):
        with torch.no_grad():
            x =  self.backbone(x)
        x =  self.finetune_layer(x)
        return x

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view((-1, *(x.size()[2:])))
        inst_feat = self.encoder(x)
        inst_feat = inst_feat.view((batch_size, -1, inst_feat.size(1)))
        set_feat = self.set_model(inst_feat)
        if self.set_model_name == 'SetTransformer':
            set_feat = set_feat.view(set_feat.size(0), -1)
        return self.fc(set_feat)


class LinearClassifier(nn.Module):
    """Linear classifier"""
    def __init__(self, name='dinov2_vitb14', num_classes=10):
        super(LinearClassifier, self).__init__()
        _, feat_dim = model_dict[name]
        self.fc = nn.Linear(feat_dim, num_classes)

    def forward(self, features):
        return self.fc(features)
