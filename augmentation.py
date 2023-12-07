import torchvision.transforms as T
from torchvision.transforms import GaussianBlur
 
from torchvision import transforms
from PIL import Image
   
imagenet_mean_std = [[0.485, 0.456, 0.406],[0.229, 0.224, 0.225]]

class Transform_single():
    def __init__(self, image_size, train, normalize=imagenet_mean_std):
        if train == True:
            self.transform = transforms.Compose([
                #T.Resize((298, 298), Image.BILINEAR),
                #T.RandomResizedCrop(image_size, scale=(0.08, 1.0), ratio=(3.0/4.0,4.0/3.0), interpolation=Image.BICUBIC),
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
                T.ToTensor(),
                T.Normalize(*normalize)
            ])
        else:
            self.transform = transforms.Compose([
                #transforms.Resize((298, 298), interpolation=Image.BICUBIC), # 224 -> 256 
                transforms.CenterCrop(image_size),
                T.ToTensor(),
                T.Normalize(*normalize)
            ])

    def __call__(self, x):
        return self.transform(x)

class SimSiamTransform():
    def __init__(self, image_size, mean_std=imagenet_mean_std):
        image_size = 224 if image_size is None else image_size # by default simsiam use image size 224
        p_blur = 0.5 if image_size > 32 else 0 # exclude cifar
        # the paper didn't specify this, feel free to change this value
        # I use the setting from simclr which is 50% chance applying the gaussian blur
        # the 32 is prepared for cifar training where they disabled gaussian blur
        self.transform = T.Compose([
            #T.Resize((298, 298), Image.BILINEAR),
            #T.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomApply([T.ColorJitter(0.4,0.4,0.4,0.1)], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.RandomApply([T.GaussianBlur(kernel_size=image_size//20*2+1, sigma=(0.1, 2.0))], p=p_blur),
            T.ToTensor(),
            T.Normalize(*mean_std)
        ])
    def __call__(self, x):
        x = self.transform(x)
        return x

class SimCLRTransform():
    def __init__(self, image_size, mean_std=imagenet_mean_std, s=1.0):
        image_size = 224 if image_size is None else image_size 
        self.transform = T.Compose([
            #T.Resize((298, 298), Image.BILINEAR),
            #T.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomApply([T.ColorJitter(0.8*s,0.8*s,0.8*s,0.2*s)], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.RandomApply([T.GaussianBlur(kernel_size=image_size//20*2+1, sigma=(0.1, 2.0))], p=0.5),
            # We blur the image 50% of the time using a Gaussian kernel. We randomly sample σ ∈ [0.1, 2.0], and the kernel size is set to be 10% of the image height/width.
            T.ToTensor(),
            T.Normalize(*mean_std)
        ])
    def __call__(self, x):
        x = self.transform(x)
        return x #, x2 

def get_aug(name='simsiam', image_size=224, train=True, train_classifier=None):
    if train==True:
        if name == 'simsiam':
            augmentation = SimSiamTransform(image_size)
        #elif name == 'byol':
        #    augmentation = BYOL_transform(image_size)
        elif name == 'simclr':
            augmentation = SimCLRTransform(image_size)
        else:
            raise NotImplementedError
    
    elif train==False:
        if train_classifier is None:
            raise Exception
        augmentation = Transform_single(image_size, train=train_classifier)
    else:
        raise Exception
    
    return augmentation




