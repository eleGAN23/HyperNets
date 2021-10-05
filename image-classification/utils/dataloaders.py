import os
import torch
import numpy as np
import pickle

from torchvision import datasets, transforms
import PIL
from PIL import Image
import matplotlib.pyplot as plt

# from multiprocessing import cpu_count



class Pad(object):
    def __init__(self):
        return
    
    def __call__(self, input):
        
        return(torch.nn.functional.pad(input, pad= (0,0,0,0,1,0), mode= 'constant', value= 0))
    
    
class Concat_GrayScale(object):
    def __init__(self):
        return
    def __call__(self, input):

        # gray_scaled = transforms.Grayscale(input)
        # gray_scaled = transforms.ToPILImage()(input).convert('L')
        gray_scaled = np.asarray(input.convert('L'))
        gray_scaled = gray_scaled.reshape(gray_scaled.shape[0], gray_scaled.shape[1], 1)
        
        # print("GRAY", np.asarray(gray_scaled).reshape())
        # print("INPUT", np.asarray(input).shape)
        # print(type(gray_scaled))
        # print(type(input))
        # return torch.cat([torch.FloatTensor(input), torch.FloatTensor(gray_scaled)], dim=0)
        return np.concatenate([gray_scaled, input], axis=-1)


def preprocessing(quat_data, img_size, normalize):
        ''' quat_data: if data is quaternionic
            img_size: data reshape size (squared)
            normalize: if data has to be normalized in [0,1]
            
            '''
        R = transforms.Resize(img_size)
        C = transforms.CenterCrop(img_size)
        T = transforms.ToTensor()
        lista = []
        lista.extend([R, C, T])
        
        if quat_data:
            P = Pad()
            lista.append(P)
            if normalize:
                N = transforms.Normalize((0.5, 0.5, 0.5, 0.5), (0.5, 0.5, 0.5, 0.5))
                lista.append(N)
        else:
            if normalize:
                N = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                lista.append(N)
        print('Preprocessing:\n',transforms.Compose(lista))
        return transforms.Compose(lista)



#-----------------------------------------------------------------------------------------------------------#

class ARRange(object):
    ''' Change values of the pixels of the images to match the output of the generator in [-1,1] '''
    def __init__(self, out_range, in_range=[0,255]):
        self.in_range = in_range
        self.out_range = out_range
        self.scale = (np.float32(self.out_range[1]) - np.float32(self.out_range[0])) / (np.float32(self.in_range[1]) - np.float32(self.in_range[0]))
        self.bias = (np.float32(self.out_range[0]) - np.float32(self.in_range[0]) * self.scale)
        
        # print(self.scale, self.bias)
    def __call__(self, input):

        return input * self.scale + self.bias


class To_Tensor_custom(object):
    ''' Change values of the pixels of the images to match the output of the generator in [-1,1] '''
    def __init__(self):

        pass
        
    def __call__(self, pic):
        # handle PIL Image
        # print(pic.mode)
        
        # if pic.mode == 'I':
        #     img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        # elif pic.mode == 'I;16':
        #     img = torch.from_numpy(np.array(pic, np.int16, copy=False))
        # elif pic.mode == 'F':
        #     img = torch.from_numpy(np.array(pic, np.float32, copy=False))
        # elif pic.mode == '1':
        #     img = 255 * torch.from_numpy(np.array(pic, np.uint8, copy=False))
        # else:
        img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
    
        img = img.view(pic.size[1], pic.size[0], len(pic.getbands()))
        # put it from HWC to CHW format
        img = img.permute((2, 0, 1)).contiguous()
        
        
        # print(input.shape)
        return img


# class Image_dataset():
def preprocessing2(quat_data, img_size, normalize):
        ''' quat_data: if data is quaternionic
            img_size: data reshape size (squared)
            normalize: if data has to be normalized
            
            '''
        R = transforms.Resize(img_size)
        C = transforms.CenterCrop(img_size)
        # T = transforms.ToTensor()
        # T = transforms.PILToTensor()
        T = To_Tensor_custom()
        lista = []
        lista.extend([ R, C, T])
        
        if quat_data:
            P = Pad()
            lista.append(P)
            
        if normalize:
            N = ARRange([-1,1])
            # N = transforms.Normalize((0.5, 0.5, 0.5, 0.5), (0.5, 0.5, 0.5, 0.5))
        else:
            N = ARRange([0,1])
        lista.append(N)
       
        
        return transforms.Compose(lista)    
    

def add_dim(img):
    print(img.size())
    return img.view(img.size(0), img.size(1), img.size(2), img.size(3), 1)


def preprocessing_HQ(quat_data, img_size):
        ''' quat_data: if data is quaternionic
            img_size: data reshape size (squared)
            '''
        lista = []
        T = To_Tensor_custom()
        lista.append(T)
        
        if quat_data:
            P = Pad()
            lista.append(P)
            
        N = ARRange([0,1])
        lista.append(N)
        return transforms.Compose(lista)    
    
def preprocessing_cifar(quat_data, img_size, train=True):
        ''' quat_data: if data is quaternionic
            img_size: data reshape size (squared)
            '''
        lista = []
#         C = transforms.RandomCrop(32, padding=4, padding_mode="reflect")
#         H_flip = transforms.RandomHorizontalFlip()
#         V_flip = transforms.RandomVerticalFlip()
        T = transforms.ToTensor()
        N = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        
#         if train:
#             lista.append(C)
#             lista.append(H_flip)
#             lista.append(V_flip)
        lista.append(T)
        lista.append(N)
        
        if quat_data:
            P = Pad()
            lista.append(P)
            
        return transforms.Compose(lista) 
    
def preprocessing_cifar100(quat_data, img_size, train=True):
        ''' quat_data: if data is quaternionic
            img_size: data reshape size (squared)
            '''
        lista = []
        # C = transforms.RandomCrop(32, padding=4, padding_mode="reflect")
        # H_flip = transforms.RandomHorizontalFlip()
        T = transforms.ToTensor()
        N = transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        # N_01 = ARRange(out_range=[0,1])
        # if train:
        #     lista.append(C)
        #     lista.append(H_flip)
        lista.append(T)
        lista.append(N)
        # lista.append(N_01)
        
        if quat_data:
            P = Pad()
            lista.append(P)
            
        return transforms.Compose(lista) 
        

def preprocessing_cifar_gray(quat_data, img_size, train=True):
        ''' quat_data: if data is quaternionic
            img_size: data reshape size (squared)
            '''
        lista = []
        C = transforms.RandomCrop(32, padding=4, padding_mode="reflect")
        H_flip = transforms.RandomHorizontalFlip()
        T = transforms.ToTensor()
        N = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        # N_01 = ARRange(out_range=[0,1])
        if train:
            lista.append(C)
            lista.append(H_flip)

        if quat_data:
            # P = Pad()
            # lista.append(P)
            G = Concat_GrayScale()
            lista.append(G)
            N = transforms.Normalize((0.5, 0.4914, 0.4822, 0.4465), (0.5, 0.2023, 0.1994, 0.2010))


        lista.append(T)
        lista.append(N)
        # lista.append(N_01)
                    
        return transforms.Compose(lista)    
    
#----------------------------------------------------------------------------------------------------#

''' CIFAR10 '''

# torchvision.datasets.CIFAR10(root: str, train: bool = True, transform: Union[Callable, NoneType] = None, target_transform: Union[Callable, NoneType] = None, download: bool = False) → None


def CIFAR10_dataloader(root, quat_data, img_size, batch_size, num_workers=1, eval=False, eval_percentage=0.2):
    """CIFAR10 dataloader with resized and normalized images."""
    
    name = 'CIFAR10'
    print('Dataset:', name)
    
    dataset = datasets.CIFAR10(root=root, download= True,
                            transform = preprocessing_cifar(quat_data, img_size)) 
    
    # dataset = datasets.LSUN(root= root, classes=['bedroom_train'],
    #         transform = preprocessing2(quat_data, img_size, normalize))

    

    print('Samples:', dataset.__len__())
    print()
    print('Preprocessing:\n', dataset.transform)

    # Create the dataloader

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                            shuffle=True, num_workers=num_workers)

    dataset_test = datasets.CIFAR10(root=root, download= True, train=False,
                                transform = preprocessing_cifar(quat_data=quat_data, img_size=32))
    eval_len = int(len(dataset_test) * eval_percentage)

    test, dataset_eval = torch.utils.data.random_split(dataset_test,
                                            [len(dataset_test)-eval_len, eval_len])
    eval_loader = torch.utils.data.DataLoader(dataset_eval, batch_size=batch_size,
                                            shuffle=False, num_workers=num_workers, drop_last=True)  

    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=len(dataset_test), shuffle=False)

    return train_loader, test_loader, eval_loader, name  

        # for i, (data, _) in enumerate(test_loader):
        #     # print(data.size())
        #     # print(data)
        #     # break
        #     plt.ioff()
        #     plt.axis("off")
        #     imgs = data[0].permute(1,2,0)
        #     img_path = test_root + str(i) + '.png'
        #     plt.imsave(img_path, imgs.numpy())



        # with open("C:/Users/eleon/Documents/Dottorato/Code/QRotGAN/CIFAR/data/Test_FID_cifar/cifar-10-batches-py/test_batch", 'rb') as fo:
        #     test_dict = pickle.load(fo, encoding='bytes')
        # for elem in test_dict:
        #     # print(elem.shape)
        #     print(elem)
        #     img = Image.fromarray(elem[1], 'RGB')
        #     img.save("C:/Users/eleon/Documents/Dottorato/Code/QRotGAN/CIFAR/data/Test_FID_cifar/test_cifar")

def CIFAR100_dataloader(root, quat_data, img_size, batch_size, num_workers=1, eval=False, eval_percentage=0.2):
    """CIFAR100 dataloader with resized and normalized images."""
    
    name = 'CIFAR100'
    print('Dataset:', name)
    
    dataset = datasets.CIFAR100(root=root, download= True,
                            transform = preprocessing_cifar100(quat_data, img_size)) 
    
    # dataset = datasets.LSUN(root= root, classes=['bedroom_train'],
    #         transform = preprocessing2(quat_data, img_size, normalize))

                            
    print('Samples:', dataset.__len__())
    print()
    print('Preprocessing:\n', dataset.transform)
    # Create the dataloader
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                            shuffle=True, num_workers=num_workers)


    dataset_test = datasets.CIFAR100(root=root, download= True, train=False,
                                transform = preprocessing_cifar100(quat_data=quat_data, img_size=32, train=False))

    eval_len = int(len(dataset_test) * eval_percentage)
    
    test, dataset_eval = torch.utils.data.random_split(dataset_test,
                                            [len(dataset_test)-eval_len, eval_len])

    eval_loader = torch.utils.data.DataLoader(dataset_eval, batch_size=batch_size,
                                            shuffle=False, num_workers=num_workers, drop_last=True)  

    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, eval_loader, name  

def SVHN_dataloader(root, quat_data, img_size, batch_size, num_workers=1, eval=False, eval_percentage=0.2):

    name = 'SVHN'
    print('Dataset:', name)

    dataset = datasets.SVHN(root=root, split="train",
                            download=True, transform=preprocessing_HQ(quat_data, img_size))

    print('Samples:', dataset.__len__())
    print()
    print('Preprocessing:\n', dataset.transform)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                            shuffle=True, num_workers=num_workers, drop_last=True)

    dataset_test = datasets.SVHN(root=root, split="test",
                                download=True, transform=preprocessing_HQ(quat_data, img_size))

    eval_len = int(len(dataset_test) * eval_percentage)
    
    test, dataset_eval = torch.utils.data.random_split(dataset_test,
                                            [len(dataset_test)-eval_len, eval_len])

    eval_loader = torch.utils.data.DataLoader(dataset_eval, batch_size=batch_size,
                                            shuffle=False, num_workers=0, drop_last=True)  
    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size,
                                            shuffle=False, num_workers=0, drop_last=True)
    
    return train_loader, test_loader, eval_loader, name   

def STL10_dataloader(root, quat_data, img_size, batch_size, num_workers=1, eval=False, eval_percentage=0.2):

    name = 'STL10'
    print('Dataset:', name)

    dataset = datasets.STL10(root=root, split="train",
                            download=True, transform=preprocessing_HQ(quat_data, img_size))

    print('Samples:', dataset.__len__())
    print()
    print('Preprocessing:\n', dataset.transform)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                            shuffle=True, num_workers=num_workers, drop_last=True)

    dataset_test = datasets.STL10(root=root, split="test",
                                download=True, transform=preprocessing_HQ(quat_data, img_size))

    eval_len = int(len(dataset_test) * eval_percentage)
    
    test, dataset_eval = torch.utils.data.random_split(dataset_test,
                                            [len(dataset_test)-eval_len, eval_len])

    eval_loader = torch.utils.data.DataLoader(dataset_eval, batch_size=batch_size,
                                            shuffle=False, num_workers=0, drop_last=True)  
    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size,
                                            shuffle=False, num_workers=0, drop_last=True)
    
    return train_loader, test_loader, eval_loader, name   
