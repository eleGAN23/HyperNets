# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 11:32:43 2020

@author: Edoardo
"""


def GetModel(str_model,  quat_data , n, num_classes=10):
    'Models: resnet20, ...'
    print('Model:', str_model)
    print()
    if str_model == 'resnet20':
        from models.resnet import resnet20
        return (resnet20(channels=3, n=n))
    if str_model == 'resnet20large':
        from models.resnet import resnet20large
        return (resnet20large(channels=3, n=n))

    elif str_model == 'qresnet20':
        from models.qresnet import qresnet20
        return (qresnet20(channels=4, n=n))
    elif str_model == 'qresnet20large':
        from models.qresnet import qresnet20large
        return (qresnet20large(channels=4, n=n))

    elif str_model == 'phmresnet20':
        from models.phmresnet import phmresnet20
        if quat_data:
            return (phmresnet20(channels=4, n=n))
        else:
            return (phmresnet20(channels=3, n=n))
    elif str_model == 'phmresnet20large':
        from models.phmresnet import phmresnet20large
        if quat_data:
            return (phmresnet20large(channels=4, n=n))
        else:
            return (phmresnet20large(channels=3, n=n))

    if str_model == 'resnet56':
        from models.resnet import resnet56
        return (resnet56(channels=3, n=n))
    elif str_model == 'qresnet56':
        from models.qresnet import qresnet56
        return (qresnet56(channels=4, n=n))
    elif str_model == 'phmresnet56':
        from models.phmresnet import phmresnet56
        if quat_data:
            return (phmresnet56(channels=4, n=n))
        else:
            return (phmresnet56(channels=3, n=n))

    if str_model == 'resnet110large':
        from models.resnet import resnet110large
        return (resnet110large(channels=3, n=n))
    elif str_model == 'qresnet110large':
        from models.qresnet import qresnet110large
        return (qresnet110large(channels=4, n=n))
    elif str_model == 'phmresnet110large':
        from models.phmresnet import phmresnet110large
        if quat_data:
            return (phmresnet110large(channels=4, n=n))
        else:
            return (phmresnet110large(channels=3, n=n))

    if str_model == 'resnet18':
        from models.resnet_regular import ResNet18
        return (ResNet18())
    elif str_model == 'qresnet18':
        from models.qresnet_regular import QResNet18
        return (QResNet18(channels=4))
    elif str_model == 'phmresnet18':
        from models.phmresnet_regular import PHMResNet18
        if quat_data:
            return (PHMResNet18(channels=4, n=n))
        else:
            return (PHMResNet18(channels=3, n=n))

        
    if str_model == 'resnet50':
        from models.resnet_regular import ResNet50
        return (ResNet50())
    elif str_model == 'qresnet50':
        from models.qresnet_regular import QResNet50
        return (QResNet50(channels=4))
    elif str_model == 'phmresnet50':
        from models.phmresnet_regular import PHMResNet50
        if quat_data:
            return (PHMResNet50(channels=4, n=n))
        else:
            return (PHMResNet50(channels=3, n=n))

    if str_model == 'resnet18large':
        from models.resnet_regular import ResNet18Large
        return (ResNet18Large(num_classes))
    elif str_model == 'qresnet18large':
        from models.qresnet_regular import QResNet18Large
        return (QResNet18Large(channels=4, num_classes=num_classes))
    elif str_model == 'phmresnet18large':
        from models.phmresnet_regular import PHMResNet18Large
        if quat_data:
            return (PHMResNet18Large(channels=4, n=n, num_classes=num_classes))
        else:
            return (PHMResNet18Large(channels=3, n=n, num_classes=num_classes))


    if str_model == 'resnet50large':
        from models.resnet_regular import ResNet50Large
        return (ResNet50Large(num_classes))
    elif str_model == 'qresnet50large':
        from models.qresnet_regular import QResNet50Large
        return (QResNet50Large(channels=4, num_classes=num_classes))
    elif str_model == 'phmresnet50large':
        from models.phmresnet_regular import PHMResNet50Large
        if quat_data:
            return (PHMResNet50Large(channels=4, n=n, num_classes=num_classes))
        else:
            return (PHMResNet50Large(channels=3, n=n, num_classes=num_classes))

        
    if str_model == 'resnet152large':
        from models.resnet_regular import ResNet152Large
        return (ResNet152Large(num_classes))
    elif str_model == 'qresnet152large':
        from models.qresnet_regular import QResNet152Large
        return (QResNet152Large(channels=4, num_classes=num_classes))
    elif str_model == 'phmresnet152large':
        from models.phmresnet_regular import PHMResNet152Large
        if quat_data:
            return (PHMResNet152Large(channels=4, n=n, num_classes=num_classes))
        else:
            return (PHMResNet152Large(channels=3, n=n, num_classes=num_classes))
        
    ### VGG MODELS ###

    if str_model == 'vgg11':
        from models.vgg import vgg11_bn
        return (vgg11_bn(channels=3))
    if str_model == 'vgg11large':
        from models.vgg import vgg11large_bn
        return (vgg11large_bn(channels=3))


    if str_model == 'qvgg11':
        from models.qvgg import qvgg11_bn
        return (qvgg11_bn(channels=4))
    if str_model == 'qvgg11large':
        from models.qvgg import qvgg11large_bn
        return (qvgg11large_bn(channels=4))


    if str_model == 'phmvgg11':
        from models.phmvgg import phmvgg11_bn
        if quat_data:
            return (phmvgg11_bn(channels=4, n=n))
        else:
            return (phmvgg11_bn(channels=3, n=n))
    if str_model == 'phmvgg11large':
        from models.phmvgg import phmvgg11large_bn
        if quat_data:
            return (phmvgg11large_bn(channels=4, n=n))
        else:
            return (phmvgg11large_bn(channels=3, n=n))

    if str_model == 'vgg16':
        from models.vgg import vgg16_bn
        return (vgg16_bn(channels=3))
    if str_model == 'qvgg16':
        from models.qvgg import qvgg16_bn
        return (qvgg16_bn(channels=4))
    if str_model == 'phmvgg16':
        from models.phmvgg import phmvgg16_bn
        if quat_data:
            return (phmvgg16_bn(channels=4, n=n))
        else:
            return (phmvgg16_bn(channels=3, n=n))

    if str_model == 'vgg19large':
        from models.vgg import vgg19large_bn
        return (vgg19large_bn(channels=3, num_classes=num_classes))
    if str_model == 'qvgg19large':
        from models.qvgg import qvgg19large_bn
        return (qvgg19large_bn(channels=4, num_classes=num_classes))
    if str_model == 'phmvgg19large':
        from models.phmvgg import phmvgg19large_bn
        if quat_data:
            return (phmvgg19large_bn(channels=4, n=n, num_classes=num_classes))
        else:
            return (phmvgg19large_bn(channels=3, n=n, num_classes=num_classes))
        
    else:
        raise ValueError ('Model not implemented, check allowed models (-help) \n \
             Models: resnet20, qresnet20, phmresnet20, vgg11, ...')
        
        
