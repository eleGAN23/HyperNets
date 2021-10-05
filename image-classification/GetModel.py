# -*- coding: utf-8 -*-
"""
@author: xxx
"""


def GetModel(str_model,  quat_data , n, num_classes=10):
    'Models: ...'

    print('Model:', str_model)
    print()
    # if str_model == 'resnet20':
    #     from models.resnet import resnet20
    #     return (resnet20(channels=3, n=n))
    # if str_model == 'resnet20large':
    #     from models.resnet import resnet20large
    #     return (resnet20large(channels=3, n=n))

    # elif str_model == 'qresnet20':
    #     from models.qresnet import qresnet20
    #     return (qresnet20(channels=4, n=n))
    # elif str_model == 'qresnet20large':
    #     from models.qresnet import qresnet20large
    #     return (qresnet20large(channels=4, n=n))

    # elif str_model == 'phmresnet20':
    #     from models.phmresnet import phmresnet20
    #     if quat_data:
    #         return (phmresnet20(channels=4, n=n))
    #     else:
    #         return (phmresnet20(channels=3, n=n))
    # elif str_model == 'phmresnet20large':
    #     from models.phmresnet import phmresnet20large
    #     if quat_data:
    #         return (phmresnet20large(channels=4, n=n))
    #     else:
    #         return (phmresnet20large(channels=3, n=n))

    # if str_model == 'resnet56':
    #     from models.resnet import resnet56
    #     return (resnet56(channels=3, n=n))
    # elif str_model == 'qresnet56':
    #     from models.qresnet import qresnet56
    #     return (qresnet56(channels=4, n=n))
    # elif str_model == 'phmresnet56':
    #     from models.phmresnet import phmresnet56
    #     if quat_data:
    #         return (phmresnet56(channels=4, n=n))
    #     else:
    #         return (phmresnet56(channels=3, n=n))

    # if str_model == 'resnet110large':
    #     from models.resnet import resnet110large
    #     return (resnet110large(channels=3, n=n))
    # elif str_model == 'qresnet110large':
    #     from models.qresnet import qresnet110large
    #     return (qresnet110large(channels=4, n=n))
    # elif str_model == 'phmresnet110large':
    #     from models.phmresnet import phmresnet110large
    #     if quat_data:
    #         return (phmresnet110large(channels=4, n=n))
    #     else:
    #         return (phmresnet110large(channels=3, n=n))

    if str_model == 'resnet18':
        from models.real import resnet
        return (resnet.ResNet18())
    elif str_model == 'qresnet18':
        from models.quat import qresnet
        return (qresnet.QResNet18(channels=4))
    elif str_model == 'phcresnet18':
        from models.phc import phcresnet
        if quat_data:
            return (phcresnet.PHCResNet18(channels=4, n=n))
        else:
            return (phcresnet.PHCResNet18(channels=3, n=n))

        
    if str_model == 'resnet50':
        from models.real import resnet
        return (resnet.ResNet50())
    elif str_model == 'qresnet50':
        from models.quat import qresnet
        return (qresnet.QResNet50(channels=4))
    elif str_model == 'phcresnet50':
        from models.phc import phcresnet
        if quat_data:
            return (phcresnet.PHCResNet50(channels=4, n=n))
        else:
            return (phcresnet.PHCResNet50(channels=3, n=n))

    if str_model == 'resnet18large':
        from models.real import resnet
        return (resnet.ResNet18Large(num_classes))
    elif str_model == 'qresnet18large':
        from models.quat import qresnet
        return (qresnet.QResNet18Large(channels=4, num_classes=num_classes))
    elif str_model == 'phcresnet18large':
        from models.phc import phcresnet
        if quat_data:
            return (PHCResNet18Large(channels=4, n=n, num_classes=num_classes))
        else:
            return (PHCResNet18Large(channels=3, n=n, num_classes=num_classes))


    if str_model == 'resnet50large':
        from models.real import resnet
        return (resnet.ResNet50Large(num_classes))
    elif str_model == 'qresnet50large':
        from models.quat import qresnet
        return (qresnet.QResNet50Large(channels=4, num_classes=num_classes))
    elif str_model == 'phcresnet50large':
        from models.phc import phcresnet
        if quat_data:
            return (phcresnet.PHCResNet50Large(channels=4, n=n, num_classes=num_classes))
        else:
            return (phcresnet.PHCResNet50Large(channels=3, n=n, num_classes=num_classes))

        
    if str_model == 'resnet152large':
        from models.real import resnet
        return (resnet.ResNet152Large(num_classes))
    elif str_model == 'qresnet152large':
        from models.quat import qresnet
        return (qresnet.QResNet152Large(channels=4, num_classes=num_classes))
    elif str_model == 'phcresnet152large':
        from models.phc import phcresnet
        if quat_data:
            return (phcresnet.PHCResNet152Large(channels=4, n=n, num_classes=num_classes))
        else:
            return (phcresnet.PHCResNet152Large(channels=3, n=n, num_classes=num_classes))
        
    ### VGG MODELS ###

    # if str_model == 'vgg11':
    #     from models.vgg import vgg11_bn
    #     return (vgg11_bn(channels=3))
    # if str_model == 'vgg11large':
    #     from models.vgg import vgg11large_bn
    #     return (vgg11large_bn(channels=3))


    # if str_model == 'qvgg11':
    #     from models.qvgg import qvgg11_bn
    #     return (qvgg11_bn(channels=4))
    # if str_model == 'qvgg11large':
    #     from models.qvgg import qvgg11large_bn
    #     return (qvgg11large_bn(channels=4))


    # if str_model == 'phmvgg11':
    #     from models.phmvgg import phmvgg11_bn
    #     if quat_data:
    #         return (phmvgg11_bn(channels=4, n=n))
    #     else:
    #         return (phmvgg11_bn(channels=3, n=n))
    # if str_model == 'phmvgg11large':
    #     from models.phmvgg import phmvgg11large_bn
    #     if quat_data:
    #         return (phmvgg11large_bn(channels=4, n=n))
    #     else:
    #         return (phmvgg11large_bn(channels=3, n=n))

    if str_model == 'vgg16':
        from models.real import vgg
        return (vgg.vgg16_bn(channels=3))
    if str_model == 'qvgg16':
        from models.quat import qvgg
        return (qvgg.qvgg16_bn(channels=4))
    if str_model == 'phcvgg16':
        from models.phc import phcvgg
        if quat_data:
            return (phcvgg.phcvgg16_bn(channels=4, n=n))
        else:
            return (phcvgg.phcvgg16_bn(channels=3, n=n))

    if str_model == 'vgg19large':
        from models.real import vgg
        return (vgg.vgg19large_bn(channels=3, num_classes=num_classes))
    if str_model == 'qvgg19large':
        from models.quat import qvgg
        return (qvgg.qvgg19large_bn(channels=4, num_classes=num_classes))
    if str_model == 'phcvgg19large':
        from models.phc import phcvgg
        if quat_data:
            return (phcvgg.phcvgg19large_bn(channels=4, n=n, num_classes=num_classes))
        else:
            return (phcvgg.phcvgg19large_bn(channels=3, n=n, num_classes=num_classes))
        
    else:
        raise ValueError ('Model not implemented, check allowed models (-help) \n \
             Check the model you typed.')
        
        
