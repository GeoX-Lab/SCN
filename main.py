from SCN import TMN_M_H_model
from shortnets import resnet18_cbam, resnet50_cbam

"""
Hyper-parameters:
              -intergration_lr
              -short_lr
              -temperature of sigmoid-distillation
              -transfer method
                    
Datasets Information:
    UCMerced : 21 classes, 100 pics each class, image size 256x256
    RSICB256 : 35 classes, about 690 pic each class , image size 256x256
    AID      : 30 classes, 220~420 pic each class ,image size 600x600
    *SIRI-WHU : 12 classes, 200 pics each class, image size 200x200

Short Nets Information:
    Resnet : typical network for TMN-v1
    pyconv : using pyramid conv to get better multi-scale feature based on Resnet (was one of the short Nets candidate)
        *FPN : a typical way to get multi-scale feature in detection area (which is useless here)
    
    * means not Implemented yet
"""

# integration_extractor = resnet18_cbam()  # load pretrained model for integration network;
# basic Hyper Param
batch_size = 64
epochs = 100

# TMN Hyper Param
integration_lr = 0.01
short_lr = 0.008
temp = 10
method = 'gated'

# Experimental Param
parser = 9
dataset = 'RSICB256'
short_net = 'res2net'  # 'pyconv'  # 'pyconv'  # resnet18,fpn,dcd,pyconv,eca
img_size = 256

# Outcome Param
paint_confusion_matrix = False

TMN = TMN_M_H_model(method, parser, resnet50_cbam(), batch_size, epochs, integration_lr, short_lr, temp,
                    dataset=dataset, short_net=short_net, img_size=img_size)

for i in range(parser):
    print("**************learning task %d " % (i + 1))
    if i == 0:
        TMN.beforeTrain()
        accuracy_integration = TMN.train_integration()
        TMN.afterTrain(paint_cf=paint_confusion_matrix)
    else:
        TMN.beforeTrain()
        TMN.train_short()
        TMN.train_integration()
        TMN.afterTrain(paint_cf=paint_confusion_matrix)
