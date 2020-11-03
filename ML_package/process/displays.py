from matplotlib import pyplot as plt
from matplotlib import cm as CM
import PIL.Image as Image
import numpy as np
import h5py
from transforms import *

def display_densitymap(model,img_path,gt_path=None):
    img = simple_transform(Image.open(img_path).convert('RGB')).cuda()

    output = model(img.unsqueeze(0))
    print("Predicted Count : ",int(output.detach().cpu().sum().numpy()))
    temp = np.asarray(output.detach().cpu().reshape(output.detach().cpu().shape[2],output.detach().cpu().shape[3]))
    plt.imshow(temp,cmap = CM.jet)
    plt.show()
    if gt_path is not None:
        temp = h5py.File(gt_path, 'r')
        temp_1 = np.asarray(temp['density'])
        plt.imshow(temp_1,cmap = CM.jet)
        print("Original Count : ",int(np.sum(temp_1)) + 1)
        plt.show()


def display_comparaison(original,predicted,dmap=None):
    predicted=predicted.detach().cpu()
    print("Predicted Count : ",int(predicted.sum().numpy()))
    temp = np.asarray(predicted.reshape(predicted.shape[2],predicted.shape[3]))
    plt.imshow(temp,cmap = CM.jet)
    plt.show()
    if len(original.shape)>3:
        original=original.detach().cpu()
        print("Original Count : ",int(original.sum().numpy()))
        temp = np.asarray(original.reshape(original.shape[2],original.shape[3]))
    else:
        print("Original Count : ",int(original.sum()))
        temp = np.asarray(original)
    plt.imshow(temp,cmap = CM.jet)
    plt.show()    

