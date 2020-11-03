import numpy as np
import scipy
import scipy.spatial
from scipy.spatial import *
import scipy.io as io
from scipy.ndimage.filters import gaussian_filter
import os
import glob
from matplotlib import pyplot as plt
import h5py
import PIL.Image as Image
from matplotlib import cm as CM
    #User's modules
from dm_generator import *

class KNN_Gaussian_Kernal_DMGenerator(DensityMapGenerator):
    
    

    def generate_densitymap(self,image,pointsList):
        '''
        This code use k-nearst, will take one minute or more to generate a density-map with one thousand people.

        points: a two-dimension list of pedestrians' annotation with the order [[col,row],[col,row],...].
        image_shape: the shape of the image, same as the shape of required density-map. (row,col). Note that can not have channel.

        return:
        density: the density-map we want. Same shape as input image but only has one channel.

        example:
        points: three pedestrians with annotation:[[163,53],[175,64],[189,74]].
        image_shape: (768,1024) 768 is row and 1024 is column.
        '''
        image_shape=[image.shape[0],image.shape[1]]
        print("\t Shape of current image: ",image_shape,". Totally need generate ",len(pointsList),"gaussian kernels.")
        density_map = np.zeros(image_shape, dtype=np.float32)
        ground_truth_count = len(pointsList)
        if ground_truth_count == 0:
            return density_map

        leafsize = 2048
        # build kdtree
        tree = scipy.spatial.KDTree(pointsList.copy(), leafsize=leafsize)
        # query kdtree
        distances, locations = tree.query(pointsList, k=4)

        print ('\t generate density...')
        for i, pt in enumerate(pointsList):
            pt2d = np.zeros(image_shape, dtype=np.float32)
            if int(pt[1])<image_shape[0] and int(pt[0])<image_shape[1]:
                pt2d[int(pt[1]),int(pt[0])] = 1.
            else:
                continue
            if ground_truth_count > 1:
                sigma = (distances[i][1]+distances[i][2]+distances[i][3])*0.1
            else:
                sigma = np.average(np.array(gt.shape))/2./2. #case: 1 point
            density_map += scipy.ndimage.filters.gaussian_filter(pt2d, sigma, mode='constant')
        print ('\t done.')
        return density_map


if __name__=="__main__":
    root = 'C:\\Users\\PC\\Desktop\\PFE related\\existing works\\Zhang_Single-Image_Crowd_Counting_CVPR_2016_paper code sample\\MCNN-pytorch-master\\MCNN-pytorch-master\\ShanghaiTech'
    
    # generate the ShanghaiA's ground truth
    part_A_train = os.path.join(root,'part_A\\train_data','images')
    part_A_test = os.path.join(root,'part_A\\test_data','images')
    # part_B_train = os.path.join(root,'part_B_final/train_data','images')
    # part_B_test = os.path.join(root,'part_B_final/test_data','images')
    # path_sets = [part_A_train,part_A_test]
    
    # img_paths = []
    # for path in path_sets:
    #     for img_path in glob.glob(os.path.join(path, '*.jpg')):
    #         img_paths.append(img_path)
    
    # for img_path in img_paths:
    #     print(img_path)
    #     mat = io.loadmat(img_path.replace('.jpg','.mat').replace('images','ground-truth').replace('IMG_','GT_IMG_'))
    #     img= plt.imread(img_path)#768行*1024列
    #     k = np.zeros((img.shape[0],img.shape[1]))
    #     points = mat["image_info"][0,0][0,0][0] #1546person*2(col,row)
    #     mdGen=KNN_Gaussian_Kernal_DMGenerator()
    #     k = mdGen.generate_densitymap(img,points)
    #     # plt.imshow(k,cmap=CM.jet)
    #     # save density_map to disk
    #     np.save(img_path.replace('.jpg','.npy').replace('images','ground-truth'), k)        
    x=np.load('C:\\Users\\PC\\Desktop\\PFE related\\existing works\\Zhang_Single-Image_Crowd_Counting_CVPR_2016_paper code sample\\MCNN-pytorch-master\\MCNN-pytorch-master\\ShanghaiTech\\ShanghaiTech\\part_A\\test_data\\ground-truth\\IMG_80.npy')
    print(type(x))
    print(x.shape)