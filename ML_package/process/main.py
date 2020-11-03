import os,sys,inspect,glob,io,subprocess,re,gc
import argparse
def import_or_install(package,pipname):
    try:
        __import__(package) 
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pipname])

import_or_install("matplotlib","matplotlib")
import_or_install("visdom","visdom")
import_or_install("numpy","numpy")
import_or_install("pydrive","Pydrive")
import_or_install("github","PyGithub")
import_or_install("plotly","plotly")
import_or_install("chart_studio","chart-studio")
import_or_install("piexif","piexif")
import_or_install("psutil","psutil")

import torch
from torch import nn
import matplotlib as plt
import numpy as np

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
    # User's modules from another directory
sys.path.append(os.path.join(parentdir , "bases"))
sys.path.append(os.path.join(parentdir , "models"))
sys.path.append(os.path.join(parentdir , "data_loaders"))
sys.path.append(os.path.join(parentdir , "density_map_generators"))
sys.path.append(os.path.join(parentdir , "configs"))

from datasets import *
from params import *
import utils
from dm_generator import *
from knn_gaussian_kernal import *
from loaders import *
from mcnn import *
from CSRNet import *
from SANet import *
from CCNN import *
import plots
import displays
import trainsparams
from torch.utils.data.sampler import SubsetRandomSampler



def prepare_datasets(baseRootPath,datasets_list:list,dm_generator,resetFlag=False):
    '''
        Prepars DataSets for training by generating ground-truth density map for every image of every Dataset.
    '''
    print("####### Preparing Data...")
    paths_list=[]
    for dataset_name in datasets_list:
        if 'ShanghaiTech_partA'==dataset_name:
            paths_list.append(prepare_ShanghaiTech_dataset(os.path.join(baseRootPath,'ShanghaiTech'),'A',dm_generator,resetFlag))
        elif 'ShanghaiTech_partB'==dataset_name:
            paths_list.append(prepare_ShanghaiTech_dataset(os.path.join(baseRootPath,'ShanghaiTech'),'B',dm_generator,resetFlag))   
        else:
            paths_list.append(prepare_dataset(baseRootPath,dataset_name,dm_generator,resetFlag))   
       

    
    return paths_list         
    

def prepare_ShanghaiTech_dataset(root,part,dm_generator,resetFlag=False):
    root=os.path.join(root,"ShanghaiTech")
    paths_dict=dict()
    print('\t  #Preparing Dataset : ShanghaiTech part ',part,' :')
        # generate the ShanghaiA's ground truth
    if not part=="A" and not part=="B": raise Exception("Invalide parts passed for shanghai ")

    train_path=os.path.join(root,'part_'+part,'train_data')
    test_path=os.path.join(root,'part_'+part,'test_data')

    

        # save both train and test paths
    paths_dict["images"]=os.path.join(train_path,'images')
    paths_dict["ground-truth"]=os.path.join(train_path,'ground-truth')

    path_sets = [paths_dict["images"],paths_dict["ground-truth"]]
    
    img_paths = []
        # Grab all .jpg images paths
    for path in path_sets:
        for img_path in glob.glob(os.path.join(path, '*.jpg')):
            img_paths.append(img_path)

            # Generate density map for each image
    for img_path in img_paths:
        if os.path.exists(img_path.replace('.jpg','.npy').replace('images','ground-truth')) and not resetFlag:
            #print("\t Already exists.")
            continue
        print('\t\t Generating Density map for : ',os.path.basename(img_path)," :",end=' ')

            # load matrix containing ground truth infos
        mat = io.loadmat(img_path.replace('.jpg','.mat').replace('images','ground-truth').replace('IMG_','GT_IMG_'))
        img= plt.imread(img_path)#768行*1024列
        density_map = np.zeros((img.shape[0],img.shape[1]))
        points = mat["image_info"][0,0][0,0][0] #1546person*2(col,row)

            # Generate the density map
        density_map = dm_generator.generate_densitymap(img,points)

            # save density_map on disk
        np.save(img_path.replace('.jpg','.npy').replace('images','ground-truth'), density_map)
    print('\t Done.')
    return paths_dict        

def prepare_dataset(root,dirname,dm_generator,resetFlag=False):
    root=os.path.join(root,dirname)
    paths_dict=dict()
    
    print('\t #Preparing Dataset : ',dirname)
        # save both train and test paths
    paths_dict["images"]=os.path.join(root,'images')
    paths_dict["ground-truth"]=os.path.join(root,'ground-truth')

    path_sets = [paths_dict["images"],paths_dict["ground-truth"]]
    
    img_paths = []
        # Grab all .jpg images paths
    
    for img_path in glob.glob(os.path.join(paths_dict["images"], '*.jpg')):
        img_paths.append(img_path)

            # Generate density map for each image
    for img_path in img_paths:
        if os.path.exists(img_path.replace('.jpg','.npy').replace('images','ground-truth')) and not resetFlag:
            #print("\t Already exists.")
            continue
        print('\t\t Generating Density map for : ',os.path.basename(img_path)," :")

            # load matrix containing ground truth infos
        mat = io.loadmat(img_path.replace('.jpg','.mat').replace('images','ground-truth'))
        img= plt.imread(img_path)#768行*1024列
        density_map = np.zeros((img.shape[0],img.shape[1]))
        # points = mat["image_info"][0,0][0,0][0] #1546person*2(col,row)
        key=[el for el in list(mat) if el.lower().endswith('points')][0]
        points = [tuple(el) for el in mat[key]] #1546person*2(col,row)

            # Generate the density map
        density_map = dm_generator.generate_densitymap(img,points)

            # save density_map on disk
        np.save(img_path.replace('.jpg','.npy').replace('images','ground-truth'), density_map)
    print('\t Done.')
    return paths_dict


def create_samplers(dataSet_size,test_size=20,validation_size=10):
   
    indices = list(range(dataSet_size))
    split = int(np.floor(test_size * dataSet_size/100))
    val_split=int(np.floor(validation_size * (dataSet_size-split)/100))
    random_seed = 42

    np.random.seed(random_seed)
    np.random.shuffle(indices)
    train_sampler = SubsetRandomSampler(list(indices[split+val_split:]))
    validation_sampler= SubsetRandomSampler(list(indices[split:split+val_split]))
    test_sampler = SubsetRandomSampler(list(indices[:split]))
    return train_sampler,validation_sampler,test_sampler
    
def check_previous_loaders(loader_type,img_gtdm_paths,params:dict=None):
    '''
        Checks for previous versions of the loader, in order to optimize creating and generating Dataloaders.
    '''
    print("\t Checking for previous loader ...")
    if params is None:
        test_size=20
        batch_size=1
    else:
        test_size=params['test_size']
        batch_size=params['batch_size']
    restore_path=os.path.join(utils.BASE_PATH,'obj','loaders',loader_type,'saved2.pkl')    
   

    # saved_infos=utils.load_obj(os.path.join(restore_path,'saved.pkl'))
    # if saved_infos['paths']!=img_gtdm_paths: return None
    # if saved_infos['test_size']!=test_size: return None
    # if saved_infos['batch_size']!=batch_size: return None

    # return saved_infos['samplers']
    if not os.path.exists(restore_path) :
        return None
    if len( glob.glob(restore_path) )==0:
        return None
    restored=torch.load(restore_path)
    if not isinstance(restored,dict) or not 'paths_index' in restored: return None
    for couple in restored['paths_index']:
        if not couple in img_gtdm_paths:
            return None

    return restored        







def getloader(loader_type,img_gtdm_paths,restore_flag=True):
    '''
        Returns a new loader according to the passed type.
    '''
    print("####### Getting DataLoader...")
    if loader_type=="GenericLoader":
        return GenericLoader(img_gtdm_paths)


    

def getModel(model_type,load_saved=False,weightsFlag=False):
    '''
        Loads a models according to type, if a previous version was found, load it directely.
    '''
    print("####### Getting Model : ",model_type,"...")
    if load_saved and  os.path.exists(os.path.join(utils.BASE_PATH,'obj','models',model_type)):
        return torch.load(os.path.join(utils.BASE_PATH,'obj','models',model_type))
    if model_type=="MCNN":
        return MCNN(weightsFlag)
    elif model_type=="CSRNet":
        return CSRNet(weightsFlag)
    elif model_type=="SANet":
        return SANet()            
    elif model_type=="CCNN":
        return CCNN()            

def get_best_model(min_epoch,className):
    '''
        Loads the best model resulting from train.
    '''
    
    if not os.path.exists(os.path.join(utils.BASE_PATH , 'checkpoints2',className)):
        raise Exception("Cannot load model. Checkpoint directory not found!")
    if not os.path.exists(os.path.join(utils.BASE_PATH , 'checkpoints2',className,'epoch_'+str(min_epoch)+'.pth')):
        raise Exception("Cannot load model.Best epoch checkpoint does not exists!")

    return torch.load(os.path.join(utils.BASE_PATH , 'checkpoints2',className,'epoch_'+str(min_epoch)+'.pth'))

def show_plots(model):
    '''
        Display error's evaluation during train and test phase via plots.
    '''
    path=os.path.join(model.checkpoints_dir,'summary.json')
    if not os.path.exists(path):
        raise FileNotFoundError('Summary file for the model not found.')
    summary=utils.load_json(path)
    losses=summary['train_summary']
    train_loss=[]
    validations_loss=[]
    epochs=[]
    min_error_point=(summary['min_epoch'],summary['min_MAE'],'min error')
    min_loss_point=()
    for chkpt in losses:
        train_loss.append( chkpt['loss'])
        validations_loss.append( chkpt['mae'])
        epochs.append(chkpt['epoch'])
        if summary['min_loss']==chkpt['loss']:
            min_loss_point=(chkpt['epoch'],summary['min_loss'],'min train loss')


    plots.showLineChart([(epochs,train_loss),(epochs,validations_loss)], ['Train loss','Validation loss'], title=model.__class__.__name__+' Errors Plot', x_title='Epochs', y_title='Error', special_points=[min_error_point,min_loss_point])
    
ap = argparse.ArgumentParser()
ap.add_argument("-r", "--root", required=True,
	help="root path to DataSets location")
ap.add_argument("-m", "--model-type",type=str, required=False,default="CSRNet",
	help="Model Name (Case sensitive) {MCNN,CSRNet,SANet,CCNN}")
ap.add_argument("-n", "--new-train", type=bool, default=False,nargs='?',const=True,
	help="New train flag")
ap.add_argument("--no-loss-plot", type=bool, default=False,nargs='?',const=True,
	help="Choose to not show the loss/error plots")
ap.add_argument("--no-resume", type=bool, default=False,nargs='?',const=True,
	help="Resume training flag")
ap.add_argument("-s", "--skip-frames", type=int, default=30,
	help="# of skip frames between detections")
args = vars(ap.parse_args())

if __name__=="__main__":
    
   
    if args['root'] is not None:
        root = args['root']
    else :
        root = 'C:\\Users\\PC\\Desktop\\PFE related\\existing works\\Zhang_Single-Image_Crowd_Counting_CVPR_2016_paper code sample\\MCNN-pytorch-master\\MCNN-pytorch-master\\ShanghaiTech'
    dm_generator_type="knn_gaussian_kernal"
    dataset_names=["ShanghaiTech_partA","ShanghaiTech_partB"]
    dm_generator=None
    loader_type="GenericLoader"
    model_type=args['model_type']#sys.argv[2] if len(sys.argv)>2 else "CSRNet"
    model=None
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    resume_flag=not args['no_resume']
    i=0
    import time
    while True:
        i+=1
        if i>10:
            print('epoch: ',i,'\n\t error : ',i*2,' min_MAE: ',i*1.5)
        else:print('fgdfgdfgdfgf')
        time.sleep(3)
    # params={"lr":1e-6,
    #         "momentum":0.95,
    #         "maxEpochs":1000,
    #         "criterionMethode":'MSELoss',
    #         "optimizationMethod":'SGD'
    #         }
    params=getattr(trainsparams,model_type+'_PARAMS')

    print('Launching script with root=',args['root'],' model=',args['model_type'],'new train=',args['new_train'],' and resume=',resume_flag)
    
    if dm_generator_type=="knn_gaussian_kernal":
        dm_generator=KNN_Gaussian_Kernal_DMGenerator()

    datasets_paths=prepare_datasets(root,dataset_names,dm_generator)
    img_gtdm_paths=[(el["images"],el["ground-truth"]) for el in datasets_paths]

        # Modifications from here
    concat_paths=[(os.path.join(root_img,fname),os.path.join(root_dm,fname.replace('.jpg','.npy'))) for (root_img,root_dm) in img_gtdm_paths for fname in os.listdir(root_img) ]
    
    restore=check_previous_loaders(loader_type,concat_paths,dict(batch_size=params['batch_size'],test_size=20))
    if restore is None:
        dataset=BasicCrowdDataSet(concat_paths)
        train_sampler,validation_sampler,test_sampler=create_samplers(len(dataset))

        train_loader = torch.utils.data.DataLoader(dataset, batch_size=params['batch_size'],
                                                            sampler=train_sampler,num_workers=0)
        validation_loader = torch.utils.data.DataLoader(dataset, batch_size=params['batch_size'],
                                                            sampler=validation_sampler,num_workers=0)
        test_loader = torch.utils.data.DataLoader(dataset, batch_size=params['batch_size'],
                                                    sampler=test_sampler,num_workers=0)
        loader_backup=dict(paths_index=concat_paths,train_loader=train_loader,test_loader=test_loader,validation_loader=validation_loader)
        torch.save(loader_backup,os.path.join(utils.BASE_PATH,'obj','loaders',loader_type,'saved2.pkl'))                                            
    else:
        print('\t A previous version of the loader was found! Restoring samplers ...')
        train_loader,validation_loader,test_loader=restore['train_loader'],restore['validation_loader'],restore['test_loader']
        print('\t Done.')
      
    print('\t\t [Info] Dataset of ',len(concat_paths),' instances, Train size is ',len(train_loader),' Validation size is ',len(validation_loader),'Test size is ',len(test_loader))
    torch.cuda.empty_cache()
    # data_loader=getloader(loader_type,img_gtdm_paths)
    # samplers=check_previous_loaders(loader_type,img_gtdm_paths,dict(batch_size=params['batch_size'],test_size=20))
    # if samplers is None:
    #     dataloaders=data_loader.load(batch_size=params['batch_size'],save=True)
        
    # else:
    #     print('\t A previous version of the loader was found! Restoring samplers ...')
    #     dataloaders=data_loader.load_from_samplers(samplers,params=dict(batch_size=params['batch_size'],test_size=20))    
        
   
    

    #     # This loop is basically used in experimentations
    # # for train_loader,test_loader in dataloaders:
    # #     model.train_model(train_loader,test_loader,train_params)

    #     # ToDo: use listDataSet
    # merged_train_dataset,merged_test_dataset=data_loader.merge_datasets(dataloaders)
    # train_dataloader=torch.utils.data.DataLoader(merged_train_dataset)
    # test_dataloader=torch.utils.data.DataLoader(merged_test_dataset)
    
    model=getModel(model_type,load_saved=True)
        # defining train params
    train_params=TrainParams(device,model,params["lr"],params["momentum"],params["maxEpochs"],params["criterionMethode"],params["optimizationMethod"])
        # Launch the train
    # train_loss_list,test_error_list,min_epoch,min_MAE=model.train_model(merged_train_dataset,merged_test_dataset,train_params,resume=resume_flag,new_train=args['new_train'])
    train_loss_list,test_error_list,min_epoch,min_MAE=model.train_model(train_loader,validation_loader,train_params,resume=resume_flag,new_train=args['new_train'])
    print(train_loss_list,test_error_list,min_epoch,min_MAE)
    torch.cuda.empty_cache()
    _,model.min_MAE,model.min_epoch=model.load_chekpoint(os.path.join(model.checkpoints_dir,'epoch_'+str(min_epoch)+'.pth'))
    # Model.save(model)
    # gc.collect()

    model.eval_model(test_loader)
    # print('Evaluation Results',model.eval_model(test_dataloader))

        # Plots learning results
    if not args['no_loss_plot']:
        show_plots(model)

    
