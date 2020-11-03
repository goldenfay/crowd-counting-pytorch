import os,sys,inspect,psutil
import numpy as np
import torch
from torch.utils.data.sampler import SubsetRandomSampler

currentdir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
# User's modules from another directory
sys.path.append(os.path.join(parentdir, "bases"))
import utils
import datasets
from datasets import *


class Loader:
    def __init__(self, reset_samplers=True):
        self.reset_samplers_flag = reset_samplers

    def load(self,test_size=20,batch_size=1,shuffle_flag=True,save=False): pass

    @staticmethod
    def merge_datasets(loaders_list: list, shuffleFlag=True):
        train_set = []
        test_set = []
        dataset = torch.utils.data.Dataset()
        for train_loader, test_loader in loaders_list:
            for index, features in enumerate(train_loader):
                train_set.append(features)
            for index, features in enumerate(test_loader):
                test_set.append(features)
        del train_loader, test_loader, features
        if shuffleFlag:
            np.random.shuffle(train_set)
            np.random.shuffle(test_set)

        train_dataset = BasicDataSet(train_set)
        test_dataset = BasicDataSet(test_set)
        return train_dataset, test_dataset


class SimpleLoader(Loader):

    def __init__(self, img_rootPath, gt_dmap_rootPath, reset_samplers=True):
        super(SimpleLoader, self).__init__(reset_samplers)
        self.img_rootPath = img_rootPath
        self.gt_dmap_rootPath = gt_dmap_rootPath

    def load(self,test_size=20, batch_size=1, shuffle_flag=True,save=False):
        dataset = CrowdDataset(self.img_rootPath, self.gt_dmap_rootPath)
        self.dataSet_size = len(dataset)
        indices = list(range(self.dataSet_size))
        split = int(np.floor(test_size * self.dataSet_size/100))

        load_flag = False

        if not self.reset_samplers_flag and utils.path_exists('./obj/loaders/samplers.pkl'):
                print("\t Found a sampler restore point...")
                samplers_recov = torch.load('../../obj/loaders/samplers.pkl')
                for obj in samplers_recov:
                    if obj['img_rootPath'] == self.img_rootPath and obj['gt_dmap_rootPath'] == self.gt_dmap_rootPath:
                        train_sampler = obj['train_sampler']
                        test_sampler = obj['test_sampler']
                        load_flag = True

        if not load_flag:
            random_seed = 30
            if shuffle_flag:
                np.random.seed(random_seed)
                np.random.shuffle(indices)

            train_sampler = SubsetRandomSampler(list(indices[split:]))
            test_sampler = SubsetRandomSampler(list(indices[:split]))

        train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                   sampler=train_sampler,num_workers=0)
        test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                  sampler=test_sampler,num_workers=0)

        if save:
            torch.save({'paths':(self.img_rootPath,self.gt_dmap_rootPath),'loaders':[(train_loader, test_loader)]},
            os.path.join(utils.BASE_PATH,'obj','loaders','samplers.pkl'))
        return [(train_loader, test_loader)]


class GenericLoader(Loader):

    def __init__(self, img_gt_dmap_list, reset_samplers=False):
        super(GenericLoader, self).__init__(reset_samplers)
        self.img_gt_dmap_list = img_gt_dmap_list

    def load(self,test_size=20,batch_size=1,shuffle_flag=True,save=False):
        print('\t Loading DataSets ...')
        all_datasets = []
        load_flag = False
        if save:samplers_list=[]

        if not self.reset_samplers_flag and utils.path_exists('./obj/loaders/generic_samplers.pkl'):
            print("\t Found a sampler restore point...")
            samplers_recov = torch.load('../../obj/loaders/samplers.pkl')
            img_paths = [obj['train_sampler'] for obj in self.img_gt_dmap_list]
            gt_map_paths = [obj['test_sampler']
                            for obj in self.img_gt_dmap_list]
            load_flag = True

            list_samplers = [obj for obj in samplers_recov
                             if obj['img_rootPath']in img_paths and obj['dm_root_path'] in gt_map_paths]

            # if ==self.img_rootPath and ==self.gt_dmap_rootPath:
            #     train_sampler=obj['train_sampler']
            #     test_sampler=obj['test_sampler']
            #     load_flag=True
        if load_flag:
            print("\t Found dataset from restor point, loading samples....")
            for obj in list_samplers:
                dataset = CrowdDataset(
                    obj['img_rootPath'], obj['dm_root_path'])
                all_datasets.append((torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                                 sampler=obj['train_sampler'],num_workers=0),

                                     torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                                 sampler=obj['test_sampler'],num_workers=0)
                                     )
                                    )
            print("\t Done. Dataset restored.")

        else:
            
            for img_root_path, dm_root_path in self.img_gt_dmap_list:
                if psutil.virtual_memory().percent>=98:
                    print('\t [Warning] Memory is about to overflow')
                dataset = CrowdDataset(img_root_path, dm_root_path)
                dataSet_size = len(dataset)

                indices = list(range(dataSet_size))
                split = int(np.floor(test_size * dataSet_size/100))

                random_seed = 30
                if shuffle_flag:
                    np.random.seed(random_seed)
                    np.random.shuffle(indices)

                train_sampler = SubsetRandomSampler(list(indices[split:]))
                test_sampler = SubsetRandomSampler(list(indices[:split]))

                train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                           sampler=train_sampler,num_workers=0)
                test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                          sampler=test_sampler,num_workers=0)

                all_datasets.append((train_loader, test_loader))

                if save:
                    samplers_list.append((train_sampler,test_sampler))

                del dataset,train_sampler,test_sampler,train_loader,test_loader

        if save:
            utils.make_path(os.path.join(utils.BASE_PATH,'obj','loaders',self.__class__.__name__))
            utils.save_obj({
                'paths': self.img_gt_dmap_list,
                'samplers': samplers_list,
                'test_size':test_size,
                'batch_size':batch_size
            },
            os.path.join(utils.BASE_PATH,'obj','loaders',self.__class__.__name__,'saved.pkl')
            )
        return all_datasets

    def load_from_samplers(self,samplers, params:dict={'test_size':20,'batch_size':1}):
        
        print('\t Loading DataSets from previous samplers...')
        all_datasets=[]
        for i in range(len(self.img_gt_dmap_list)):
            img_root_path, dm_root_path=self.img_gt_dmap_list[i]
            dataset = CrowdDataset(img_root_path, dm_root_path)

            train_loader = torch.utils.data.DataLoader(dataset, batch_size=params['batch_size'],
                                                           sampler=samplers[i][0],num_workers=0)
            test_loader = torch.utils.data.DataLoader(dataset, batch_size=params['batch_size'],
                                                          sampler=samplers[i][1],num_workers=0)

            all_datasets.append((train_loader, test_loader))

        del dataset,train_loader,test_loader
        print('\t Done')
        return all_datasets    


