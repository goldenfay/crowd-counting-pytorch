import torch
import torch.nn as NN
import torch.nn.functional as F
from torchvision import transforms
import pickle
import matplotlib.pyplot as plt
import matplotlib.cm as CM
import os,sys,inspect,glob,random,re,time,datetime,gc
import numpy as np

currentdir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
# User's module from another directory
sys.path.append(os.path.join(parentdir, "bases"))
sys.path.append(os.path.join(parentdir, "process"))
import utils
from params import *
from gitmanager import *
import storagemanager
import displays
mean_std = ([0.446139603853, 0.409515678883, 0.395083993673], [0.288205742836, 0.278144598007, 0.283502370119])
class Model(NN.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.params = TrainParams.defaultTrainParams()
        self.checkpoints_dir = os.path.join(
            utils.BASE_PATH, 'checkpoints', self.__class__.__name__)
        self.git_manager = GitManager(
            user='myuser', pwd='mypassword')
        self.git_manager.authentification()

    def build(self, weightsFlag):
        """
            Build Net Architecture
        """
        pass

    def train_model(self, train_dataloader, test_dataloader, train_params: TrainParams, resume=False,new_train=False):
        """
            Start training the model with specified parameters.
        """
        print("####### Training The model...")
        self.params = train_params
        self.optimizer = train_params.optimizer
        # Get the device (GPU/CPU) and migrate the model to it
        device = train_params.device
        print("\t Setting up model on ", device.type, "...")
        if not os.path.exists(self.checkpoints_dir):
            os.mkdir(self.checkpoints_dir)

        target_repo = self.git_manager.get_repo('checkpoints')
            # Initialize training variables
        print("\t Initializing ", "...")
        self.min_MAE = 10000
        self.min_epoch = 0
        train_loss_list = []
        test_error_list = []
        start_epoch = 0

        dirs=utils.list_dirs(self.checkpoints_dir)
        train_dirs=re.findall('Train_[0-9]+',' '.join(dirs))
        if len(train_dirs)==0:
            
            last_train=1#self.checkpoints_dir = os.path.join(self.checkpoints_dir,('Train_1'))
        else:
            last_train=max(sorted([int(re.sub('Train_','',dirname)) for dirname in train_dirs]))  
            
        # If resume option is specified, restore state of model and resume training
        if new_train or (not resume):
            if len(train_dirs)==0:
                self.checkpoints_dir = os.path.join(self.checkpoints_dir,'Train_1')
                
            else:
                self.checkpoints_dir = os.path.join(self.checkpoints_dir, 'Train_'+str(last_train+1) )  

        else:
            self.checkpoints_dir = os.path.join(self.checkpoints_dir, 'Train_'+str(last_train))
            params_hist = [utils.extract_number(file_path) for file_path in glob.glob(
                os.path.join(os.path.join(self.checkpoints_dir), '*.pth'))]
            
            if len(params_hist) > 0:
                print("\t Restore Checkpoints2 found! Resuming training...")
                sorted_hist = sorted(params_hist)
                start_epoch = max(sorted_hist)
                last_epoch = glob.glob(os.path.join(os.path.join(
                    self.checkpoints_dir, 'epoch_'+str(start_epoch)+'.pth')))[0]

                _, self.min_MAE, self.min_epoch = self.load_chekpoint(
                    last_epoch)

                files_to_push = []
                for epoch in sorted_hist:
                    if epoch != self.min_epoch and epoch != start_epoch and epoch!= train_params.maxEpochs:
                        path = glob.glob(os.path.join(os.path.join(
                            self.checkpoints_dir, 'epoch_'+str(epoch)+'.pth')))[0]
                        obj = torch.load(path, map_location=device)
                        if obj['model_state_dict'] is not None or obj['optimizer_state_dict']is not None:
                            obj['model_state_dict'] = None
                            obj['optimizer_state_dict'] = None
                            self.save_checkpoint(obj, path)
                            files_to_push.append(path)

                if len(files_to_push)>0:
                    res = self.git_manager.push_files(
                        target_repo, files_to_push, 'checkpoints migration', branch=self.__class__.__name__,dir=os.path.basename(self.checkpoints_dir))
                    if isinstance(res, int)and res == len(files_to_push):
                        print(
                            '\t Successfully comitted previous checkpoints(', res, ' files).')

                    else:
                        raise RuntimeError('Couldn\'t push all files')
        self.to(device)
       
        start_epoch += 1

            # Start Train
        for epoch in range(start_epoch, train_params.maxEpochs+1):
            start = time.time()
                # Set the Model on training mode
            self.train()
            epoch_loss = 0
                # Run training pass (feedforward,backpropagation,...) for each batch
            for i, (img, gt_dmap) in enumerate(train_dataloader):
                torch.cuda.empty_cache()
                img = img.to(device).detach()
                gt_dmap = gt_dmap.to(device).detach()
                    # forward propagation
                try:
                    est_dmap = self(img)
                except RuntimeError as e:
                    if 'out of memory' in str(e):
                        torch.cuda.empty_cache()
                        est_dmap = self(img)

                if not est_dmap.size() == gt_dmap.size():

                    est_dmap = F.interpolate(est_dmap, size=(
                        gt_dmap.size()[2], gt_dmap.size()[3]), mode='bilinear')
                    # est_dmap = F.interpolate(est_dmap, size=(
                    #     gt_dmap.size()[1], gt_dmap.size()[2]), mode='bilinear')
                # if torch.isnan(est_dmap): print('Estimated is nan')
                # if torch.isnan(gt_dmap): print('Ground truth is nan')
                    # calculate loss
                loss = train_params.criterion(est_dmap, gt_dmap)
                epoch_loss += loss.item()
                # if i%5==0: print(est_dmap.data.sum(),gt_dmap.data.sum())
               
                torch.cuda.empty_cache()
                
                    # Setting gradient to zero ,(only in pytorch , because of backward() that accumulate gradients)
                self.optimizer.zero_grad()
                    # Backpropagation
                loss.backward()
                self.optimizer.step()
                del img, gt_dmap, est_dmap
                
            print("\t epoch:"+str(epoch)+"\n", "\t\t loss:",
                  epoch_loss/len(train_dataloader))
            train_loss_list.append(epoch_loss/len(train_dataloader))      

           

                # Set the Model on validation mode
            self.eval()
            MAE = 0
            MSE = 0
            for i, (img, gt_dmap) in enumerate(test_dataloader):
                img = img.to(device)
                gt_dmap = gt_dmap.to(device)
                    # forward propagation
                try:
                    est_dmap = self(img)
                except RuntimeError as e:
                    if 'out of memory' in str(e):
                        torch.cuda.empty_cache()
                        est_dmap = self(img)

                if not est_dmap.size() == gt_dmap.size():
                    est_dmap = F.interpolate(est_dmap, size=(
                        gt_dmap.size()[2], gt_dmap.size()[3]), mode='bilinear')
                    # est_dmap = F.interpolate(est_dmap, size=(
                    #     gt_dmap.size()[1], gt_dmap.size()[2]), mode='bilinear')
                mae=abs(est_dmap.data.sum()-gt_dmap.data.sum()).item()
                MAE += mae
                MSE += mae**2
                del img, gt_dmap, est_dmap
                torch.cuda.empty_cache()
            MAE = MAE/len(test_dataloader)
            MSE = np.math.sqrt(MSE/len(test_dataloader))

            if MAE < self.min_MAE:
                self.min_MAE = MAE
                self.min_epoch = epoch
            test_error_list.append(MAE)
            print("\t\t error:"+str(MAE)+" min_MAE:" +
                  str(self.min_MAE)+" min_epoch:"+str(self.min_epoch))

            end = time.time()      
            check_point = {
                'model_state_dict': self.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'loss': epoch_loss/len(train_dataloader),
                'mae': MAE,
                'min_MAE': self.min_MAE,
                'min_epoch': self.min_epoch,
                'duration':str(datetime.timedelta(seconds=end-start))
            }
                # Save checkpoint
            self.save_checkpoint(check_point, os.path.join(
                self.checkpoints_dir, 'epoch_'+str(epoch)+'.pth'))
   
        
            # Save training summary into disk
        self.make_summary(finished=True)
    
        print('Training finished.')
        return (train_loss_list, test_error_list, self.min_epoch, self.min_MAE)

    def retrain_model(self, params=None):
        pass

    def eval_model(self, test_dataloader, eval_metrics='all'):
        """
            Evaluate/Test the model after train is completed and output performence metrics used for test purpose.
        """
        print("####### Validating The model...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)
        self.eval()
        MAE = 0
        MSE = 0
        cpt = 1
        fig = plt.figure(figsize=(64, 64))
        all=test_dataloader.__len__()
        with torch.no_grad():
            for i, (img, gt_dmap) in enumerate(test_dataloader):
                    # Transfer input and target to Device(GPU/CPU)
                img = img.to(device)
                gt_dmap = gt_dmap.to(device)

                    # Forward propagation
                est_dmap = self(img)
                if i%50==0: print('\t\t (',est_dmap.data.sum(),gt_dmap.data.sum(),')')
                mae=abs(est_dmap.data.sum()-gt_dmap.data.sum()).item()
                MAE += mae
                MSE += mae**2
                # Show the estimated density map via matplotlib
                if i % 10 == 0:
                    print(img.shape,est_dmap.shape,gt_dmap.shape)
                
                    ax=fig.add_subplot(int(all/10),3,cpt)
                    ax.title.set_text('Original image')
                    img = np.asanyarray(img.squeeze().permute(1,2,0).cpu(),dtype=np.int)
                    plt.imshow(img)

                    ax1=fig.add_subplot(int(all/10),3,cpt+1)
                    ax1.title.set_text('Estimated crowd number :'+str(np.sum(np.asarray(est_dmap.cpu()))))
                    est_dmap = est_dmap.squeeze().cpu().numpy()
                    plt.imshow(est_dmap, cmap=CM.jet)

                    ax2=fig.add_subplot(int(all/10),3,cpt+2)
                    ax2.title.set_text('Ground Truth number'+str(np.sum(np.asarray(gt_dmap.cpu()))))
                    gt_dmap=gt_dmap.squeeze().cpu().numpy()
                    plt.imshow(gt_dmap, cmap=CM.jet)
                    
                    cpt+=3
                del img, gt_dmap, est_dmap
            plt.show()    
            MAE = MAE/len(test_dataloader)
            MSE = np.math.sqrt(MSE/len(test_dataloader))
        # gc.collect(0)    
        print("\t Test MAE : ", MAE, "\t test MSE : ", MSE)
        self.make_summary(finished=True, test_mse=MSE, test_mae=MAE)
        print('Validation finished.')
        return (MAE, MSE)

    def save_checkpoint(self, chkpt, path):
        """
            Save a checkpoint in the specified path.
        """
        # If the directory doesn't exist, create it.
        utils.make_path(os.path.split(path)[0])
        # torch.save(chkpt, path)
        env = 'drive' if 'drive/My Drive' in path else 'os'
        flag = storagemanager.save_file(path, chkpt, env, self.min_epoch)

        if flag == 0:  # There isn't available space on drive
            print("\t Optimizing space...")
            parent_path = os.path.split(path)[0]
            sorted_hist = sorted([utils.extract_number(
                file_path) for file_path in glob.glob(os.path.join(parent_path, '*.pth'))])
            files_to_push = []
            for epoch in sorted_hist:
                if epoch != self.min_epoch:
                    path = glob.glob(os.path.join(os.path.join(
                        parent_path, 'epoch_'+str(epoch)+'.pth')))[0]
                    obj = torch.load(path, map_location=torch.device(
                        'cuda' if torch.cuda.is_available() else 'cpu'))
                    if obj['model_state_dict'] is not None or obj['optimizer_state_dict']is not None:
                        obj['model_state_dict'] = None
                        obj['optimizer_state_dict'] = None
                        self.save_checkpoint(obj, path)
                        files_to_push.append(path)
            print("\t Pushing checkpoints to github...")

            if len(files_to_push)>0:
                target_repo = self.git_manager.get_repo('checkpoints')
                res = self.git_manager.push_files(
                    target_repo, files_to_push, 'checkpoints migration', branch=self.__class__.__name__,dir=os.path.basename(self.checkpoints_dir))
                if isinstance(res, int)and res == len(files_to_push):
                    print('\t Successfully comitted previous checkpoints(', res, ' files).')

                else:
                    raise RuntimeError('Couldn\'t push all files')

            torch.save(chkpt, path)

    
    def load_chekpoint(self, path):
        """
            Load a checkpoint from the specified path in order to resume training.
        """
        device=torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        chkpt = torch.load(path, map_location=device)
        self.load_state_dict(chkpt['model_state_dict'])
        self.optimizer.load_state_dict(chkpt['optimizer_state_dict'])
        self.optimizer=self.migrate(self.optimizer,device)

        return chkpt['loss'], chkpt['min_MAE'], chkpt['min_epoch']

    @staticmethod
    def migrate(optimizer,device):
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        return optimizer            

    @staticmethod
    def save(model):
        """
            Save the whole model. This method is called once training is finished in order to keep the best model.

        """
        path = os.path.join(utils.BASE_PATH, 'obj', 'models',
                            model.__class__.__name__+'.pth')
        utils.make_path(os.path.split(path)[0])
        try:
            torch.save(model, path)
        except:
            model.clearstate()
            torch.save(model, path,)   

    def make_summary(self, finished=False, test_mse=None, test_mae=None):
        path = os.path.join(self.checkpoints_dir, 'summary.json')
        summary = {
            'status': 'finished' if finished else 'training',
            'min_epoch': self.min_epoch,
            'min_MAE': self.min_MAE,
            'train_params': {
                'lr': self.params.lr,
                'momentum': self.params.momentum,
                'maxEpochs': self.params.maxEpochs,
                'criterionMethode': self.params.criterion.__class__.__name__,
                'optimizationMethod': self.params.optimizer.__class__.__name__
            }

        }
        if finished:
            device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu')
            checkpoint_files = glob.glob(os.path.join(
                os.path.join(self.checkpoints_dir), '*.pth'))
            checkpoints = []
            min_loss=1000000
            for chkpt in checkpoint_files:
                recap = {}
                obj = torch.load(chkpt, map_location=device)
                dic = {
                    'epoch': utils.extract_number(chkpt),
                    'loss': obj['loss'],
                    'mae': obj['mae'],

                }
                if dic['loss']<min_loss:
                    min_loss=dic['loss']
                checkpoints.append(dic)
        summary['train_summary'] = checkpoints
        summary['min_loss']=min_loss
        if test_mae is not None and test_mse is not None:
            summary['test_summary'] = {
                'mae': test_mae,
                'mse': test_mse
            }

        utils.make_path(os.path.split(path)[0])
        utils.save_json(summary, path)

