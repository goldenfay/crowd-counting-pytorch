import os,traceback,time,datetime
from github import InputGitTreeElement
from github import Github
from github.Repository import Repository
import base64
from github.ContentFile import ContentFile
class GitManager:

    def __init__(self,access_token=None,user=None,pwd=None):
       self.token=access_token
       self.user=user
       self.pwd=pwd
       

    def authentification(self):
        '''
            Authenticate user to github either by access token or by (username,password).
            NOTE : Authentication via username/password is deprecated, We recommand using access token instead.
             
        '''
    
        
        self.gth = Github(self.token) if self.token is not None else Github(self.user,self.pwd)

        return self.gth

    
    def get_repo(self,repo_name) ->Repository:
        '''
            Returns a repository based on its name.
        '''
        
        for repo in self.gth.get_user().get_repos():
            if repo.name==repo_name:
                return repo
            
        return None

    @classmethod    
    def get_repo_files(cls,repo:Repository):
        '''
            Lists all the files in a specific repository.
        '''
        return [el.path for el in repo.get_contents('')]

    def push_files(self,repo:Repository,files_list,push_msg,branch='master',dir=''):
        '''
            Push a list of files to a specific directory in a specific branch of a specific repository.

        '''
        if dir!='': dir+='/'
        branch_ref = repo.get_git_ref('heads/'+branch)
        branch_sha = branch_ref.object.sha
        master_sha=repo.get_git_ref('heads/'+branch).object.sha
        base_tree = repo.get_git_tree(master_sha,recursive=True)
        element_list = list()
        for entry in files_list:
            with open(entry, 'rb') as input_file:
                data = input_file.read()
            if entry.endswith('.pth'):
                data = base64.b64encode(data)
            block=data.decode("utf-8")    
            blob = repo.create_git_blob(block, "base64")    
            element = InputGitTreeElement(dir+os.path.basename(entry), '100644', 'blob', sha=blob.sha)
            element_list.append(element)  
        if len(element_list)!=0:
                  
            tree = repo.create_git_tree(element_list, base_tree)
            parent = repo.get_git_commit(branch_sha)
            commit = repo.create_git_commit(push_msg, tree, [parent])
            
            branch_ref.edit(commit.sha)
            self.log_commit('commit.txt',files_list)
            print('\t Done.',end=' ')
        return len(element_list) 

    @classmethod
    def log_commit(cls,logfile_path,files_list):
        '''
            Log commited files into a log file.
        '''
        with open(logfile_path,'a') as f:
            f.write('Commit :'+datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")+'\nList :'+','.join(files_list)+'\n')
