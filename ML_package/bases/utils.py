import os, glob,sys,inspect,re
from concurrent.futures.process import ProcessPoolExecutor

    # Root path of the ML Package
BASE_PATH=os.path.dirname(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))))

sys.path.append(os.path.join(BASE_PATH, "models"))


def path_exists(path):
    '''
        Check if a relative path separated with '/' exists.
    '''
    path=[el for el in path.split('/') if el!='.']
    
    return os.path.exists(os.path.join(BASE_PATH,*path))


def list_dirs(path):
    '''
        Lists all child directories found in path
    '''
    return [el for el in os.listdir(path) if not os.path.isfile(el)]


def make_path(dir_path):
    '''
        Make a directory (and all subdirectories) in system if it doesn't exits.
    '''
    if not os.path.exists(dir_path):
        make_path(os.path.split(dir_path)[0])
        os.mkdir(dir_path)

def split_path(path):
    '''
        Split a path into all directories composing it.
    '''
    ( head, tail ) = os.path.split(path)
    return split_path(head) + [ tail ] if head and head != path else [ head or tail ]


def save_json(dictionary,path):
    '''
        Save a dictionnary in json format.
    '''
    import json
    with open(path,'w') as outFile:
        json.dump(dictionary,outFile)

def load_json(path):
    '''
        Load a dictionnary from a .json file.
    '''
    import json
    with open(path,'r') as inFile:
        return json.load(inFile)

def save_obj(obj,path):
    '''
        Save an object as binary file.
    '''
    import pickle
    with open(path, 'wb') as outFile:
        pickle.dump(obj, outFile, protocol=pickle.HIGHEST_PROTOCOL)

def load_obj(path):
    '''
        Load an object from a binary file.
    '''
    import pickle
    with open(path, 'rb') as inFile:
        return pickle.load(inFile)

def extract_number(path):
    '''
        Extract number from a pth (only the last occurence)
    '''
    return int(re.sub("[^0-9]+","",path[list(re.finditer("[\\\/]",path))[-1].start(0):]))


def parallel_processing(func,args,workers):
    '''
        Basic function for executing parallel process using same function for all workers.
    '''
    with ProcessPoolExecutor(workers) as ex:
            res = ex.map(func, args)
    return list(res)