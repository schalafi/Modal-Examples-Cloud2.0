import os 

def check_dir(dir_path:str):
    """
    dir_path: str
        check  of the dir exists 
        if not, it will create it.
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    
    

    
