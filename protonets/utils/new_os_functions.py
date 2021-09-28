import os

def mkdir_if_not_exist(dir_name):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
        
        return 1

    return 0

def rename_file(parent_dir, curr_name, new_name):
    curr_name = os.path.join(parent_dir, curr_name) 
    new_name = os.path.join(parent_dir, new_name)

    os.rename(curr_name, new_name)