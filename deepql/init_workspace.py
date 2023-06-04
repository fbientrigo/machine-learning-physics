import os

def create_workspace():
    folder_names = ['data', 'errors', 'evolution_force', 'gen_data', 'model']
    
    for folder_name in folder_names:
        if not os.path.exists(folder_name):
            os.mkdir(folder_name)
