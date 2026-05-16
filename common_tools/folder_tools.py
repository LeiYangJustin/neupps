import glob
import os
import shutil

def find_no_ckpt_folder(path_to_check, save_to=None):
    logs = glob.glob(path_to_check+"/*/config.json")
    ckpts = glob.glob(path_to_check+"/*/10000.pth")

    logs_folder = [os.path.split(p)[0] for p in logs]
    ckpts_folder = [os.path.split(p)[0] for p in ckpts]

    logs_folder = sorted(logs_folder)
    ckpts_folder = sorted(ckpts_folder)

    no_ckpts_folder = []
    for l in logs_folder:
        flag = False
        for c in ckpts_folder:
            if c == l:
                flag = True
                break
        if not flag:
            no_ckpts_folder.append(l)

    print(len(ckpts_folder))
    print(len(logs_folder))

    print("no ckpts", len(no_ckpts_folder))
    print(no_ckpts_folder)

    if save_to is not None:
        assert save_to is str
        with open(save_to, "w") as f:
            f.write("folders have logs without ckpts\n")
            for l in no_ckpts_folder:
                f.write(l+"\n")

def remove_empty_folders(folders_to_remove):
    for f in folders_to_remove:
        if os.path.exists(f) and os.path.isdir(f):
            files = glob.glob(f+"/*")
            if len(files) < 3:
                print(files)
                shutil.rmtree(f)

def get_specified_ckpt_from_subfolders(parent_folder, epoch=100000, save_to=None, subfolder_name="/*"):
    
    folder_list = glob.glob(parent_folder+subfolder_name)
    print(len(folder_list), "subfolders under ", parent_folder)

    ckpt_path_list = []

    for f in folder_list:
        print(f)
        ckpt_path = os.path.join(f, f"{epoch}.pth")
        print(ckpt_path)
        if os.path.exists(ckpt_path):
            ckpt_path_list.append(ckpt_path)

    # if save_to is not None:
    #     assert type(save_to) is str
    #     with open(save_to, "w") as f:
    #         for c in ckpt_path_list:
    #             f.write(c+"\n")        

    if not os.path.exists(save_to):
        os.makedirs(save_to)


    for c in ckpt_path_list:
        print(c)    
        c_splits = c.split(os.sep)
        print(c_splits)

        # # logfolder = os.path.join(*c_splits[:-1], "logs") 
        # # shutil.copytree(logfolder, dst=os.path.join(foldername, "logs"))

        subfolder = os.path.join(save_to, c_splits[-2])
        if not os.path.exists(subfolder):
            os.makedirs(subfolder)
        filename = os.path.join(subfolder, f"{epoch}.pth")
        shutil.copyfile(c, filename)
        cfgfile = os.path.join(*c_splits[:-1], "config.json")
        shutil.copyfile(cfgfile, os.path.join(subfolder, "config.json"))


parent_folder = "res/lucy"
assert os.path.exists(parent_folder), "no such a folder"
get_specified_ckpt_from_subfolders(parent_folder, epoch=15000, subfolder_name="/exp_lucy_*_0430*", save_to="server_lucy_0430")
# find_no_ckpt_folder(parent_folder)





