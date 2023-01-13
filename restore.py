import os
import glob
import time
import datetime
import numpy as np
import pandas as pd

# target datetime
dt_target = datetime.datetime(2023, 1, 2)

# list all directories recursively

def list_directories(root_dir):
    res = []
    for path, directories, files in os.walk(root_dir):
        for directory in directories:
            res.append(os.path.join(path, directory))
    return res

def list_python_files(dir):
    res = []
    for file in glob.glob(os.path.join(dir, '*.py')):
        res.append(file)
    return res

def backup_files(file):
    head, tail = os.path.split(file)
    bkglobstr = f"{head}/.~/{tail.replace('.py', '')}*"
    out = np.array(glob.glob(bkglobstr))
    # custom sort based on the last date modified
    dates = np.array([datetime.datetime.strptime(time.ctime(os.stat(x).st_mtime), '%a %b %d %H:%M:%S %Y') for x in out])
    idxsort = np.argsort(dates)
    out_sort = out[idxsort]
    dates_sort = dates[idxsort]

    df = pd.DataFrame(out_sort, index=dates_sort)

    return df

    

res = list_directories('src/osmg/')

not_backup = []

for thing in res:
    if '.~' not in thing:
        not_backup.append(thing)

not_backup.append('src/osmg')

# find all available filenames
files = {}
for thing in not_backup:
    files[thing] = list_python_files(thing)


# replace occurrence in all files

backup_files_dict = {}
for file_list in files.values():
    for file in file_list:
        df = backup_files(file)
        if len(df) != 0:
            # find the target backup version
            sub = df.index[df.index<dt_target]
            if len(sub) != 0:
                backup_date = sub[-1]
                date_diff = dt_target - backup_date
                # diff must be less than one day
                if date_diff < datetime.timedelta(days=5):
                    backup_file = df.loc[backup_date, 0]
                    backup_files_dict[file] = backup_file
                else:
                    backup_files_dict[file] = None
            else:
                backup_files_dict[file] = None
        else:
            backup_files_dict[file] = None
        
# # find the latest available version that was last edited yesterday

# backup_files_dict = {}
# for file_list in files.values():
#     for file in file_list:
#         df = backup_files(file)
#         if len(df) != 0:
#             # find the target backup version
#             sub = df.index[df.index<dt_target]
#             if len(sub) != 0:
#                 backup_date = sub[-1]
#                 date_diff = dt_target - backup_date
#                 # diff must be less than one day
#                 if date_diff < datetime.timedelta(days=5):
#                     backup_file = df.loc[backup_date, 0]
#                     backup_files_dict[file] = backup_file
#                 else:
#                     backup_files_dict[file] = None
#             else:
#                 backup_files_dict[file] = None
#         else:
#             backup_files_dict[file] = None
        

# # restore the current file with that file
# for update, source in backup_files_dict.items():
#     if update is not None and source is not None:
#         with open(source, 'r', encoding='utf-8') as f:
#             contents = f.read()
#         with open(update, 'w', encoding='utf-8') as f:
#             f.write(contents)


# # undo restore -- replace from backup
# for update, source in backup_files_dict.items():
#     head, tail = os.path.split(update)
#     source = f'../OpenSees_Model_Generator_bkp/{head}/{tail}'
#     with open(source, 'r', encoding='utf-8') as f:
#         contents = f.read()
#     with open(update, 'w', encoding='utf-8') as f:
#         f.write(contents)

