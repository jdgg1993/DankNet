import os
import random
import shutil

rootdir = './data/dank'
outdir = './data/not_dank'
dirsAndFiles = {}   # here we store a structure  {folder: [file1, file2], folder2: [file2, file4] }
dirs = [x[0] for x in os.walk(rootdir)] # here we store all sub-dirs

for dir in dirs:
    dirsAndFiles[dir] = [f for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f))]

for (dir, files) in dirsAndFiles.items():
    for i in range(int(0.2*len(files))):  # copy 20% of files
        fe = random.choice(files)
        if fe != '.DS_Store':
            files.remove(fe)
            shutil.move(os.path.join(dir, fe), outdir)