import os
import shutil #可以用作移动文件
path = '../train'
filenames = os.listdir(path)
print(filenames[0].split('.'))
for filename in filenames:
    if filename.split('.')[0] == 'cat':
        shutil.move(path + '/' + filename, './image/cat')
    if filename.split('.')[0] == 'dog':
        shutil.move(path + '/' + filename, './image/dog')
