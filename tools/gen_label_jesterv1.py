# Code for "TSM: Temporal Shift Module for Efficient Video Understanding"
# arXiv:1811.08383
# Ji Lin*, Chuang Gan, Song Han
# {jilin, songhan}@mit.edu, ganchuang@csail.mit.edu
# ------------------------------------------------------
# Code adapted from https://github.com/metalbubble/TRN-pytorch/blob/master/process_dataset.py
# processing the raw data of the video Something-Something-V1
"""
将下载下来的4个csv文件放在tools文件夹下面
直接在本文件右击Run运行，无需在根目录下运行
将生成的3个txt文件剪切到dataset/jester文件夹下
"""
import os

if __name__ == '__main__':
    dataset_name = 'jester-v1'
    with open('%s-labels.csv' % dataset_name) as f:
        lines = f.readlines()
    categories = []
    for line in lines:
        line = line.rstrip()
        categories.append(line)
    categories = sorted(categories)  # 这句代码会让生成的category.txt里的标签顺序跟csv里的不一样
    with open('../dataset/jester/category.txt', 'w') as f:
        f.write('\n'.join(categories))

    dict_categories = {}
    for i, category in enumerate(categories):
        dict_categories[category] = i

    files_input = ['%s-validation.csv' % dataset_name, '%s-train.csv' % dataset_name]
    files_output = ['val_videofolder.txt', 'train_videofolder.txt']
    for (filename_input, filename_output) in zip(files_input, files_output):
        with open(filename_input) as f:
            lines = f.readlines()
        folders = []
        idx_categories = []
        for line in lines:
            line = line.rstrip()
            items = line.split(';')  # jester提供的csv里面文件夹名称和手势标签名称是通过;分隔的
            folders.append(items[0])  # folders存的是一串代表文件夹名的数字，范围从1到148092
            idx_categories.append(dict_categories[items[1]])  # idx_categories存的是一串代表手势id的数字，避免了名称长单词之间有空格，范围从0到26
        output = []
        for i in range(len(folders)):
            curFolder = folders[i]
            curIDX = idx_categories[i]
            # counting the number of frames in each video folders
            dir_files = os.listdir(os.path.join('../dataset/jester/20bn-jester-v1', curFolder))

            #                           后面代码会在'jester/20bn-jester-v1/'前面接上ROOT_DATASET = './dataset/'
            output.append('%s %d %d' % (curFolder, len(dir_files), curIDX))  # txt文件里每行的内容
            print('%d/%d' % (i, len(folders)))
        with open(filename_output, 'w') as f:
            f.write('\n'.join(output))
