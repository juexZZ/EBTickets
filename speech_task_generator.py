# few shot speech classification task generator
# based on the Relation Networks: https://github.com/floodsung/LearningToCompare_FSL
import torch
from torch.utils.data import DataLoader,Dataset
import random
import os
import numpy as np
from torch.utils.data.sampler import Sampler

def voxceleb_speech_folder(data_folder, train_num):
    # organize data folders
    # data file structure : data_folder/xvectorfile(merged, per person)
    speaker_files = [os.path.join(data_folder, persons) for persons in os.listdir(data_folder)]
    random.seed(1)
    random.shuffle(speaker_files)

    metatrain_speaker_files = speaker_files[:train_num]
    metaval_speaker_files = speaker_files[train_num:]

    return metatrain_speaker_files, metaval_speaker_files

class VoxFewshotTask(object):

    def __init__(self, speaker_files, num_classes, train_num, test_num):

        self.speaker_files = speaker_files
        self.num_classes = num_classes
        self.train_num = train_num
        self.test_num = test_num
        self.files2vec = dict()
        for f in self.speaker_files:
            with open(f, 'r') as vecs:
                 temp = [line.split()[2:-1] for line in vecs]
            self.files2vec[f] = random.sample(temp, len(temp))

    def sample_episode(self):
        # sample classes (persons)
        class_files = random.sample(self.speaker_files,self.num_classes)
        labels = np.array(range(len(class_files)))# assign labels to sampled classes
        labels = dict(zip(class_files, labels))
        #print(labels)

        self.train_vecs = []
        self.test_vecs = []
        self.train_labels = []
        self.test_labels = []
        for f in class_files:
            #print(f, labels[f])
            self.train_vecs += self.files2vec[f][:self.train_num]
            self.test_vecs += self.files2vec[f][self.train_num:self.train_num+self.test_num]
            self.train_labels += [labels[f]]*self.train_num
            self.test_labels += [labels[f]]*self.test_num
            #print(self.train_labels)
            #print(self.test_labels)


class FewShotDataset(Dataset):

    def __init__(self, task, split='train'):
        self.task = task
        self.split = split
        self.speech_vecs = self.task.train_vecs if self.split == 'train' else self.task.test_vecs
        self.labels = self.task.train_labels if self.split == 'train' else self.task.test_labels

    def __len__(self):
        return len(self.speech_vecs)

    def __getitem__(self, idx):
        raise NotImplementedError("This is an abstract class. Subclass this class for your particular dataset.")

class VoxFewShot(FewShotDataset):

    def __init__(self, *args, **kwargs):
        super(VoxFewShot, self).__init__(*args, **kwargs)

    def __getitem__(self, idx):
        vec = self.speech_vecs[idx]
        vec = np.array([float(ele) for ele in vec], dtype=np.float32)
        label = self.labels[idx]
        return vec, label

class ClassBalancedSampler(Sampler):
    ''' Samples 'num_inst' examples each from 'num_cl' pools
        of examples of size 'num_per_class' '''

    def __init__(self, num_per_class, num_classes, num_inst,shuffle=True):
        self.num_per_class = num_per_class
        self.num_classes = num_classes
        self.num_inst = num_inst
        self.shuffle = shuffle

    def __iter__(self):
        # return a single list of indices, assuming that items will be grouped by class
        if self.shuffle:
            batch = [[i+j*self.num_inst for i in torch.randperm(self.num_inst)[:self.num_per_class]] for j in range(self.num_classes)]
        else:
            batch = [[i+j*self.num_inst for i in range(self.num_inst)[:self.num_per_class]] for j in range(self.num_classes)]
        batch = [item for sublist in batch for item in sublist]

        if self.shuffle:
            random.shuffle(batch)
        return iter(batch)

    def __len__(self):
        return 1

def get_data_loader(task, num_per_class=1, split='train',shuffle=True):
    # NOTE: batch size here is # instances PER CLASS

    dataset = VoxFewShot(task,split=split)
    #print(len(dataset.speech_vecs))
    #print(len(dataset.labels))

    if split == 'train':
        sampler = ClassBalancedSampler(num_per_class, task.num_classes, task.train_num,shuffle=shuffle)
    else:
        sampler = ClassBalancedSampler(num_per_class, task.num_classes, task.test_num,shuffle=shuffle)
    loader = DataLoader(dataset, batch_size=num_per_class*task.num_classes, sampler=sampler)

    return loader



