# training script for voxceleb fewshot learninng
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
import numpy as np
import speech_task_generator as tg
import os
import math
import argparse
import random

parser = argparse.ArgumentParser(description="Few Shot Speech Classification -> diarisation")
parser.add_argument("-dir", default = "../vecs/")
parser.add_argument("--train_cls", type=int, default=30)# how many persons will be in the training part
parser.add_argument("-f","--feature_dim",type = int, default = 512)
parser.add_argument("-r","--relation_dim",type = int, default = 8)
parser.add_argument("-w","--class_num",type = int, default = 5)
parser.add_argument("-s","--sample_num_per_class",type = int, default = 5)
parser.add_argument("-b","--batch_num_per_class",type = int, default = 15)
parser.add_argument("-e","--episode",type = int, default= 1000000)
parser.add_argument("-t","--test_episode", type = int, default = 1000)
parser.add_argument("-l","--learning_rate", type = float, default = 0.001)
parser.add_argument("-g","--gpu",type=int, default=0)
parser.add_argument("-u","--hidden_unit",type=int,default=10)
args = parser.parse_args()

# Hyper Parameters
DATA_DIR = args.dir
TRAIN_NUM = args.train_cls
FEATURE_DIM = args.feature_dim
RELATION_DIM = args.relation_dim
CLASS_NUM = args.class_num
SAMPLE_NUM_PER_CLASS = args.sample_num_per_class
BATCH_NUM_PER_CLASS = args.batch_num_per_class
EPISODE = args.episode
TEST_EPISODE = args.test_episode
LEARNING_RATE = args.learning_rate
GPU = args.gpu
HIDDEN_UNIT = args.hidden_unit

# modules
# Relation modules for speech
class RelationNetwork(nn.Module):
    """Relation Network"""
    def __init__(self, input_size, hidden_size):
        super(RelationNetwork, self).__init__()
        self.input_size = input_size # vector feature dim
        self.fc1 = nn.Linear(input_size*2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, sample, query, num_class):
        """
        sample: (sample_per_class x num_class) x vec_dim
        query: (batch_per_class x num_class) x vec_dim
        """
        sample = sample.view(num_class, -1, self.input_size)
        sample_num_per_class = sample.size(1)
        sample = torch.sum(sample, 1).squeeze(1) # sum within each class -> num_class x vec_dim

        query = query.view(num_class, -1, self.input_size)
        batch_num_per_class = query.size(1)
        # align
        sample_ext = sample.unsqueeze(0).repeat(batch_num_per_class*num_class, 1, 1) # (batch_per_class x num_class) x num_class x vec_dim
        query = query.view(-1, self.input_size)
        query_ext = query.unsqueeze(0).repeat(num_class, 1, 1) # num_classes x (batch_per_class x num_class) x vec_dim
        query_ext = torch.transpose(query_ext, 0, 1) # (batch_per_class x num_class) x num_class x vec_dim
        # concat
        relation_pairs = torch.cat((sample_ext, query_ext), 2).view(-1, self.input_size*2)
        # calculate relations
        out = F.relu(self.fc1(relation_pairs))
        out = torch.sigmoid(self.fc2(out))
        out = out.view(-1, num_class)
        return out

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm') != -1:
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        n = m.weight.size(1)
        m.weight.data.normal_(0, 0.01)
        m.bias.data = torch.ones(m.bias.data.size())

def main():
    # Step1: init data files:
    print("init data files")
    metatrain_speech_files, metatest_speech_files = tg.voxceleb_speech_folder(data_folder=DATA_DIR, train_num=TRAIN_NUM)

    # Step2: init neural networks:
    print("init neural networks")
    relation_network = RelationNetwork(FEATURE_DIM,RELATION_DIM)
    relation_network.apply(weights_init)

    relation_network_optim = torch.optim.Adam(relation_network.parameters(),lr=LEARNING_RATE)
    relation_network_scheduler = StepLR(relation_network_optim,step_size=100000,gamma=0.5)

    if os.path.exists(str("./models/vox_relation_network_"+ str(CLASS_NUM) +"way_" + str(SAMPLE_NUM_PER_CLASS) +"shot.pkl")):
        relation_network.load_state_dict(torch.load(str("./models/vox_relation_network_"+ str(CLASS_NUM) +"way_" + str(SAMPLE_NUM_PER_CLASS) +"shot.pkl")))
        print("load relation network success")

    # Step3: training
    print("Training...")

    last_accuracy = 0.0

    for episode in range(EPISODE):

        relation_network_scheduler.step(episode)

        # init dataset
        # sample_dataloader is to obtain previous samples for compare
        # batch_dataloader is to batch samples for training
        task = tg.VoxFewshotTask(metatrain_speech_files,CLASS_NUM,SAMPLE_NUM_PER_CLASS,BATCH_NUM_PER_CLASS)
        sample_dataloader = tg.get_data_loader(task,num_per_class=SAMPLE_NUM_PER_CLASS,split="train",shuffle=False)
        batch_dataloader = tg.get_data_loader(task,num_per_class=BATCH_NUM_PER_CLASS,split="test",shuffle=True)


        # sample datas
        samples,sample_labels = sample_dataloader.__iter__().next()
        batches,batch_labels = batch_dataloader.__iter__().next()

        # move to GPU

        # calculate relations
        # each batch sample link to every samples to calculate relations
        relations = relation_network(sample=samples, query=batches, num_class=CLASS_NUM)

        # calculate loss
        mse = nn.MSELoss()
        one_hot_labels = torch.zeros(BATCH_NUM_PER_CLASS*CLASS_NUM, CLASS_NUM).scatter_(1, batch_labels.view(-1,1), 1)
        loss = mse(relations,one_hot_labels)


        # training
        relation_network.zero_grad()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(relation_network.parameters(),0.5)

        relation_network_optim.step()

        if (episode+1)%100 == 0:
                print("episode:",episode+1,"loss",loss.data[0])

        if (episode+1)%5000 == 0:

            # test
            print("Testing...")
            total_rewards = 0

            for i in range(TEST_EPISODE):
                
                task = tg.VoxFewshotTask(metatest_speech_files,CLASS_NUM,SAMPLE_NUM_PER_CLASS,SAMPLE_NUM_PER_CLASS)
                sample_dataloader = tg.get_data_loader(task,num_per_class=SAMPLE_NUM_PER_CLASS,split="train",shuffle=False)
                test_dataloader = tg.get_data_loader(task,num_per_class=SAMPLE_NUM_PER_CLASS,split="test",shuffle=True)

                samples,sample_labels = sample_dataloader.__iter__().next()
                tests,test_labels = test_dataloader.__iter__().next()

                # move to GPU

                # calculate relations
                # each batch sample link to every samples to calculate relations
                relations = relation_network(sample=samples, query=tests, num_class=CLASS_NUM)

                _,predict_labels = torch.max(relations.data,1)

                rewards = [1 if predict_labels[j]==test_labels[j] else 0 for j in range(CLASS_NUM*SAMPLE_NUM_PER_CLASS)]

                total_rewards += np.sum(rewards)

            test_accuracy = total_rewards/1.0/CLASS_NUM/SAMPLE_NUM_PER_CLASS/TEST_EPISODE

            print("test accuracy:",test_accuracy)

            if test_accuracy > last_accuracy:

                # save networks
                torch.save(relation_network.state_dict(),str("./models/vox_relation_network_"+ str(CLASS_NUM) +"way_" + str(SAMPLE_NUM_PER_CLASS) +"shot.pkl"))

                print("save networks for episode:",episode)

                last_accuracy = test_accuracy





if __name__ == '__main__':
    main()
























