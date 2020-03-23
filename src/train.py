import torch
import numpy as np
from Model import My_model
from PrepareData import Dataset
import random
import torch.nn as nn
import torch.nn.functional as F


def train(inputfile):
    dataset = Dataset(inputfile)
    item_num,user_num,item2price_dict = dataset.itemnum,dataset.usernum,dataset.item2price

    batch_size = 128

    print("item:num:{},user_num:{}".format(item_num,user_num))
    model = My_model(item_num,user_num,item2price_dict)

    epoch = 1000
    lr = 1e-3

    # for i in model.parameters():
    #     print(i)
    optimizer = torch.optim.Adam(model.parameters(),lr=lr)
    train_loss_list = []
    all_dataset,labels = dataset.get_traindataset()

    train_valid_rate = 0.2
    train_num = int(len(all_dataset)*(1-train_valid_rate))
    train_dataset = all_dataset[:train_num]
    valid_dataset = all_dataset[train_num:]
    train_labels = labels[:train_num]
    valid_labels = labels[train_num:]
    crossentropyloss = nn.CrossEntropyLoss ()

    train_dataset_list = list(range(len(train_dataset)))
    for i in range(epoch):
        print("epoch:",i)
        random.shuffle(train_dataset_list)
        train_loss = torch.tensor ([0.0])
        #print(train_loss.type())
        for index,j in enumerate(train_dataset_list):

            output = torch.sigmoid(model(train_dataset[j]))
            #print(output.type())
            if index % 10 == 0 and index > 100:
                print("batch: %d loss: %.2f"%(index,np.mean(train_loss_list)))

                #train_loss += crossentropyloss(output,train_labels[j])
                #print(output)
            real_prob = output.unsqueeze(0).unsqueeze(0)
            feak_prob = (torch.tensor(1.0)-output).unsqueeze(0).unsqueeze(0)
            out = torch.cat((feak_prob,real_prob),dim=1)

            label = torch.tensor([train_labels[j]]).long()
            Loss = crossentropyloss(out,label).unsqueeze(0)
            train_loss = torch.cat((train_loss,Loss),dim=0)
            #print(train_loss.detach_().numpy())


            if index % batch_size == 0 and index!=0:
                train_loss = torch.mean(train_loss)
                #print("================")
                #print(train_loss)
                train_loss_list.append(train_loss.detach().numpy())
                #print('train_loss',train_loss.detach().numpy())
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()
                #model.adjust_gradient_from_prior()

            if index % batch_size == 0:
                train_loss = torch.tensor ([0.0])

        #valid:
        hit = 0.0
        for index,j in enumerate(valid_dataset):
            valid_out = torch.sigmoid(model(j)).detach().numpy()
            if valid_out > 0.5 and valid_labels[index] == 1:
                hit += 1
            elif valid_out < 0.5 and valid_labels[index] == 0:
                hit += 1

        acc = hit/len(valid_dataset)
        print("acc:%2f"%(acc))


if __name__ == "__main__":
    train("data/train.tsv")