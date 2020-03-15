import torch
import numpy as np
from Model import My_model
from PrepareData import Dataset


def train(inputfile):
    dataset = Dataset(inputfile)
    item_num,user_num,item2price_dict = dataset.itemnum,dataset.usernum,dataset.item2price
    batch_size = dataset.batchsize
    print("item:num:{},user_num:{}".format(item_num,user_num))
    model = My_model(item_num,user_num,item2price_dict)
    epoch = 1000
    lr = 1e-3

    # for i in model.parameters():
    #     print(i)
    optimizer = torch.optim.Adam(model.parameters(),lr=lr)
    train_loss_list = []
    for i in range(epoch):
        print("epoch:",i)
        train_transaction,train_itemset,train_userid = dataset.get_traindataset()
        print("======")
        batches = len(train_transaction)
        #loss = []
        for batch in range(batches):
            model.sample_vectors(item_num,user_num)

            #for index in range(batch_size):
            loss = model(train_transaction[batch],train_itemset[batch],train_userid[batch],batch_size)

            #loss = torch.tensor(loss)
            if batch % 1 == 0:
                print('loss:{},batch:{}'.format(loss,batch))
            #loss = torch.mean(torch.tensor(loss))
            train_loss = -torch.mean(loss)
            train_loss_list.append(train_loss)
            #print(train_loss)
            print('train_loss',train_loss.detach().numpy())
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()


if __name__ == "__main__":
    train("data/train.tsv")