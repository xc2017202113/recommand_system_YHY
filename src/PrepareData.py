import numpy as np
import random
import copy
class Dataset(object):
    def __init__(self,inputfile):



        self.user2transaction = {}
        self.user2trainsaction_list = []

        self.train_file = inputfile
        self.item_sess_prices_files = "data/item_sess_price.tsv"
        self.itemset = []
        self.item2price = {}
        self.item_encode = {}

        self.init_paras()

        self.basketnum = 8
        self.batchsize = 5 #every batch need 5 users to train

    def init_paras(self):
        session_item2price = {}
        file = open(self.item_sess_prices_files,'r')
        lines = file.readlines()
        file.close()

        item_encode = 0
        for line in lines:
            line = line.strip()
            line_list = line.split('\t')
            itemid = int(line_list[0])
            price = float(line_list[2])

            if itemid not in self.item_encode.keys():

                self.itemset.append(item_encode)
                self.item2price[item_encode] = [price]

                self.item_encode[itemid] = item_encode
                item_encode += 1
            else:
                self.item2price[self.item_encode[itemid]].append(price)
            session_item2price[(int (self.item_encode[itemid]), int (line_list[1]))] = float (line_list[2])
        for key in self.item2price.keys():
            self.item2price[key] = np.mean(self.item2price[key])
        #print(session_item2price)

        self.itemnum = len(self.itemset)

        file  = open(self.train_file,'r')
        lines = file.readlines()
        file.close()
        for line in lines:
            line = line.strip()
            line_list = line.split('\t')
            userid = int(line_list[0])
            itemid = int(line_list[1])
            itemid = self.item_encode[itemid]
            sessionid = int(line_list[2])
            score = float(line_list[3])
            price = session_item2price[(itemid,sessionid)]
            if userid not in self.user2transaction.keys():
                self.user2transaction[userid] = {}
                self.user2transaction[userid][sessionid] = [[itemid,price]]
            else:
                if sessionid not in self.user2transaction[userid]:
                    self.user2transaction[userid][sessionid] = [[itemid, price]]
                else:
                    self.user2transaction[userid][sessionid].append([itemid, price])

        #ingore the sessionid
        sorted(self.user2transaction)#according userid sort
        for key in self.user2transaction.keys():
            self.user2trainsaction_list.append([])
            for i in self.user2transaction[key].keys():
                self.user2trainsaction_list[key-1].append(self.user2transaction[key][i])

            # print(self.user2trainsaction_list[0][0])
            # exit(0)
        self.user2trainsaction_list = np.array(self.user2trainsaction_list)

        self.usernum = len(self.user2trainsaction_list)
        #print(self.user2trainsaction_list)

    #[userid,target_item,price,[now_basket],[itemset]]
    def get_traindataset(self):
        users_list = list(range(250))
        random.shuffle(users_list) #make the users_list unorder
        users_list = np.array(users_list)
        #print(self.user2trainsaction_list[0][0])
        return_list = []
        label_list = []
        for user in users_list:
            #print(user)
            for index,trip in enumerate(self.user2trainsaction_list[user]):
                now_basket = []
                for items in trip:
                    target_item = items[0]
                    price = items[1]
                    basket = self.make_basket(now_basket,target_item)
                    return_list.append([user,target_item,price,now_basket,basket])
                    label_list.append(1.0)

                    #make neg_samples:
                    for neg_sample in basket:
                        if neg_sample not in now_basket and neg_sample != target_item:
                            return_list.append([user,neg_sample,self.item2price[neg_sample],now_basket,basket])
                            label_list.append(0.0)

        #print(len(return_list))
        return return_list,label_list


    def make_basket(self,now_basket,target_item):
        #item = self.itemnum
        basket = copy.copy(now_basket)
        basket.append(target_item)
        #print(basket)
        for i in range(self.basketnum-len(basket)):
            #print(i)
            j = np.random.randint(0,self.itemnum)

            while j in basket:
                j = np.random.randint (0, self.itemnum)
            basket.append(j)

        return basket







# test_dataset = Dataset("data/train.tsv")
# #print(len(test_dataset.get_traindataset()))
# test_dataset.get_traindataset()