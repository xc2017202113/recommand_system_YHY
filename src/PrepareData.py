import numpy as np
import random
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
    def get_traindataset(self):
        users_list = list(range(250))
        random.shuffle(users_list) #make the users_list unorder
        users_list = np.array(users_list)
        return_list = []
        return_items_list = []
        return_user_id_list = []
        for index,userid in enumerate(users_list):

            if index % self.batchsize == 0:
                return_list.append([])
                return_items_list.append ([])
                return_user_id_list.append([])

            return_list[int(index/self.batchsize)].append(self.user2trainsaction_list[userid])
            return_user_id_list[int(index/self.batchsize)].append(userid)
            userid2itemset = []
            for index2,trips in enumerate(self.user2trainsaction_list[userid]):
                for items_price in trips:
                    if items_price[0] not in userid2itemset:
                        userid2itemset.append(items_price[0])

                    if len(userid2itemset) >= self.basketnum:
                        userid2itemset = userid2itemset[0:self.basketnum]
                    else:
                        for i in range(len(userid2itemset),self.basketnum):
                            randomset =  int(random.randint(0,len(self.itemset)-1))
                            # print(randomset)
                            # print(userid2itemset)
                            # print(self.itemset)
                            while randomset in userid2itemset:#find other items not in the basket
                                randomset = int(random.randint (0, len (self.itemset)-1))
                            userid2itemset.append(randomset)


            return_items_list[int(index/self.batchsize)].append(userid2itemset)



        return_list = np.array(return_list)
        return_items_list = np.array(return_items_list)
        return_user_id_list = np.array(return_user_id_list)
        # print(return_list[0][0][0])
        # print(return_user_id_list[0])
        # exit(0)
        return return_list,return_items_list,return_user_id_list



#test_dataset = Dataset("data/train.tsv")
#print(len(test_dataset.get_traindataset()))