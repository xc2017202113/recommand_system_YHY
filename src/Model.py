import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random


class My_model(nn.Module):
    def __init__(self,itemnums,usernums,item2price_dict):
        super(My_model,self).__init__()
        #init the hyper parameters
        self.embed_size = 100
        self.q_hidden_num = 64
        self.item2price_dict = item2price_dict



        self.sigma_list = torch.tensor([1.0,1.0,1.0,0.1,0.1,1.0,1.0],requires_grad=True)
        self.mu_list  = torch.tensor([0.0,0.0,0.0,0.0,0.0,0.0,0.0],requires_grad=True)

        # self.alpha_c = torch.tensor(np.random.randn(itemnums,self.embed_size))
        # self.rho_c = torch.tensor(np.random.randn(itemnums,self.embed_size))
        # self.lambda_c = torch.tensor(np.random.randn(itemnums,self.embed_size))
        # self.beta_c = torch.tensor(np.random.randn(itemnums,self.embed_size))
        # self.mu_c = torch.tensor(0.1*np.random.randn(itemnums,self.embed_size))
        #
        # self.theta_u = torch.tensor(np.random.randn(usernums,self.embed_size))
        # self.gamma_u = torch.tensor(np.random.randn(usernums,self.embed_size))

        self.q_hidden_layer = nn.Linear(self.embed_size,self.q_hidden_num)
        self.q_output_layer = nn.Linear(self.q_hidden_num,1)

    def sample_vectors(self,itemnums,usernums):
        self.alpha_c = torch.tensor (self.sigma_list[0]*torch.tensor(np.random.randn (itemnums, self.embed_size))+self.mu_list[0],requires_grad=False).float()
        self.rho_c = torch.tensor (self.sigma_list[1]*torch.tensor(np.random.randn (itemnums, self.embed_size))+self.mu_list[1],requires_grad=False).float()
        self.lambda_c = torch.tensor (self.sigma_list[2]*torch.tensor(np.random.randn (itemnums, self.embed_size))+self.mu_list[2],requires_grad=False).float()
        self.beta_c = torch.tensor (self.sigma_list[3]*torch.tensor(np.random.randn (itemnums, self.embed_size))+self.mu_list[3],requires_grad=False).float()
        self.mu_c = torch.tensor (self.sigma_list[4]* torch.tensor(np.random.randn (itemnums, self.embed_size))+self.mu_list[4],requires_grad=False).float()

        self.theta_u = torch.tensor (self.sigma_list[5]*torch.tensor(np.random.randn (usernums, self.embed_size))+self.mu_list[5],requires_grad=False).float()
        self.gamma_u = torch.tensor (self.sigma_list[6]*torch.tensor(np.random.randn (usernums, self.embed_size))+self.mu_list[6],requires_grad=False).float()


    def Q_function(self,input):
        hidden = F.relu(self.q_hidden_layer(input))
        hidden = F.dropout(hidden)

        out =  F.sigmoid(self.q_output_layer(hidden))
        return out

    def get_gaussion_prior(self,sigma,mu,z):
        gradient = -(z-mu)/sigma
        return 0.5*(z-mu)*gradient

    def get_vectors_gaussion_prior(self,vectors,paras_index):
        #vector_shape (1,embedding_size)
        vectors = vectors.reshape(self.embed_size)
        p_l = torch.tensor(0.0).float()
        for i in vectors:
            p_l += self.get_gaussion_prior(self.sigma_list[paras_index],self.mu_list[paras_index],i)

        return p_l
    def get_f_prior(self,itemset,userid):
        q_l = torch.tensor(0.0).float()
        p_l = torch.tensor(0.0).float()
        #print(itemset)
        for i in itemset:
            #print(itemset)

            # print(self.alpha_c[0].shape)
            # exit(0)
            #print("----------",i)
            q_l += torch.mean(self.Q_function(self.alpha_c[i]))
            p_l += self.get_vectors_gaussion_prior(self.alpha_c[i],0)
            q_l += torch.mean (self.Q_function (self.rho_c[i]))
            p_l += self.get_vectors_gaussion_prior (self.rho_c[i], 1)
            q_l += torch.mean (self.Q_function (self.lambda_c[i]))
            p_l += self.get_vectors_gaussion_prior (self.lambda_c[i], 2)
            q_l += torch.mean (self.Q_function (self.beta_c[i]))
            p_l += self.get_vectors_gaussion_prior (self.beta_c[i], 3)
            q_l += torch.mean (self.Q_function (self.mu_c[i]))
            p_l += self.get_vectors_gaussion_prior (self.mu_c[i], 4)

        #print("-----------")
        q_l += torch.mean(self.Q_function(self.theta_u[userid]))
        q_l += torch.mean(self.Q_function(self.gamma_u[userid]))
        p_l += self.get_vectors_gaussion_prior (self.theta_u[userid], 5)
        p_l += self.get_vectors_gaussion_prior (self.gamma_u[userid], 6)
        #print('-----------------')
        return p_l-q_l


    #this funtion need the input:per user [[trip1],[trip2],[trip3]]
    def evaluate_f_function(self,input,itemset,userid):
        # print(input[0])
        # print(itemset)
        # exit(0)
        f = self.get_f_prior(itemset,userid)
        #print(f)
        #exit(0)
        for index,trips in enumerate(input):
            basket_now = []
            for items in trips:
                item_id = items[0]
                price = items[1]
                basket_now.append(item_id)
                for c in itemset:
                    if c not in basket_now:
                        temp_baseket = basket_now
                        temp_baseket.append(c)
                        f += torch.log(torch.sigmoid(self.Kesai_function(price,item_id,itemset,userid,self.item2price_dict,basket_now)-\
                                                     self.Kesai_function(self.item2price_dict[c],c,itemset,userid,self.item2price_dict,temp_baseket)))

        return f

    def forward(self, input,itemset,userid,batch_size):
        loss = torch.tensor(0.0)
        for batch in range(batch_size):
            loss += self.evaluate_f_function(input[batch],itemset[batch],userid[batch])
        return torch.mean(loss)

    #this function computer per user per trip's Kesai_function
    #trip_id start from 0 to compute
    def Kesai_function(self,price,target_item,itemset,customer_id,item2price_dict,basket):
        kesai = self.kesai_function(target_item,customer_id,price)
        interact_with_other_items_sum = torch.zeros(self.embed_size)

        for i in range(len(basket)):
            #here to simple the storage use the itemset represent the baseket
            interact_with_other_items_sum += self.alpha_c[basket[i]]

        interact_with_other_items = interact_with_other_items_sum/len(basket)
        interact_with_other_items = torch.mean(self.rho_c[target_item]*(interact_with_other_items))

        look_forward = 0
        for c in itemset:
            if c not in basket:
                c_kesai = self.kesai_function(c,customer_id,item2price_dict[c])
                step = interact_with_other_items_sum + self.alpha_c[target_item]
                step /= len(basket)+1
                step = torch.mean(self.rho_c[c]*(step))
                if look_forward < step + c_kesai:
                    look_forward = step + c_kesai

        return kesai+interact_with_other_items + look_forward



    def kesai_function(self,target_item,customer_id,price):
        item_popularity = torch.mean(self.lambda_c[target_item])
        customer_perference = torch.mean(self.theta_u[customer_id]*(self.alpha_c[target_item]))
        price_effects = torch.mean(self.gamma_u[customer_id]*(self.beta_c[target_item])*np.log(price))
        return item_popularity+customer_perference-price_effects


