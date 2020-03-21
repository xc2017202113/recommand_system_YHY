import numpy as np
import torch
import torch.nn as nn


class My_model(nn.Module):
    def __init__(self,itemnums,usernums,item2price_dict):
        super(My_model,self).__init__()
        #init the hyper parameters
        self.embed_size = 50
       # self.q_hidden_num = 64
        self.item2price_dict = item2price_dict
        self.itemnums = itemnums
        self.usernums = usernums

        self.alpha_c_mu = torch.zeros(itemnums,self.embed_size,requires_grad=True)
        self.alpha_c_sigma = torch.ones(itemnums,self.embed_size,requires_grad=True)

        self.rho_c_mu = torch.zeros (itemnums, self.embed_size,requires_grad=True)
        self.rho_c_sigma = torch.ones (itemnums, self.embed_size,requires_grad=True)

        self.lambda_c_mu = torch.zeros (itemnums, self.embed_size,requires_grad=True)
        self.lambda_c_sigma = torch.ones (itemnums, self.embed_size,requires_grad=True)

        self.beta_c_mu = torch.zeros (itemnums, self.embed_size,requires_grad=True)
        self.beta_c_sigma = torch.ones (itemnums, self.embed_size,requires_grad=True)

        self.mu_c_mu = torch.zeros (itemnums, self.embed_size,requires_grad=True)
        self.mu_c_sigma = torch.ones (itemnums, self.embed_size,requires_grad=True)

        self.theta_u_mu = torch.zeros(usernums,self.embed_size,requires_grad=True)
        self.theta_u_sigma = torch.ones(usernums,self.embed_size,requires_grad=True)

        self.gamma_u_mu = torch.zeros (usernums, self.embed_size,requires_grad=True)
        self.gamma_u_sigma = torch.ones (usernums, self.embed_size,requires_grad=True)



        self.sample_vec()
    def adjust_gradient_from_prior(self):
        self.alpha_c_mu += (self.alpha_c-self.alpha_c_mu)
        self.alpha_c_sigma += (self.alpha_c-self.alpha_c_mu)*(self.alpha_c-self.alpha_c_mu)/self.alpha_c_sigma
        self.rho_c_mu += (self.rho_c - self.rho_c_mu)
        self.rho_c_sigma += (self.rho_c - self.rho_c_mu) * (self.rho_c - self.rho_c_mu) / self.rho_c_sigma
        self.lambda_c_mu += (self.lambda_c - self.lambda_c_mu)
        self.lambda_c_sigma += (self.lambda_c - self.lambda_c_mu) * (self.lambda_c - self.lambda_c_mu) / self.lambda_c_sigma
        self.beta_c_mu += (self.beta_c - self.beta_c_mu)
        self.beta_c_sigma += (self.beta_c - self.beta_c_mu) * (self.beta_c - self.beta_c_mu) / self.beta_c_sigma
        self.mu_c_mu += (self.mu_c - self.mu_c_mu)
        self.mu_c_sigma += (self.mu_c - self.mu_c_mu) * (self.mu_c - self.mu_c_mu) / self.mu_c_sigma

        self.theta_u_mu += (self.theta_u - self.theta_u_mu)
        self.theta_u_sigma += (self.theta_u - self.theta_u_mu) * (self.theta_u - self.theta_u_mu) / self.theta_u_sigma

        self.gamma_u_mu += (self.gamma_u - self.gamma_u_mu)
        self.gamma_u_sigma += (self.gamma_u - self.gamma_u_mu) * (self.gamma_u - self.gamma_u_mu) / self.gamma_u_sigma


    #input_format:[userid,target_item,price,[now_basket],[itemset]]
    def forward(self, input):
        #self.sample_vec()
        userid = input[0]
        target_item = input[1]
        price = input[2]
        basket = input[3]
        itemset = input[4]
        result = self.Kesai_function(price,target_item,itemset,userid,self.item2price_dict,basket)
        return result


    #this function computer per user per trip's Kesai_function
    #trip_id start from 0 to compute
    def Kesai_function(self,price,target_item,itemset,customer_id,item2price_dict,basket):
        kesai = self.kesai_function(target_item,customer_id,price)
        interact_with_other_items_sum = torch.zeros(self.embed_size)


        for i in range(len(basket)):
            #here to simple the storage use the itemset represent the baseket
            interact_with_other_items_sum += self.alpha_c[basket[i]]

        k = torch.zeros(self.embed_size)
        if len(basket)>0:
            k = interact_with_other_items_sum/len(basket)
            k = self.rho_c[target_item]*(k)


        interact_with_other_items_sum = torch.mean(k)

        look_forward = 0
        for c in itemset:
            if c not in basket:
                c_kesai = self.kesai_function(c,customer_id,item2price_dict[c])
                step = interact_with_other_items_sum + self.alpha_c[target_item]
                step /= len(basket)+1
                step = torch.mean(self.rho_c[c]*(step))
                if look_forward < step + c_kesai:
                    look_forward = step + c_kesai

        return kesai+interact_with_other_items_sum + look_forward

    def generate_Guassion(self,N,M):
        return torch.tensor(np.random.randn(N,M)).float()

    def sample_vec(self):
        self.lambda_c = self.lambda_c_mu + self.generate_Guassion(self.itemnums,self.embed_size)*self.lambda_c_sigma
        self.rho_c = self.lambda_c_mu + self.generate_Guassion(self.itemnums,self.embed_size)*self.lambda_c_sigma
        self.alpha_c = self.alpha_c_mu + self.generate_Guassion(self.itemnums, self.embed_size) * self.alpha_c_sigma
        self.beta_c = self.beta_c_mu + self.generate_Guassion (self.itemnums, self.embed_size) * self.beta_c_sigma
        self.mu_c = self.mu_c_mu + self.generate_Guassion(self.itemnums,self.embed_size)*self.mu_c_sigma
        self.theta_u = self.theta_u_mu + self.generate_Guassion (self.usernums, self.embed_size) * self.theta_u_sigma
        self.gamma_u = self.gamma_u_mu + self.generate_Guassion (self.usernums, self.embed_size) * self.gamma_u_sigma

    def kesai_function(self,target_item,customer_id,price):


        item_popularity = torch.mean(self.lambda_c[target_item])

        customer_perference = torch.mean(self.theta_u[customer_id]*(self.alpha_c[target_item]))

        price_effects = torch.mean(self.gamma_u[customer_id]*(self.beta_c[target_item])*np.log(price))
        return item_popularity+customer_perference-price_effects


