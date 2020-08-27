# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 13:35:13 2019

@author: chf
"""

import numpy as np
'''
    初始化数据集
    Ann 表示状态转移概率矩阵，
    Bnm 表示观测概率矩阵，
    Pi 表示初始状态概率向量
    O是对应的观测序列
    N 表示可能的状态数
    M 表示可能的观测数
'''
class HMM:
    def __init__(self, Ann, Bnm, Pi, O):
        self.A = np.array(Ann, np.float)
        self.B = np.array(Bnm, np.float)
        self.Pi = np.array(Pi, np.float)
        self.O = np.array(O)
        self.N = self.A.shape[0]             #N = A.shape[0]为数组A的行数， 
        self.M = self.B.shape[1]             #M = B.shape[1]为数组O的列数

    def viterbi(self):
    # given O,lambda .finding I

        T = self.O.shape[1]
        I = np.zeros((1, T), np.int16)
        
        delta = np.zeros((T, self.N), np.float)  
        psi = np.zeros((T, self.N), np.float)
        
        
        for i in range(self.N):
            delta[0, i] = self.Pi[0, i] * self.B[i, O[0, 0]-1]
            psi[0, i] = 0
        
        for t in range(1, T):
            for i in range(self.N):
                delta[t, i] = self.B[i, O[0, t]-1] *np.array( [delta[t-1,j] * self.A[j,i]  
                    for j in range(self.N)] ).max()
                psi[t,i] = np.array( [delta[t-1,j] * self.A[j,i] 
                    for j in range(self.N)] ).argmax()
        print("Delta矩阵: \n %r" % delta)
        print("Psi矩阵: \n %r" % psi)
        
        P_best = delta[T-1, :].max()
#        print(P_T)
        I[0, T-1] = delta[T-1, :].argmax()
#
        for t in range(T-2, -1, -1):
            I[0, t] = psi[t+1, I[0, t+1]]
        print("最优路径概率: \n %r" % P_best)
        print("最优路径: \n %r" % I)
        return I
    
    def forward(self):
        T = self.O.shape[1]
        alpha = np.zeros((T, self.N), np.float)

        for i in range(self.N):        
            alpha[0,i] = self.Pi[0, i] * self.B[i, self.O[0, 0] - 1]

        for t in range(T-1):
            for i in range(self.N):
                summation = 0   # for every i 'summation' should reset to '0'
                for j in range(self.N):
                    summation += alpha[t,j] * self.A[j,i]
                alpha[t+1, i] = summation * self.B[i, self.O[0, t+1] - 1]

        summation = 0.0
        for i in range(self.N):
            summation += alpha[T-1, i]
        Polambda = summation
        print("Alpha取值: \n %r" % alpha)
        print("P(O|λ): \n %r" % Polambda)
        return Polambda,alpha
        
    def backward(self):
        T = self.O.shape[1]
        beta = np.zeros((T, self.N), np.float)
        for i in range(self.N):
            beta[T-1, i] = 1.0

        for t in range(T-2,-1,-1):
            for i in range(self.N):
                summation = 0.0     # for every i 'summation' should reset to '0'
                for j in range(self.N):
                    summation += self.A[i,j] * self.B[j, self.O[0, t+1] - 1] * beta[t+1,j]
                beta[t,i] = summation

        Polambda = 0.0
        for i in range(self.N):
            Polambda += self.Pi[0, i] * self.B[i, self.O[0, 0]] * beta[0, i]
        print("Beta: \n %r" % beta)
        print("P(O|λ): \n %r" % Polambda)
        return Polambda, beta
    
    def compute_gamma(self,alpha,beta):
        T = self.O.shape[1]
        gamma = np.zeros((T, self.N), np.float)       # the probability of Ot=q
        for t in range(T):
            for i in range(self.N):
                gamma[t, i] = alpha[t,i] * beta[t,i] / sum(
                    alpha[t,j] * beta[t,j] for j in range(self.N) )
        print("Gamma: \n %r" % gamma)
        return gamma
    
    def compute_xi(self,alpha,beta):
        T = self.O.shape[1]
        xi = np.zeros((T-1, self.N, self.N), np.float)  # note that: not T
        for t in range(T-1):   # note: not T
            for i in range(self.N):
                for j in range(self.N):
                    numerator = alpha[t,i] * self.A[i,j] * self.B[j,self.O[0, t+1] -1] * beta[t+1,j]
                    # the multiply term below should not be replaced by 'nummerator'，
                    # since the 'i,j' in 'numerator' are fixed.
                    # In addition, should not use 'i,j' below, to avoid error and confusion.
                    denominator = sum( sum(     
                        alpha[t,i1] * self.A[i1,j1] * self.B[j1,self.O[0, t+1] -1] * beta[t+1,j1] 
                        for j1 in range(self.N) )   # the second sum
                            for i1 in range(self.N) )    # the first sum
                    xi[t,i,j] = numerator / denominator
        print("ξ: \n %r" % xi)
        return xi
    
#输入格式如下：
A = np.array([[.5,.2,.3],[.3,.5,.2],[.2,.3,.5]])
B = np.array([[.5,.5],[.4,.6],[.7,.3]])
Pi = np.array([[.2,.4,.4]])
O = np.array([[1,2,1]])

Hmm = HMM(A, B, Pi, O)    
Hmm.viterbi()
Polambda1,alpha = Hmm.forward()
Polambda2, beta = Hmm.backward()
Hmm.compute_gamma(alpha,beta)
Hmm.compute_xi(alpha,beta)



#
#viterbi(Hmm)

    