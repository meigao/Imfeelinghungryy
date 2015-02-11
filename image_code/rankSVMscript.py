# -*- coding: utf-8 -*-
"""
Created on Wed Jan 21 22:44:21 2015

@author:
"""

import numpy as np
import scipy.io
from subprocess import call
import glob
import os

def arraylevels(x):
    sortidx = np.argsort(x);
    y = x[sortidx]
    curElem = y[0];
    curOrder = 1;
    levels = np.zeros(len(x))
    for idx, ele in enumerate(y):
        if curElem != ele:
            curElem = ele;
            curOrder = curOrder+1
        
        levels[idx] = curOrder

    invidx = np.arange(0,len(x));
    invidx[sortidx] = np.arange(0,len(x))
    return levels[invidx]
    
def savefeatureFile(path, feat, y = []):
    if len(y) == 0:
        print('save testing data')
    else:
        print('save training data')
        
    with open(path +'rankFeature.txt', 'wb') as rankFile:
        for i in range(0, len(feat)):
            # write label
            if np.mod(i, 100) == 0:
                print('processing image # ' + str(i) + '/' + str(len(feat)))
            
            if len(y) == 0:
                rankFile.write(str(1) + ' ')
            else:
                rankFile.write(str(y[i]) + ' ')
                
            for j in range(0, len(feat[i])-1):
                rankFile.write(str(j+1) + ':' + str(feat[i][j]) + ' ')
            
            rankFile.write(str(len(feat[i])) + ':' + str(feat[i][-1]) + '\n')
    rankFile.close()
    print('finished...')
    
def trainmodel(train_path, nfold = 0):
    # train rank-SVM model
    # @train_path: path of training feature
    # @nfold n-fold cross validation 
    # if nfold < 2, crosss validation will be skiped
    print('Training model...')
    svmDir = './liblinear-ranksvm-1.94/'
    if nfold > 2:
        print('cross validation with n-fold = ' + str(nfold))
        cmd = [svmDir + './train', '-s', '8','-c','0.01','-B','1','-v', 
               str(nfold), train_path +'rankFeature.txt']
    else:
        print('no cross validation')
        cmd = [svmDir + './train', '-s', '8','-c','0.01','-B','1', 
               train_path +'rankFeature.txt']
    call(cmd)
    cmd = ['mv', 'rankFeature.txt.model',train_path + '.']
    call(cmd)
    
def processCategory(category):
    # cateogry of burger
    key_words = [['american', 'burgers', 'hot dog', 'fast food', 'donut'],
                 ['grill', 'barbecue'],
                 ['chinese foods', 'sichuan', 'cocktail'],
                 ['dessert', 'ice cream'],
                 ['pasta'],
                 ['pizza'],
                 ['janpanese', 'sushi'],
                 ['wine', 'alcohol', 'bar']
                 ]
    replaces = ['burger', 'bbq', 'chinese', 'ice cream', 'pasta', 'pizza', 'sushi', 'cocktail']
    for idx, key_word in enumerate(key_words):
        for word in key_word:
            if category in word or word in category:
                category = replaces[idx]
            
    return category
    
def cpname(test_path, recommend_path):
    cmd = ['cp', test_path+'name.txt', recommend_path+'.']
    call(cmd)
    
def recommend(test_path, recommend_path='', top_num = 5, train_path = ''):
    svmDir = './liblinear-ranksvm-1.94/'
    if len(train_path) == 0:
        train_path = './data/'
    all_categories = [f for f in os.listdir(train_path) if os.path.isdir(train_path + f)]
    
    # assign model
    with open(test_path +'name.txt', 'r') as restfile:
        line = restfile.readline()
        num_category = int(restfile.readline())
        tag_categories = []
        for i in range(0, num_category):
            line = restfile.readline();
            category = processCategory(line[0:-1].lower())
            tag_categories.append(category)
        
        train_path = './data/burger/'
        if any(x in all_categories for x in tag_categories):
            print('classified as category ' + category)
            category = [x for x in tag_categories if x in all_categories][0]
            train_path = './data/' + category + '/'
        else:
            print('use default burger')
            
    restfile.close()
        
    
    cmd = [svmDir + './predict', test_path +'rankFeature.txt', 
    train_path +'rankFeature.txt.model', test_path+'output.txt']
    call(cmd)
    predicted = np.loadtxt(test_path +'output.txt')
    sortedres = (-predicted).argsort()
    if np.isscalar(sortedres):
        recommend_idx = np.zeros((1,1), dtype = int)[0]
    else:
        recommend_idx = sortedres[0:top_num]
        
    if recommend_path == '':
        recommend_path = test_path + "recommend/"
    if os.path.isdir(recommend_path) == False:
        print('Create directory ' + recommend_path)
        cmd = ['mkdir', recommend_path]
        call(cmd)
    else:
        cmd = ['rm', '-f',recommend_path+'*.jpg']
        call(cmd)
        cmd = ['rm', '-f',recommend_path+'*.png']
        call(cmd)
        
    # dir all images
    img_list = glob.glob(test_path +"*.jpg")
    if len(img_list) == 0:
        img_list = glob.glob(test_path +"*.png")
    if len(img_list) != 0:
        for idx in recommend_idx:
            cmd = ['cp', img_list[idx], recommend_path+'.']
            call(cmd)
            
            
def recommend_all(rests_path, recmds_path):
    if not os.path.exists(recmds_path):
        os.makedirs(recmds_path)

    rest_list = [f for f in os.listdir(rests_path) if os.path.isdir(rests_path + f)]
    num_rest = len(rest_list)
    for idx, rest_id in enumerate(rest_list):
        print('recommend image for restaurant id ' + rest_id + ', ' + str(idx) + '/' + str(num_rest))
        recmd_path = recmds_path + rest_id + '/'
        if not os.path.exists(recmd_path):
            os.makedirs(recmd_path)
        
        test_path = rests_path + rest_id + '/'
        cpname(test_path, recmd_path)
        matfeat = scipy.io.loadmat(test_path + 'data.mat')
        test_feat = matfeat['feat']
        savefeatureFile(test_path, test_feat)
        recommend(test_path, recmd_path, 1)
        
def train_all(data_path):
    neg_path = data_path + 'neg/'
    categories = [f for f in os.listdir(data_path) if os.path.isdir(data_path + f)]
    for category in categories:
        if category == 'neg' or category == 'burger' or category == 'bbq' or category == 'chinese':
            continue
        
        print('train model category ' + category)
        train_path = data_path + category + '/'
        matfeat = scipy.io.loadmat(train_path + 'data.mat')
        pos_feat = matfeat['feat']
        
        matfeat = scipy.io.loadmat(neg_path + 'data.mat')
        neg_feat = matfeat['feat']
        
        
        # y label for positive and negative
        view = np.loadtxt(train_path + 'label.txt')
        pos_y = arraylevels(view)
        
        if len(pos_y) != len(pos_feat):
            print('trim training')
            pos_feat = pos_feat[0:len(pos_y)]
        
        neg_y = -np.ones((1,len(neg_feat)))[0]
        
        train_feat = np.vstack((pos_feat, neg_feat))
        train_y = np.concatenate((pos_y, neg_y), axis = 1)
        
        # write feature to file for training
        savefeatureFile(train_path, train_feat, train_y)
        
        trainmodel(train_path, 0)
        

#script train all models
train_all('./data/')

rests_path = './restaurants/'
recmds_path = './recommends/'

recommend_all(rests_path, recmds_path)

## train rank-svm
train_path = './data/burger/'
neg_path = './data/neg/'

# feature for positive and negative
matfeat = scipy.io.loadmat(train_path + 'data.mat')
pos_feat = matfeat['feat']

matfeat = scipy.io.loadmat(neg_path + 'data.mat')
neg_feat = matfeat['feat']

train_feat = np.vstack((pos_feat, neg_feat))

# y label for positive and negative
view = np.loadtxt(train_path + 'label.txt')
pos_y = arraylevels(view)

neg_y = -np.ones((1,len(neg_feat)))[0]

train_y = np.concatenate((pos_y, neg_y), axis = 1)

# write feature to file for training
savefeatureFile(train_path, train_feat, train_y)

# cross validation
trainmodel(train_path, 3)

# train model
trainmodel(train_path, 0)

# test rank-svm
test_path = './testdata/burger/'

matfeat = scipy.io.loadmat(test_path + 'data.mat')
test_feat = matfeat['feat']

savefeatureFile(test_path, test_feat)
recommend(test_path)