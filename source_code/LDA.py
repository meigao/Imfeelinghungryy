# -*- coding: utf-8 -*-
"""
Created on Mon Jan 19 10:24:23 2015

@author: gaomei
"""

import json
import pandas as pd
import numpy as np
import csv as csv
import sqlite3
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.lancaster import LancasterStemmer
import operator
import pickle
from subprocess import call
from gensim import corpora, models, similarities
from itertools import chain
import nltk
from nltk.corpus import stopwords
from operator import itemgetter
import re
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
from operator import itemgetter
from sklearn.metrics import classification_report
import csv
import os
import matplotlib.pyplot as plt
import pandas as pd
os.chdir('/Users/gaomei/Desktop/Insight_Demo/')


db = sqlite3.connect('/Users/gaomei/Desktop/Yelp_project/yelp.db')
c = db.cursor()
c.execute('select distinct business.business_id, review.reviews from business, review where business.business_id=review.business_id')
count=0
review_list=[]
for row in c:
    count+=1
    print count
    review_list.append(row[1])
    
pickle.dump(review_list, open('review_list.p', "wb"))
review_list = pickle.load(open('review_list.p', "rb"))

c.execute('select distinct business.business_id, review.reviews from business, review where business.business_id=review.business_id')
count=0
business_review={}
for row in c:
    count+=1
    print count
    if row[0] not in business_review:
        business_review[row[0]]=[]
        business_review[row[0]].append(row[1])
    else:
        business_review[row[0]].append(row[1])
        

    
    
punt_list=['.',',','!','?','(',')',':','<','>','/','&','-','[',']','"','|',';','%','$','=','+','*']    
documents=review_list
stoplist = stopwords.words('english')
mystoplist=stoplist+['a','b','c','d','e','f','g','h','j','k','l','m','o','p','q','r','s','t','u','x','y','z','n','oz','my','the','st','and','of','is','in','that','this','to','was','you','on','they','were','are','their','ve','us','an','ll','it','for','with','would']
document_dic={}
count=0
for document in documents:
    s=list(document.lower())
    ts=''.join([ o for o in s if not o in  punt_list ]).split()
    for word in ts:
        if word not in mystoplist:
            if word.isdigit()==False:
                if "'s" not in word:
                    if word not in document_dic:
                        document_dic[word]=1
                    else:
                        document_dic[word]+=1  
    count+=1
    print count

pickle.dump(document_dic, open('document_dic.p', "wb"))
document_dic = pickle.load(open('document_dic.p', "rb"))

sorted_dic = sorted(document_dic.items(), key=operator.itemgetter(1),reverse=True)[0:10000]  
myworddic={}
for word in sorted_dic:
    myworddic[word[0]]=word[1]

texts=[]
count=0    
for document in documents:
    tmp_doc=[]
    s=list(document.lower())
    ts=''.join([ o for o in s if not o in  punt_list ]).split()
    for word in ts:
        if word in myworddic:
            tmp_doc.append(word)
    count+=1
    print count
    texts.append(tmp_doc)

#to calculate the proportion of each topic in the whole corpus    
mytext=[]
count=0
for document in documents:
    s=list(document.lower())
    ts=''.join([ o for o in s if not o in  punt_list ]).split()
    for word in ts:
        if word in myworddic:
            mytext.append(word)
    count+=1
    print count

    
pickle.dump(texts, open('texts.p', "wb"))
texts = pickle.load(open('texts.p', "rb"))

pickle.dump(mytext, open('mytext.p', "wb"))
mytext = pickle.load(open('mytext.p', "rb"))
    
dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]

tfidf = models.TfidfModel(corpus) 
corpus_tfidf = tfidf[corpus]    

import logging 
import logging.handlers
logger = logging.getLogger('')
logger.setLevel(logging.INFO)
formatter = logging.Formatter(
    '%(filename)s[%(process)d]: %(levelname)s %(message)s')
fh = logging.FileHandler('log20.txt')
fh.setLevel(logging.INFO)
fh.setFormatter(formatter)
logger.addHandler(fh)
topic_num=[1,5,10,15,20,25,30,35,40,45,50,100,200]
n_topics = 200
lda = models.LdaModel(corpus_tfidf, id2word=dictionary, num_topics=n_topics, passes=5)
lda = models.ldamulticore.LdaMulticore(corpus_tfidf, id2word=dictionary, num_topics=n_topics, passes=5)

chunk=corpus[0:1000]
print lda.log_perplexity(chunk, total_docs=None)

perplexity=[]
for i in topic_num:
    print 'now processing        '+str(i)
    tmp_lda=models.LdaModel(corpus_tfidf, id2word=dictionary, num_topics=i, passes=5)
    perplexity.append(tmp_lda.log_perplexity(chunk, total_docs=None))

pickle.dump(perplexity, open('perplexity.p', "wb"))
perplexity = pickle.load(open('perplexity.p', "rb"))    
    
pickle.dump(lda, open('lda.p', "wb"))
lda = pickle.load(open('lda.p', "rb"))

for i in range(0, n_topics):
    temp = lda.show_topic(i, 40)
    terms = []
    for term in temp:
        terms.append(term[1])
    print "Top 10 terms for topic #" + str(i) + ": "+ ", ".join(terms)

count=0
business_review_clean={}    
for item in business_review:
    business_review_clean[item]=[]
    count+=1
    print count
    for review in business_review[item]:
        s=list(review.lower())
        ts=''.join([ o for o in s if not o in  punt_list ]).split()
        for word in ts:
            if word in myworddic:
                business_review_clean[item].append(word)
#extract the bow features 
count=0
business_review_bow={}
for item in business_review_clean:
    count+=1
    print count
    business_review_bow[item]=tfidf[dictionary.doc2bow(business_review_clean[item])]  


             
pickle.dump(business_review_clean, open('business_review_clean.p', "wb"))
business_review_clean = pickle.load(open('business_review_clean.p', "rb"))

business_review_lda={}
count=0
for item in business_review_clean:
    doc_lda = lda[dictionary.doc2bow(business_review_clean[item])]
    business_review_lda[item]=doc_lda
    count+=1
    print count

#to calculate the proportion of each topic in the whole corpus  
whole_lda=lda[dictionary.doc2bow(mytext)]

c.execute('select distinct business.business_id, business_stars from business')
count=0
business_star={}
for row in c:
    business_star[row[0]]=row[1]
    count+=1
    print count


Y=[]
X=[]
XX=[]
business_review_lda_new={}
for business in business_star:
    Y.append(business_star[business])
    tmp1=business_review_lda[business]
    tmp2=[0]*15
    for item in tmp1:
        tmp2[item[0]]=item[1]
    X.append(tmp2)
    business_review_lda_new[business]=tmp2
    tmp3=business_review_bow[business]
    tmp4=[0]*10000
    for thing in tmp3:
        tmp4[thing[0]]=thing[1]
    XX.append(tmp4)
YY=[]
for item in Y:
    if item>3.5:
        YY.append(1)
    else:
        YY.append(-1)
        
Y=np.array(Y)
YY=np.array(YY)
X=np.array(X).astype(float)
XX=np.array(XX).astype(float)
#classification using logistic regression
from sklearn.linear_model import LogisticRegression
#lda feature
count=0
logistic_lda_accuracy=[]
for i in range(1,10):
    count+=1
    print count
    maxIter = 10
    accuracy = np.ones(maxIter)
    logisticclf = LogisticRegression(C=0.05,penalty='l2')
    for iter in range(0,maxIter):
        print iter
        trainIdx = np.random.choice(len(X), np.floor(len(X)*0.1*i), replace = False).astype(int)
        testIdx = np.ones(len(X), dtype = bool)
        testIdx[trainIdx] = False;
        testIdx=testIdx[0:len(XX)*0.1*i]
        logistic_classifier = logisticclf.fit(X[trainIdx,:], YY[trainIdx])
        y_logistic_predicted = logistic_classifier.predict(X[testIdx,:])
        accuracy[iter] = metrics.precision_score(YY[testIdx], y_logistic_predicted)
    logistic_lda_accuracy.append(accuracy.mean())    
   
pickle.dump(logistic_lda_accuracy, open('logistic_lda_accuracy.p', "wb"))
logistic_lda_accuracy = pickle.load(open('logistic_lda_accuracy.p', "rb"))
#bow feature
count=0
logistic_bow_accuracy=[]
for i in range(1,10):
    count+=1
    print count
    maxIter = 10
    accuracy = np.ones(maxIter)
    logisticclf = LogisticRegression(C=5,penalty='l2')
    for iter in range(0,maxIter):
        print iter
        trainIdx = np.random.choice(len(XX), np.floor(len(XX)*0.1*i), replace = False).astype(int)
        testIdx = np.ones(len(XX), dtype = bool)
        testIdx[trainIdx] = False;
        testIdx=testIdx[0:len(XX)*0.1*i]
        logistic_classifier = logisticclf.fit(XX[trainIdx,:], YY[trainIdx])
        y_logistic_predicted = logistic_classifier.predict(XX[testIdx,:])
        accuracy[iter] = metrics.precision_score(YY[testIdx], y_logistic_predicted)
    logistic_bow_accuracy.append(accuracy.mean())      

pickle.dump(logistic_bow_accuracy, open('logistic_bow_accuracy.p', "wb"))
logistic_bow_accuracy = pickle.load(open('logistic_bow_accuracy.p', "rb"))


#plot the figure to compare lda and bow 
import matplotlib.patches as mpatches

x=range(3,10)
x=np.array(x)
x=x*0.1
plt.ylim(0.5,0.9)
plt.ylabel('classification accuracy',fontsize=14)
plt.xlabel('proportion of data used for training',fontsize=14)
plt.plot(x,logistic_bow_accuracy[2:10],'r^--',x,logistic_lda_accuracy[2:10],'bs--')
red_patch = mpatches.Patch(color='red', label='BOW')
blue_patch = mpatches.Patch(color='blue', label='LDA')

plt.legend(handles=[red_patch,blue_patch],loc=4)



logisticclf = LogisticRegression(C=0.05,penalty='l2')
logistic_classifier=logisticclf.fit(X, YY)
y_logistic_predicted = logistic_classifier.predict(X)
metrics.precision_score(YY, y_logistic_predicted)

from sklearn import preprocessing
logistic_classifier.coef_
min_max_scaler = preprocessing.MinMaxScaler()
coefficient_scale = min_max_scaler.fit_transform(logistic_classifier.coef_.transpose())
mysum=np.sum(coefficient_scale)
coefficient_scale=coefficient_scale/mysum

#calculate the score for each business
results={}
result_name={}
result_score={}
count=0
for i in range(2**15):
    count+=1
    print count
    results[i]=[]
    result_name[i]=[]
    result_score[i]=[]
    binary='{0:015b}'.format(i)
    tmp1=[0]*15
    for idx,item in enumerate(binary):
        tmp1[idx]=int(item)
    tmp1=tmp1*coefficient_scale.transpose()
    tmp_score={}
    for business in business_review_lda_new:
        tmp2=np.array(business_review_lda_new[business])
        tmp_score[business]=np.dot(tmp2,tmp1[0])
    sorted_business = sorted(tmp_score, key=tmp_score.get,reverse=True)[0:10]
    for item in sorted_business:
        results[i].append(item)
        results[i].append(tmp_score[item])
        result_name[i].append(item)
        result_score[i].append(tmp_score[item])

pickle.dump(results, open('results.p', "wb"))
results = pickle.load(open('results.p', "rb"))

pickle.dump(result_name, open('result_name.p', "wb"))
result_name = pickle.load(open('result_name.p', "rb"))
#restaurant recommendation
import pymysql as mdb
import collections
con = mdb.connect('localhost', 'root', '', 'recomdb')
with con:
   cur = con.cursor()
   cur.execute("DROP TABLE IF EXISTS recommendation3")
   cur.execute("CREATE TABLE recommendation3(binary_id text, id1 text, rec1 text,sc1 float,star1 float, url1 text, id2 text, rec2 text,sc2 float,star2 float,url2 text, id3 text, rec3 text,sc3 float,star3 float,url3 text, id4 text, rec4 text,sc4 float,star4 float,url4 text, id5 text, rec5 text,sc5 float,star5 float, url5 text)")
count=0
for j in range(2**15):
    count+=1
    print count
    binary_id='{0:015b}'.format(j)
          
#a=int('010000000000000', 2)
    #b=results[j]
    myrecom={}
    max_review=0
    for i in range(10):
        d=(result_name[j][i],)
        c.execute('select business_name, business_stars, review_count from business where business_id=?', d)
        #print c.fetchone()
        for row in c:
            h=row[2]
        if h>max_review:
            max_review=h
    for i in range(10):
        d=(result_name[j][i],)
        e=str(result_name[j][i])
        f=result_score[j][i]
        myrecom[e]=[]
        c.execute('select business_name, business_stars, review_count from business where business_id=?', d)
        h0=''
        h1=0
        h2=0
        for row in c:
            h0=row[0]
            h1=row[1]
            h2=row[2]
            myrecom[e].append(h0)
            myrecom[e].append(0.7*h1/5+0.3*h2/max_review+10*f)
            myrecom[e].append(h1)
        myrecom[e].append(myurl[e])
    sorted_recom=collections.OrderedDict(sorted(myrecom.items(), key=lambda (k, (v1, v2, v3,v4)): -v2))

    #sorted_recom = sorted(myrecom, key=myrecom.get,reverse=True)
    toadd=[]
    for business in sorted_recom:
        #print business
        #print sorted_recom[business]
        toadd.append([sorted_recom[business][0],sorted_recom[business][1],sorted_recom[business][2],sorted_recom[business][3],business])
    cmd = 'INSERT INTO recommendation3 VALUES ("{0}","{1}","{2}",{3},{4},"{5}","{6}","{7}",{8},{9},"{10}","{11}","{12}",{13},{14},"{15}","{16}","{17}",{18},{19},"{20}","{21}","{22}",{23},{24},"{25}")'.format(binary_id.replace('"', ''), toadd[0][4].replace('"', ''),toadd[0][0].replace('"', ''), toadd[0][1], toadd[0][2],toadd[0][3].replace('"', ''),\
                toadd[1][4].replace('"', ''),toadd[1][0].replace('"', ''), toadd[1][1], toadd[1][2], toadd[1][3].replace('"', ''), toadd[2][4].replace('"', ''),toadd[2][0].replace('"', ''), toadd[2][1], toadd[2][2], toadd[2][3].replace('"', ''), toadd[3][4].replace('"', ''),toadd[3][0].replace('"', ''), toadd[3][1], toadd[3][2], toadd[3][3].replace('"', ''), toadd[4][4].replace('"', ''),toadd[4][0].replace('"', ''), toadd[4][1], toadd[4][2], toadd[4][3].replace('"', ''))
    with con:
        cur.execute(cmd)





#create recommendation table
#import pymysql as mdb
#con = mdb.connect('localhost', 'root', '', 'recomdb')
#with con:
#   cur = con.cursor()
#   cur.execute("DROP TABLE IF EXISTS recommendation")
#   cur.execute("CREATE TABLE recommendation(binary_id text, rec1 text,sc1 float,star1 float, url1 text, rec2 text,sc2 float,star2 float,url2 text, rec3 text,sc3 float,star3 float,url3 text, rec4 text,sc4 float,star4 float,url4 text, rec5 text,sc5 float,star5 float, url5 text)")
#count=0
#for j in range(2**15):
#    count+=1
#    print count
#    binary_id='{0:015b}'.format(j)
#    myrecom={}
#    max_review=0
#    for i in range(10):
#        d=(result_name[a][i],)
#        c.execute('select business_name, business_stars, review_count from business where business_id=?', d)
#        for row in c:
#            h=row[2]
#            if h>max_review:
#                max_review=h
#    for i in range(10):
#        d=(result_name[a][i],)
#        e=str(result_name[a][i])
#        f=result_score[a][i]
#        myrecom[e]=[]
#        c.execute('select business_name, business_stars, review_count from business where business_id=?', d)
#        h0=''
#        h1=0
#        h2=0
#        for row in c:
#            h0=row[0]
#            h1=row[1]
#            h2=row[2]
#            myrecom[e].append(h0)
#            myrecom[e].append(0.7*h1/5+0.3*h2/max_review+10*f)
#            myrecom[e].append(h1)
#        myrecom[e].append(myurl[e])
#    myrecom=pd.DataFrame.from_dict(myrecom, orient='index', dtype=None)
#    myrecom.columns=['name','score','url']
#    myrecom=myrecom.sort('score',ascending=False)      
#    cmd = 'INSERT INTO recommendation2 VALUES ("{0}","{1}",{2},{3},"{4}","{5}",{6},{7},"{8}","{9}",{10},{11},"{12}","{13}",{14},{15},"{16}","{17}",{18},{19},"{20}")'.format(binary_id, myrecom.iat[0,0].replace('"', ''),myrecom.iat[0,1],\
#                myrecom.iat[1,0].replace('"', ''),myrecom.iat[1,1],myrecom.iat[2,0].replace('"', ''),myrecom.iat[2,1],myrecom.iat[3,0].replace('"', ''),myrecom.iat[3,1],myrecom.iat[4,0].replace('"', ''),myrecom.iat[4,1])
#    c.execute(cmd)
#    db.commit()
##a=int('010000000000000', 2)
#    #b=results[j]
#    i=[0,2,4,6,8,10,12,14,16,18]
#    myrecom={}
#    max_review=0
#    for o in i:
#        d=(results[j][o],)
#        c.execute('select business_name, business_stars, review_count from business where business_id=?', d)
#        #print c.fetchone()
#        for row in c:
#            h=row[2]
#        if h>max_review:
#            max_review=h
#    for o in i:
#        d=(results[j][o],)
#        c.execute('select business_name, business_stars, review_count from business where business_id=?', d)
#        h0=''
#        h1=0
#        h2=0
#        for row in c:
#            h0=row[0]
#            h1=row[1]
#            h2=row[2]
#            myrecom[h0]=0.7*h1/5+0.3*h2/max_review
#    
#    sorted_recom = sorted(myrecom, key=myrecom.get,reverse=True)
#    for business in sorted_recom:
#        print business
#        print myrecom[business]
#    cmd = 'INSERT INTO recommendation VALUES ({0},"{1}",{2},"{3}","{4}",{5},{6},"{7}",{8},"{9}",{10})'.format(binary_id, sorted_recom[0].replace('"', ''),myrecom[sorted_recom[0]],\
#                sorted_recom[1].replace('"', ''),myrecom[sorted_recom[1]],sorted_recom[2].replace('"', ''),myrecom[sorted_recom[2]],sorted_recom[3].replace('"', ''),myrecom[sorted_recom[3]],sorted_recom[4].replace('"', ''),myrecom[sorted_recom[4]])
#    c.execute(cmd)
#    db.commit()

#show on screen    
#old method
a=int('000000110000000', 2)
i=[0,2,4,6,8,10,12,14,16,18]
myrecom={}
max_review=0
for o in i:
    d=(results[a][o],)
    c.execute('select business_name, business_stars, review_count from business where business_id=?', d)
    #print c.fetchone()
    for row in c:
        h=row[2]
    if h>max_review:
        max_review=h
for o in i:
    d=(results[a][o],)
    c.execute('select business_name, business_stars, review_count from business where business_id=?', d)
    h0=''
    h1=0
    h2=0
    for row in c:
        h0=row[0]
        h1=row[1]
        h2=row[2]
        myrecom[h0]=0.7*h1/5+0.3*h2/max_review
sorted_recom = sorted(myrecom, key=myrecom.get,reverse=True)
for business in sorted_recom:
        print business
        print myrecom[business]


#i=[0,2,4,6,8,10,12,14,16,18]
#new method
a=int('000000010000000', 2)
a=int('000010110000000', 2)
myrecom={}
max_review=0
for i in range(10):
    d=(result_name[a][i],)
    c.execute('select business_name, business_stars, review_count from business where business_id=?', d)
    for row in c:
        h=row[2]
    if h>max_review:
        max_review=h
for i in range(10):
    d=(result_name[a][i],)
    e=str(result_name[a][i])
    f=result_score[a][i]
    myrecom[e]=[]
    c.execute('select business_name, business_stars, review_count from business where business_id=?', d)
    h0=''
    h1=0
    h2=0
    for row in c:
        h0=row[0]
        h1=row[1]
        h2=row[2]
        myrecom[e].append(h0)
        myrecom[e].append(0.7*h1/5+0.3*h2/max_review+10*f)
    sorted_recom = sorted(myrecom, key=myrecom.get,reverse=True)
for business in sorted_recom:
    print business
    print myrecom[business]
myrecom=pd.DataFrame.from_dict(myrecom, orient='index', dtype=None)
myrecom.columns=['name','score']
myrecom=myrecom.sort('score',ascending=False)



#    myrecom[e].append(myurl[e])
#    d=(results[a][o],)
#    c.execute('select business_name, business_stars, review_count from business where business_id=?', d)
#        #print c.fetchone()
#    for row in c:
#        h=row[2]
#    if h>max_review:
#        max_review=h
#for o in i:
#    d=(results[a][o],)
#    e=str(results[a][o])
#    myrecom[e]=[]
#    c.execute('select business_name, business_stars, review_count from business where business_id=?', d)
#    h0=''
#    h1=0
#    h2=0
#    for row in c:
#        h0=row[0]
#        h1=row[1]
#        h2=row[2]
#        myrecom[e].append(h0)
#        myrecom[e].append(0.7*h1/5+0.3*h2/max_review)
#    myrecom[e].append(myurl[e])
        
sorted_recom = sorted(myrecom, key=myrecom.get,reverse=True)
for business in sorted_recom:
    print business
    print myrecom[business]
#c.execute('select business_name, business_stars from business where business_id=?', b)
#b=('aXRtJioBYidoHdS2GTTKhA',)
#print.c.fetchone()


#data story
import seaborn as sns

###plot the distribution of topics over the corpus
lda_data=pd.DataFrame(whole_lda)
lda_data.sum()



###plot the correlation between topics
mydata=pd.DataFrame.from_dict(business_review_lda_new, orient='index', dtype=None)
mydata.columns=['Japanese','Mexican','Brunch','none1','Bar/Atmosphere','Service (bad)','Compliment','Pizza','none2','Indian','Asian','Fastfood','none3','Sweets','BBQ']

sns.set(style="darkgrid")
f, ax = plt.subplots(figsize=(9, 9))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.corrplot(mydata, annot=False, sig_stars=False,
             diag_names=False, cmap=cmap, ax=ax)
plt.title('Correlation Matrix - All Topics')
f.tight_layout()

###plot the histogram of resraurant ratings for classification
star=pd.DataFrame.from_dict(business_star, orient='index', dtype=None)
n, bins, patches = plt.hist(star[0], num_bins, facecolor='red', alpha=0.5)
count=0
for i in star[0]:
    if i>3.5:
        count+=1

fig, ax = plt.subplots(1,1)
ax.hist(star, align='left')
###plot the importance of different topics

sns.set(style="white")

order=sorted(range(len(coefficient_scale)), key=lambda k: coefficient_scale[k])
order=[5, 11, 13, 10, 9, 7, 2, 1, 0, 14, 4, 6]
order_topic=mydata.columns[order]
importance=logistic_classifier.coef_.transpose()[order]
importance=pd.DataFrame(importance,index=order_topic)
importance.plot(kind='barh',width=0.8)
matplotlib.rc('ytick',labelsize=25,color='darkblue')
matplotlib.rc('xtick',labelsize=15)
matplotlib.rc('font',family='Arial')
plt.xlabel('Relative Importance For Each Topic', fontsize=25,color='darkblue')
plt.show()

f, ax = plt.subplots(figsize=(8, 6))

label=np.arange(1, 16)
sns.barplot(order_topic, importance, ci=None, palette="Blues", hline=.1)

g = sns.factorplot("year", data=coefficient_scale, palette="BuPu",
                   size=6, aspect=1.5, x_order=label)
g.set_xticklabels(step=2)    

sns.set(style="darkgrid")
f, ax = plt.subplots(figsize=(9, 9))
sns.corrplot(corrmat, annot=False, sig_stars=False,
             diag_names=False, ax=ax)      
plt.title('Correlation Matrix - All Variables')

#business_score={}
#for item in business_review_lda:
#    business_score[item]=0
#    tmp1=business_review_lda[item]
#    for thing in tmp1:
#        business_score[item]+=thing[1]*coefficient_scale[thing[0]]
#    print business_score[item]
    
#for item in business_review_lda:
#    tmp=business_review_lda[item]
#    for thing in tmp:
#        if thing[0]==1:
#            if thing[1]>value:
#                value=thing[1]
#                linshi=item

#classification using SVM
from sklearn import svm

count=0
lda_accuracy=[]
for i in range(1,10):
    count+=1
    print count
    maxIter = 10
    accuracy = np.ones(maxIter)
    svmclf = svm.SVC(C = 3, kernel = 'rbf')
    for iter in range(0,maxIter):
        print iter
        trainIdx = np.random.choice(len(X), np.floor(len(X)*0.1*i), replace = False).astype(int)
        testIdx = np.ones(len(X), dtype = bool)
        testIdx[trainIdx] = False;
        svm_classifier = svmclf.fit(X[trainIdx,:], YY[trainIdx])
        y_svm_predicted = svm_classifier.predict(X[testIdx,:])
        accuracy[iter] = metrics.precision_score(YY[testIdx], y_svm_predicted)
    accuracy    
    lda_accuracy.append(accuracy.mean())


svm_classifier.coef_

count=0
bow_accuracy=[]
for i in range(1,10):
    count+=1
    print count
    maxIter = 10
    accuracy = np.ones(maxIter)
    svmclf = svm.SVC(C = 3, kernel = 'rbf')
    for iter in range(0,maxIter):
        print iter
        trainIdx = np.random.choice(len(XX), np.floor(len(XX)*0.1*i), replace = False).astype(int)
        testIdx = np.ones(len(XX), dtype = bool)
        testIdx[trainIdx] = False;
        svm_classifier = svmclf.fit(XX[trainIdx,:], YY[trainIdx])
        y_svm_predicted = svm_classifier.predict(XX[testIdx,:])
        accuracy[iter] = metrics.precision_score(YY[testIdx], y_svm_predicted)
    accuracy    
    bow_accuracy.append(accuracy.mean())


#linear regression
from sklearn import preprocessing
from sklearn import linear_model 
Y_scaled = preprocessing.scale(Y)
maxIter = 1
residual = np.ones(maxIter)
regr = linear_model.LinearRegression(copy_X=True, fit_intercept=True, normalize=False)
for iter in range(0,maxIter):
    print iter
    trainIdx = np.random.choice(len(X), np.floor(len(X)*0.9), replace = False).astype(int)
    testIdx = np.ones(len(X), dtype = bool)
    testIdx[trainIdx] = False;
    regr.fit(X[trainIdx,:], Y_scaled[trainIdx])
    residual[iter]= np.mean((regr.predict(X[testIdx,:]) - Y[testIdx])**2)
    
regr.coef_
regr.intercept_
    
#classification using Random Forest
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score
clf = RandomForestClassifier(n_estimators=2000)
clf = clf.fit(X, YY)
scores = cross_val_score(clf, X, YY)  
scores.mean()
clf.feature_importances_
