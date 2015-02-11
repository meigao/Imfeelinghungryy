# -*- coding: utf-8 -*-
"""
Created on Sun Jan 25 15:40:10 2015

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

db = sqlite3.connect('/Users/gaomei/Desktop/Yelp_project/yelp.db')
c = db.cursor()
c.execute('select distinct business.business_id, business.business_name from business')

myurl={}
punt_list=['.',',','!','?','(',')',':','<','>','/','[',']','"','|',';','%','$','=','+','*',"'"]
for row in c:
    print row[0]
    myurl[row[0]]='"http://www.yelp.com/biz/'
    s=list(row[1].lower())
    ts=''.join([ o for o in s if not o in punt_list ]).split()
    for item in ts:
        if item=='&':
            item='and'
        myurl[row[0]]=myurl[row[0]]+str(item)+'-'
    myurl[row[0]]=myurl[row[0]]+'phoenix"'
    print myurl[row[0]]
    
#create table for pop-up words
topic0=['sushi','roll','tuna','salmon','sashimi','japanese','fish','tempura','rice','sushi']
topic1=['tacos','mexican','salsa','burrito','chips','beans','tortillas','enchiladas','carne asada','guacamole']
topic2=['breakfast','coffee','eggs','bacon','pancakes','toast','sandwich','brunch','benedict','fresh']  
#topic3=['wine','patio','atmosphere','bar','drinks','fun','beers','happyhour','music','service']
topic3=['pizza','crust','italian','italian','pepperoni','cheese','italian','cheese','pizzas','pizza']     
topic4=['indian','buffet','masala','naan','tikka','paneer','lumpia','curry','lamb','chicken']
topic5=['thai','pho','chinese','curry','springs','noodles','spicy','asian','vietnamese','tofu']
topic6=['burger','fries','potato','onion-rings','hot-dog','milkshake','fast-food','cheeseburger','bacon','burgers']
#topic8=['bagels','smoothies','cheese','service','friendly','amazing','experience','awesome','fresh','favorate']
topic7=['bbq','chicken','pork','ribs','brisket','meat','salad','lamb','rack','beef']
topic8=['wine','happyhour','bar','drinks','beers','happyhour','music','drinks','bar','wine']
topic9=['service','atmosphere','fun','friendly','amazing','awesome','favorite','service','friendly','service']
tmp0=pd.Series(topic0)
tmp1=pd.Series(topic1)
tmp2=pd.Series(topic2)
tmp3=pd.Series(topic3)
tmp4=pd.Series(topic4)
tmp5=pd.Series(topic5)
tmp6=pd.Series(topic6)
tmp7=pd.Series(topic7)
tmp8=pd.Series(topic8)
tmp9=pd.Series(topic9)

topic_data=pd.DataFrame(dict(topic0 = tmp0, topic1 = tmp1,topic2 = tmp2,topic3 = tmp3,topic4 = tmp4,topic5 = tmp5,topic6 = tmp6,topic7 = tmp7,topic8 = tmp8,topic9 = tmp9 ))
topic_data.to_sql(con=con, name='topic_table', if_exists='replace', flavor='mysql')


topics=['Japanese','Mexican','brunch','bar','service','compliment','Asian','fastfood','bbq','others']
prob=[0.039595, 0.105054,0.040948,0.046386,0.293094,0.199379,0.060035,0.026484, 0.161650,0.02737599]
text=[]
for i in range(10):
    for j in range(int(1000*prob[i])):
         text.append(topics[i])
    text.append(int(1000*prob[i])*topics[i])
text=str(text)
wordcloud = WordCloud(font_path='/Library/Fonts/Kai.ttf').generate(text)
plt.imshow(wordcloud)

#extract the lat and lon of all restaurants
c.execute('select distinct business.business_id, business.business_name, latitude, longitude from business')
location=[]
count=0
for row in c:
    location.append([row[2], row[3]])
count=0
for item in location:
    count+=1
    print count
    print item

##find the top user with the most reviews for restaurant in Phoenix
c.execute('select user_id, count(reviews) from business, review where business.business_id=review.business_id group by user_id order by count(reviews) desc limit(10)')
top_user=[]
for row in c:
    print row[0]
    top_user.append(row[0])
#ikm0UCahtK34LbLCEw4YTw
#C6IOtaaYdLIT5fWd7ZYIuA
#3gIfcQq5KxAegwCPXc83cQ
#M6oU3OBf_E6gqlfkLGlStQ
#JgDkCER12uiv4lbpmkZ9VA
#q9XgOylNsSbqZqF_SO3-OQ
#vhxFLqRok6r-D_aQz0s-JQ
#pEVf8GRshP9HUkSpizc9LA
#gcyEUr4DXcbjnGRAWFtfAQ
#usQTOj7LQ9v0Fl98gRa3Iw

#extract reviews from top users
top_user_review={}
for i in range(10):
    print i
    top_user_review[top_user[i]]=[]
    b=(top_user[i],)
    c.execute('select reviews from business, review where business.business_id=review.business_id and user_id=?', b)
    for row in c:
        s=list(row[0].lower())
        ts=''.join([ o for o in s if not o in  punt_list ]).split()
        for word in ts:
            if word in myworddic:
                top_user_review[top_user[i]].append(word)
 
user_topic={}
for user in top_user_review:
    user_topic[user]=lda[dictionary.doc2bow(top_user_review[user])]

        
               


        
        




#user review    
c.execute('select distinct business.business_id, review_stars from business, review where business.business_id=review.business_id and user_id="q9XgOylNsSbqZqF_SO3-OQ" order by review_stars')
user_review={}
count=0
for row in c:
    print count
    count+=1
    user_review[row[0]]=row[1]
user_data=pd.DataFrame.from_dict(user_review, orient='index', dtype=None)
user_data.to_csv('whatever.csv')
new_df = pd.read_csv('whatever.csv')

myrecom.to_csv('whoever.csv')
new_myrecom = pd.read_csv('whoever.csv')
new_myrecom.columns=['id','name','score']


user_data=new_df
user_data.columns=['id','star']
user_data=user_data.sort('star',ascending=False)
user_rec=pd.DataFrame(result_name[a])
user_rec.columns=['id']

tmp1=[]
for item in user_data['id']:
    tmp1.append(item)
tmp2=[]
for item in new_myrecom['id']:
    tmp2.append(item)
tmp3=set(tmp1) & set(tmp2)
len(tmp3)

count=0  
order_list=[]  
for item in tmp2:
    mytmp=str(item)
    if mytmp in list(user_data['id']):
        print mytmp
        print user_data[user_data['id']==mytmp]
        p=user_data[user_data['id']==mytmp]
        order_list.append(p.iat[0,1])

import random
random_result=[]
for k in range(1000):
    print k
    wrong_count2=0
    random_orderlist=[]
    for i in range(len(tmp3)):
        random_orderlist.append(random.randint(1, 5))
    for i in range(len(tmp3)):
        for j in range(0,i):
            if random_orderlist[i]>random_orderlist[j]:
                wrong_count2+=1
    random_result.append(wrong_count2/float(len(tmp3))/(len(tmp3)-1)*2)
print np.array(random_result).mean()

random_orderlist=[]
for i in range(len(tmp3)):
    random_orderlist.append(random.randint(1, 5))
wrong_count1=0
wrong_count2=0
for i in range(len(tmp3)):
    for j in range(0,i):
        if order_list[i]>order_list[j]:
            wrong_count1+=1
        if random_orderlist[i]>random_orderlist[j]:
            wrong_count2+=1
            
print wrong_count1/float(len(tmp3))/(len(tmp3)-1)*2
print wrong_count2/float(len(tmp3))/(len(tmp3)-1)*2
        

c.execute('select business_id from business')
test=[]
for row in c:
    print row[0]
    test.append(row[0])
count=0
for item in user_rec['id']:
    print type(item)
    if item in test:
        count+=1
        print count
        print item
   
count=0
for item in user_data['id']:
    print type(item)
    if item in test:
        count+=1
        print count
        print item





import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="white", context="talk")
f, ax = plt.subplots(figsize=(8, 6))
#x = np.array(list("ABCDEFGHIJ"))
x = np.array(['Rand','Daren','Jennifer','Janice','Sunny','Evan','Melissa','Brian','Dolores','Dottsy'])
y=np.array([0.21324803412,0.231693989071,0.27380952381,0.107719298246,0.252551020408,0.157575757576,0.252551020408,0.151282051282,0.24691358024691357,0.201132201132])
sns.barplot(x, y, ci=None, palette="husl", hline=0, ax=ax)
ax.set_ylabel("NDPM Score")
sns.despine(bottom=True)
plt.setp(f.axes, yticks=[0,0.1,0.2,0.3,0.4,0.5])
plt.tight_layout(h_pad=3)
Rand    Daren Jennifer Janice  Sunny   Evan   Melissa Brian   Dolores Dottsy    


import numpy as np
import matplotlib.pyplot as plt

error1=(0.21324803412,0.27380952381, 0.107719298246, 0.201132201132, 0.157575757576, 0.252551020408, 0.231693989071, 0.24691358024691357, 0.151282051282, 0.252551020408)
error2=(0.284980237, 0.293567251, 0.129047619, 0.21798419, 0.236538462, 0.278021978, 0.253296703, 0.252747253, 0.172435897, 0.275)
error3=(0.404466666667, 0.400295813315,0.401174833636,0.401915151515, 0.401992063492,0.400599435028, 0.399757062147, 0.398620289855,0.398562318841, 0.401038647343)
N=10
ind = np.arange(N)
width = 0.23       # the width of the bars
fig, ax = plt.subplots()
plt.ylim(0,0.6)

rects1 = ax.bar(ind, error1, width, color='red',alpha=1)
rects2 = ax.bar(ind+width, error2, width, color='yellow')
rects3 = ax.bar(ind+width+width, error3, width, color='blue')
ax.set_ylabel('NDPM Score',fontsize=20)
#ax.set_title('NDPM Score: Restaurant Recommendation',fontsize=15)
ax.set_xticks(ind+width*1.5)
ax.set_xticklabels( ('U1', 'U2', 'U3', 'U4', 'U5','U6','U7','U8','U9','U10'),fontsize=17)
#ax.set_xticklabels( ('Rand', 'Daren', 'Jennifer', 'Janice', 'Sunny','Evan','Melissa','Brian','Dolores','Dottsy'),fontsize=10)

ax.legend( (rects1[0], rects2[0], rects3[0]), ('Topic Modeling', 'Rank by Star', 'Random'),fontsize=11)
matplotlib.rc('ytick',labelsize=15)
matplotlib.rc('xtick',labelsize=5,color='black')
matplotlib.rc('font',family='bold Arial')
plt.show()

#plot the error for image classification
imageerror=[0.23259,0.21275,0.22422,0.21872,0.20921,0.21623,0.21974,0.21675]
N=8
ind = np.arange(N)
width = 0.5
f, ax = plt.subplots(figsize=(6, 3))
plt.ylim(0,0.6)
rects1 = ax.bar(ind+0.3, imageerror, width, color='red',alpha=1)
ax.set_ylabel('NDPM Score',fontsize=20)
#ax.set_title('NDPM Score: Image Recommendation',fontsize=15)
ax.set_xticks(ind+width*1.1)
ax.set_xticklabels( ('bbq', 'chinese', 'cocktail', 'coffee', 'sweets','pasta','pizza','sushi'),fontsize=10)
matplotlib.rc('ytick',labelsize=15)
matplotlib.rc('xtick',labelsize=5,color='black')
matplotlib.rc('font',family='bold Arial')

def mergesort(myarray,start,end):
    result=[]
    size=len(myarray)
    mid=(start+end)/2
    A=mergesort(myarray,start,mid)
    B=mergesort(myarray,mid+1,end)
    i=0
    j=0
    while(i<=mid and j<=end-mid-1):
        if A[i]<B[j]:
            result.append(A[i])
            i+=1
        else:
		result.append(B[j])
		j+=1
    if j==end-mid:
        result.append(A[i:mid])
    else:
        result.append(B[j:end-mid-1])
    return result
  