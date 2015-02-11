# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a scape script file to download images from yelp.
"""

import json
import urllib2
from BeautifulSoup import BeautifulSoup 
import sqlite3
import os

import oauth2

API_HOST = 'api.yelp.com'
DEFAULT_TERM = 'dinner'
DEFAULT_LOCATION = 'San Francisco, CA'
SEARCH_LIMIT = 1
SEARCH_PATH = '/v2/search/'
BUSINESS_PATH = '/v2/business/'

# OAuth credential placeholders that must be filled in by users.
CONSUMER_KEY = 'your yelp consumer_key'
CONSUMER_SECRET = 'your yelp consumer_secret'
TOKEN = 'your yelp token'
TOKEN_SECRET = 'your yelp token_secret'

def request(host, path, url_params=None):
    """Prepares OAuth authentication and sends the request to the API.
    Args:
        host (str): The domain host of the API.
        path (str): The path of the API after the domain.
        url_params (dict): An optional set of query parameters in the request.
    Returns:
        dict: The JSON response from the request.
    Raises:
        urllib2.HTTPError: An error occurs from the HTTP request.
    """
    url_params = url_params or {}
    url = 'http://{0}{1}?'.format(host, path)

    consumer = oauth2.Consumer(CONSUMER_KEY, CONSUMER_SECRET)
    oauth_request = oauth2.Request(method="GET", url=url, parameters=url_params)

    oauth_request.update(
        {
            'oauth_nonce': oauth2.generate_nonce(),
            'oauth_timestamp': oauth2.generate_timestamp(),
            'oauth_token': TOKEN,
            'oauth_consumer_key': CONSUMER_KEY
        }
    )
    token = oauth2.Token(TOKEN, TOKEN_SECRET)
    oauth_request.sign_request(oauth2.SignatureMethod_HMAC_SHA1(), consumer, token)
    signed_url = oauth_request.to_url()
    
    print 'Querying {0} ...'.format(url)

    conn = urllib2.urlopen(signed_url, None)
    try:
        response = json.loads(conn.read())
    finally:
        conn.close()

    return response
    
    
def search(term, location):
    """Query the Search API by a search term and location.
    Args:
        term (str): The search term passed to the API.
        location (str): The search location passed to the API.
    Returns:
        dict: The JSON response from the request.
    """
    url_params = {
        'term': term.replace(' ', '+'),
        'location': location.replace(' ', '+'),
        'limit': SEARCH_LIMIT
    }
    return request(API_HOST, SEARCH_PATH, url_params=url_params)

   
def get_business(business_id):
    """Query the Business API by a business ID.
    Args:
        business_id (str): The ID of the business to query.
    Returns:
        dict: The JSON response from the request.
    """
    business_path = BUSINESS_PATH + business_id

    return request(API_HOST, business_path)
    
def downloadRestImage(rest_id, rest_name, location, rests_path):        
    search_res = search(rest_name, location)
    """"
    function used to download restaurant images
    Args:
    @rest_id: restaurant id number, notice that it is different with business id
    @rest_name: restaurant name
    @location: location of restaurants, e.g. las vegas, phoniex
    @rests_path: the path to save images from restaurants
    Returns:
    none
    """"
    # if exist search result
    if search_res['total'] > 0:
            
        business_id = search_res['businesses'][0]['id']
        url = 'http://www.yelp.com/biz_photos/'+business_id+'#'+rest_id       
        soup = BeautifulSoup(urllib2.urlopen(url).read())
        
        ''''analyze html by beautiful soup to obtain image url ''''
        imgs = soup.findAll("div", {"class":"photo-box biz-photo-box pb-ms"})
        
        if len(imgs) > 0:
            print('restaurant name is ' + rest_name + ', num image is ' + str(len(imgs)))
            rest_path = rests_path + rest_id + '/' 
            if not os.path.exists(rest_path):
                os.makedirs(rest_path)
            
            with open(rest_path +'name.txt', 'wb') as restfile:
                restfile.write(rest_name + '\n')
                if search_res['businesses'][0].has_key('businesses'):
                    categories =  search_res['businesses'][0]['categories'][0]
                    restfile.write(str(len(categories)) + '\n')
                    for category in categories:
                        restfile.write(category + '\n')
                else:
                    restfile.write(str(0) + '\n')
            restfile.close()
        
        '''' Download all images from web ''''
        for idx, img in enumerate(imgs):
            txt = open(rest_path +'%04d.jpg' % (idx+1), "wb")
            img_url = img.a.contents[1]['src']
            img_url = img_url[0:-6] + 'l.jpg'
            download_img = urllib2.urlopen(img_url)
            txt.write(download_img.read())
        
            txt.close()
        
        soup.close()
        
'''' main script ''''
rests_path = './restaurants/'
if not os.path.exists(rests_path):
    os.makedirs(rests_path)

''''connect to database''''    
db = sqlite3.connect('./yelp.db')
c = db.cursor()
c.execute('SELECT business_id, business_name, city FROM business')

cnt= 0;
for idx, row in enumerate(c):
    cnt = cnt + 1
    print('download image from restaurant #' + str(idx))
    downloadRestImage(row[0], row[1], row[2], rests_path)

print(cnt)
c.close()