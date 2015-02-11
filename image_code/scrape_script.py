# -*- coding: utf-8 -*-
"""
Created on Sun Jan 18 21:02:10 2015

@author: Meigao
"""

import urllib2
import flickrapi
import numpy

def save_images(url, category, index): 
    ''''
    function to save image from url
    Args:
    @url: image url
    @category: image to which category
    @index: image index with 5 digits. e.g. 00001.jpg
    Returns:
    None
    ''''
    save_path = './data/' + category + '/%05d.jpg'
    txt = open(save_path % index, "wb")
    print 'downloading:', url
    download_img = urllib2.urlopen(url)
    txt.write(download_img.read())
    txt.close();


# up to 4000 images
def query_images(category, num_per_page, max_page):
    ''''
    function to query images from flickr by given category
    we can query 100 images per page and 8 pages, then query total 4000 images
    Args:
    @cateogry: category name to query. e.g. coffee, burge...
    @num_per_page: number of images per query. e.g. 100
    @max_page: max number of query e.g. 8
    Return:
    None
    ''''
    api_key = 'bc046a60611b9ecfcb25789bfef051e2'
    secret_key = 'c184af5b48824b16';
    flickr = flickrapi.FlickrAPI(api_key, secret = secret_key) 
    #path to save favorite count    
    save_path = './data/' + category + '/label.txt'
    txt = open(save_path, "wb")
    
    
    for i in range(0, max_page):
        photos = flickr.photos_search(text=category,sort= 'relevance', 
                                      page=i+1, per_page=num_per_page, 
                                      extras = 'views')
        for idx, photo in enumerate(photos[0]):
            index = idx + 1 + i * num_per_page 
            print 'process image # %d' % index
            img_url = "http://farm%s.static.flickr.com/%s/%s_%s_z.jpg" % (photo.get('farm'), photo.get('server'), photo.get('id'), photo.get('secret'))
            ''''save views used for rankSVM''''
            save_images(img_url, category, index);
            views_count = photo.get('views')
            txt.write(views_count + '\n')
    txt.close()
    
def query_neg_images(category, num_per_page, max_page, startidx = 1,
                     save_path = './data/neg/label.txt'):
    '''''
    function to query images from flickr by given category
    we can query 100 images per page and 8 pages, then query total 4000 images
    Args:
    @cateogry: category name to query. e.g. coffee, burge...
    @num_per_page: number of images per query. e.g. 100
    @max_page: max number of query e.g. 8
    @startidx: index to save image e.g. 00001.jpg
    Return:
    None
    ''''
    api_key = 'your filckr API key'
    secret_key = 'your flickr secret key';
    flickr = flickrapi.FlickrAPI(api_key, secret = secret_key) 
    #path to save favorite count    
    txt = open(save_path, "wb")
    
    
    for i in range(0, max_page):
        ''''query using relevance''''
        photos = flickr.photos_search(text=category,sort= 'relevance', 
                                      page=i+1, per_page=num_per_page, 
                                      extras = 'views')
        for idx, photo in enumerate(photos[0]):
            index = idx + 1 + i * num_per_page +startidx-1
            print('process image # %d' % index)
            img_url = "http://farm%s.static.flickr.com/%s/%s_%s_z.jpg" % (photo.get('farm'), photo.get('server'), photo.get('id'), photo.get('secret'))
            save_images(img_url, 'neg', index);
            txt.write('-1\n')
    txt.close()


# main script
num_per_page = 500
max_page = 8
category = 'pizza'
query_images(category, num_per_page, max_page)
category = 'burger'
query_images(category, num_per_page, max_page)
category = 'cocktail'
query_images(category, num_per_page, max_page)
category = 'food sushi'
query_images(category, num_per_page, max_page)
category = 'chinese food'
query_images(category, num_per_page, max_page)
category = 'food ice cream'
query_images(category, num_per_page, max_page)
category = 'food pasta'
query_images(category, num_per_page, max_page)
category = 'korean bbq'
query_images(category, num_per_page, max_page)
category = 'coffee'
query_images(category, num_per_page, max_page)


''''download negative images''''
category = 'eating'
query_neg_images(category, 100, 1)

category = 'restaurant'
query_neg_images(category, 100, 1, 101)

category = 'restaurant logo'
query_neg_images(category, 100, 1, 201)