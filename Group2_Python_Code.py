#!/usr/bin/env python
# coding: utf-8

#All the files were kept in the current working directory, add the path of the directory if required before importing
import scrapy
from scrapy import Request
from scrapy.spiders import CrawlSpider, Rule
from scrapy.linkextractors import LinkExtractor
# item models

import pandas as pd
import numpy as np
import nltk
import urllib.request

from sklearn.feature_extraction.text import CountVectorizer
from selenium import webdriver
from bs4 import BeautifulSoup
import time
from bs4.element import Tag
import urllib.request
import os
import cv2
import tensorflow as tf
import sys
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split



#Creating the spider to extract the SF data (Same spider was used to extract data from other cities to make training dataset.)
class HouseSpider(scrapy.Spider):
    name = 'house' # name of the spider
    allowed_domains = ['craigslist.org']
    start_urls = ['https://sfbay.craigslist.org/search/apa?']  # name of the url to scrape

    def parse(self, response):        #for the firsr layer of page
        

        #defining the path
        deals = response.xpath('//p[@class="result-info"]')
        for deal in deals:
            title = deal.xpath('a/text()').extract_first()    # gives you the title
            date_posted = deal.xpath('time[@class="result-date"]/text()').get("")  # gives date posted
            time=deal.xpath('time[@class="result-date"]/@datetime').get("")
            price = deal.xpath('span[@class="result-meta"]/span[@class="result-price"]/text()').get(" Missing ")  # price for rent
            Size = deal.xpath('span[@class="result-meta"]/span[@class="housing"]/text()').get("")  # number of bedrooms
            
            neighborhood = deal.xpath('span[@class="result-meta"]/span[@class="result-hood"]/text()').get(" Missing ")[2:-1]  # gives the area
            
            
            lower_rel_url = deal.xpath('a/@href').extract_first()  # find next layer page url
            lower_url = response.urljoin(lower_rel_url)
            # call parse_lower function
            yield Request(lower_url, callback=self.parse_lower, meta={'Title': title, 'Date': date_posted, 'Time': time, 'Price': price, 
                                                                      'Area/Bedrooms': Size, 'Neighborhood': neighborhood })

        next_rel_url = response.xpath('//a[@class="button next"]/@href').get()  # move to next page
        next_url = response.urljoin(next_rel_url)
        yield Request(next_url, callback=self.parse)

    def parse_lower(self, response):  # for next layer of pages
        
        # getting the content/ description by moving to the next page
        text = "".join(line for line in response.xpath('//*[@id="postingbody"]/text()').getall())
        im=response.xpath('//div[@class="gallery"]/span[@class="slider-info"]/text()').get()
        info = response.xpath('//p[@class="attrgroup"]/span[@class="shared-line-bubble"]/b/text()').getall()
        
        phone=response.xpath('//*[@id="postingbody"]/a[@class="show-contact"]/text()').getall()
        location = response.xpath('//div[@class="mapbox"]/div[@class="mapaddress"]/text()').get()
        
        response.meta['Text'] = text
        response.meta['Number of Images'] = im
        response.meta['Bedroom/Bathroom'] = info
        response.meta['Phone Number'] = phone
        response.meta['Location'] = location
        
        yield response.meta


#Data CLeaning and Pre-possesing


df=pd.read_csv('house_sf.csv')
#Converting into string forms and replacing symbols with blanks
df['Number of Rooms'] = df['Bedroom/Bathroom'].str.extract(pat= '([0-9])', expand=True)
df['Bathrooms'] = df['Bedroom/Bathroom'].str.extract(pat= '([,][0-9])', expand=True)
df['Bathrooms']=df['Bathrooms'].str.replace(",","")

df['Price'] = df['Price'].str.replace("$","")

df['Posting Time'] = df['Time'].str.split(r' ').str.get(1)
df['Images'] = df['Number of Images'].str.split(r' ').str.get(3)
df['Text']=df['Text'].str.lower().str.strip()

df['Text']=df['Text'].str.replace("\n", " ")
df['Text']=df['Text'].str.replace("#", "")
df['Text']=df['Text'].str.replace("*", "")

text=df['Text']

fixed_text = text[pd.notnull(text)]

c1=fixed_text.tolist()


#Tokenizing the Description and Titles columns to create Count of Descripotion & Count of Title Columns

for i in range(len(c1)):
    token_c1 = nltk.word_tokenize(c1[i])

    token_1.append(token_c1)
count_desc=[]

for j in range(len(token_1)):
    count_desc.append(len(token_1[j]))

df['Count of Description']=pd.DataFrame(count_desc)

df['Title']=df['Title'].str.lower().str.strip()

df['Title']=df['Title'].str.replace("\n", " ")

text=df['Title']

fixed_text = text[pd.notnull(text)]

c1=fixed_text.tolist()

token_1=[]


for i in range(len(c1)):
    token_c1 = nltk.word_tokenize(c1[i])

    token_1.append(token_c1)


count_title=[]

for j in range(len(token_1)):
    count_title.append(len(token_1[j]))

df['Count of Title']=pd.DataFrame(count_title)

df['Count of Description'].fillna(0, inplace=True)
df['Count of Title'].fillna(0, inplace=True)


#Creating a flag variable which says whether the location and phone number present in the post or not

for index, row in df.iterrows():
    if (len(row['Location'].str.split(" "))==0) :
        df['spam_location']=1
    else:
        df['spam_location']=0
        
for index, row in df.iterrows():
    if ((row['Phone Number']=="")):
        df['spam_number']=1
    else:
        df['spam_number']=0

#Creating a count of Images variable

df['Images'].fillna(0, inplace=True)
df['Images'].isnull()
std_images=df['Images'].astype(float).std()
mean_images=df['Images'].astype(float).mean()

df["Images"]=df["Images"].astype(float)

lower_limit=mean_images-1.5*std_images
upper_limit=mean_images+1.5*std_images
print(upper_limit)
print(lower_limit)

for index, row in df.iterrows():
    if (row['Images']>=upper_limit) or (row['Images']<=lower_limit):
        df['spam_image']=1
    else:
        df['spam_image']=0


#Scrapping images from Craiglist post and saving in our local drive

img_addrs=[]

for i in range(0,3000,120):
    driver = webdriver.Chrome('chromedriver.exe')
    driver.get('https://sfbay.craigslist.org/search/apa?s=' + str(i))
    time.sleep(3)

    soup = BeautifulSoup(driver.page_source,'html.parser')
    rowArray = soup.find_all("li", { "class":"result-row"})

    for row in rowArray:
        img = row.find("img")
        if img is None:
            continue
        if isinstance(img,Tag) and img.has_attr("src"):
            img_addrs.append(img['src'])



adr="F:/test2/"
count=0

for i in img_addrs:
    count=count+1
    filename = i.split('/')[-1]
    urllib.request.urlretrieve("https://images.craigslist.org/" + filename, adr + str(count) + "_image.jpg")

#Detecting a text from the extracted images 


# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

# Import utilites
from utils import label_map_util
from utils import visualization_utils as vis_util

l1 = list()
l2 = list()
i = 1
while i < 2985:   

# Name of the directory containing the object detection module we're using
    MODEL_NAME = 'inference_graph'
    IMAGE_NAME = str(i) + '_image.jpg'



# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
    PATH_TO_CKPT = os.path.join(CWD_PATH, MODEL_NAME, 'frozen_inference_graph.pb')

# Path to label map file
    PATH_TO_LABELS = os.path.join(CWD_PATH, 'training', 'labelmap.pbtxt')

# Path to image
    PATH_TO_IMAGE = os.path.join(CWD_PATH, IMAGE_NAME)

# Number of classes the object detector can identify
    NUM_CLASSES = 3

# Load the label map.
# Label maps map indices to category names, so that when our convolution
# network predicts `5`, we know that this corresponds to `king`.
# Here we use internal utility functions, but anything that returns a
# dictionary mapping integers to appropriate string labels would be fine
    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

# Load the Tensorflow model into memory.
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

        sess = tf.Session(graph=detection_graph)

# Define input and output tensors (i.e. data) for the object detection classifier

# Input tensor is the image
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Output tensors are the detection boxes, scores, and classes
# Each box represents a part of the image where a particular object was detected
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

# Each score represents level of confidence for each of the objects.
# The score is shown on the result image, together with the class label.
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

# Number of objects detected
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

# Load image using OpenCV and
# expand image dimensions to have shape: [1, None, None, 3]
# i.e. a single-column array, where each item in the column has the pixel RGB value
    image = cv2.imread(PATH_TO_IMAGE)
    image_expanded = np.expand_dims(image, axis=0)

# Perform the actual detection by running the model with the image as input
    (boxes, scores, classes, num) = sess.run([detection_boxes, detection_scores,
                                              detection_classes, num_detections],
                                             feed_dict={image_tensor: image_expanded})
    print(scores[0])
    x = [j for j in scores[0] if j > 0.6]
    print(x)
    len(x)
    if len(x) > 0:
        l2.append('1')
    else:
        l2.append('0')
    image_detection_variable_df= pd.DataFrame(data=l2)
    i += 1
#Creating a column for image detection variables
image_detection_variable_df.columns=['index','image_based_value']


#Merging the image_detection_variable_df to df

#Namimg the index column of df as index
df.index = df["index"]


df_1 = pd.merge(df, image_detection_variable_df[['image_based_value']], on='index')


#Build a text analytics model for the description of the ad lisintg
data = pd.read_csv("Description.csv")

data['Description'] = data['Description'].str.lower().str.strip()
data['Description'] = data['Description'].str.replace("\n", " ")


data1 = pd.read_csv("house_sf.csv")

data1['Text'] = data1['Text'].str.lower().str.strip()
data1['Text'] = data1['Text'].str.replace("\n", " ")

for i in range(len(data1['Text'])):
    data['Description'][length+i] = data1['Text'][i]

text = data['Description']
review_list = text[pd.notnull(text)]

token_list = []
token_filter_list = []

for i in range(len(review_list)):  # iterating the review_list for tokenizing
    # Tokenize
    tokenizer2 = nltk.tokenize.WhitespaceTokenizer()
    Token_d12 = tokenizer2.tokenize(review_list[i])
    token_list.append(Token_d12)  # appending to list
    
    # Lemmatizer
    lemmatizer = nltk.stem.WordNetLemmatizer()  # iterating the review_list for lemmatizing
    lemmatized_token_d2 = [lemmatizer.lemmatize(token) for token in Token_d12 if token.isalpha()]
    # Filtering stop words
    token_filtered = [token for token in lemmatized_token_d2 if not token in stopwords.words('english')]
    token_removed_words = [token for token in lemmatized_token_d2 if token in stopwords.words('english')]
    token_filter_list.append(token_filtered)  # list of filtering

# coverting the list of lists to list of sentences
new_sentence_list = []
for i in token_filter_list:
    new_list = ""
    for j in range(0, len(i)):
        new_list = new_list + " " + i[j]
    new_sentence_list.append(new_list)  # converting to list of sentences


# TF-IDF Vectorizer
vectorizer3 = TfidfVectorizer(ngram_range=(1, 2), min_df=3)
vectorizer3.fit(new_sentence_list)
v3 = vectorizer3.transform(new_sentence_list)

# POS
POS_c2 = []
for token_doc in token_list:
    POS_token_doc = nltk.pos_tag(token_doc)
    POS_token_temp = []
    for i in POS_token_doc:
        POS_token_temp.append(i[0] + i[1])
    POS_c2.append(" ".join(POS_token_temp))

vectorizer4 = TfidfVectorizer(ngram_range=(0, 2), min_df=4)
vectorizer4.fit(POS_c2)

POS_v3 = vectorizer4.transform(POS_c2)


X = POS_v3
X = X.toarray()
y = data['Label']
#Neural Network - Classifier

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

X_train = X[0:3533]
y_train = data['Label'][0:3533]
X_test = X[3533:3633]
y_test = data['Label'][3533:3633]



neigh = clf = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(1, ), random_state=0)
    # KNeighborsClassifier(n_neighbors=1)
neigh.fit(X_train, y_train)
print("accu")
print(neigh.score(X_test, y_test))
output = neigh.predict(review_list[3633:6617])
result = np.array(output, dtype=np.uint8)
np.savetxt('result_text_anlaysis.csv', result)


#Adding the text_analysis csv to the house_sf file
text_analysis_df=pd.read_csv("result_text_anlaysis.csv")


text_analysis_df.columns=['index','Text Analysis Result']
df_2 = pd.merge(df_1, text_analysis_df[['Text Analysis Result']], on='index')

# Performing Title analysis 

data = pd.read_csv("Titles.csv")

data['Title'] = data['Title'].str.lower().str.strip()
data['Title'] = data['Title'].str.replace("\n", " ")

df_2['Title'] = df_2['Title'].str.lower().str.strip()
df_2['Title'] = df_2['Title'].str.replace("\n", " ")


for i in range(len(df_2['Title'])):
    data['Title'][length+i] = df_2['Title'][i]
    
text = data['Title']
review_list = text[pd.notnull(text)]


token_list = []
token_filter_list = []

for i in range(len(review_list)):  # iterating the review_list for tokenizing
    # Tokenize
    tokenizer2 = nltk.tokenize.WhitespaceTokenizer()
    Token_d12 = tokenizer2.tokenize(review_list[i])
    token_list.append(Token_d12)  # appending to list
    # print(token_list)
    # Lemmatizer
    lemmatizer = nltk.stem.WordNetLemmatizer()  # iterating the review_list for lemmatizing
    lemmatized_token_d2 = [lemmatizer.lemmatize(token) for token in Token_d12 if token.isalpha()]
    # Filtering stop words
    token_filtered = [token for token in lemmatized_token_d2 if not token in stopwords.words('english')]
    token_removed_words = [token for token in lemmatized_token_d2 if token in stopwords.words('english')]
    token_filter_list.append(token_filtered)  # list of filtering

# coverting the list of lists to list of sentences
new_sentence_list = []
for i in token_filter_list:
    new_list = ""
    for j in range(0, len(i)):
        new_list = new_list + " " + i[j]
    new_sentence_list.append(new_list)  # converting to list of sentences

# TF-IDF Vectorizer
vectorizer3 = TfidfVectorizer(ngram_range=(1, 2), min_df=3)
vectorizer3.fit(new_sentence_list)
v3 = vectorizer3.transform(new_sentence_list)

# POS
POS_c2 = []
for token_doc in token_list:
    POS_token_doc = nltk.pos_tag(token_doc)
    POS_token_temp = []
    for i in POS_token_doc:
        POS_token_temp.append(i[0] + i[1])
    POS_c2.append(" ".join(POS_token_temp))

vectorizer4 = TfidfVectorizer(ngram_range=(0, 2), min_df=4)
vectorizer4.fit(POS_c2)

POS_v3 = vectorizer4.transform(POS_c2)

X = POS_v3
X = X.toarray()
y = data['Label']
#Neural Network - Classifier

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

X_train = X[0:3533]
y_train = data['Label'][0:3533]
X_test = X[3533:3633]
y_test = data['Label'][3533:3633]



neigh = clf = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(1, ), random_state=0)
    # KNeighborsClassifier(n_neighbors=1)
neigh.fit(X_train, y_train)

print(neigh.score(X_test, y_test))
output = neigh.predict(review_list[3633:6617])
result = np.array(output, dtype=np.uint8)
np.savetxt('result_title_anlaysis.csv', result)


#Adding the text_analysis csv to the house_sf file
text_analysis_df=pd.read_csv("result_title_anlaysis.csv")
text_analysis_df.columns=['index','Title Analysis Result']
df_3 = pd.merge(df_2, text_analysis_df[['Title Analysis Result']], on='index')

#Apply min-max normalization
cols_to_norm = ['Number of Rooms','Bathrooms','Images','Count of Description','Price', 'Count of Title']
df_3[cols_to_norm] = df_3[cols_to_norm].apply(lambda x: (x - x.min()) / (x.max() - x.min()))

#List of all the predictor variables
feature_cols = ['spam_location', 'spam_number', 'Title Analysis Result', 'Text Analysis Result','Number of Rooms',
                'Bathrooms','Price','Count of Description','Images','Count of Title','Image_based_value']


#Preparing training dataset
X = df_3[feature_cols][0:2499]

y = df_3.Y_label[0:2499]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)


#Running Logistic Regression
logreg = LogisticRegression()
logreg.fit(X_train,y_train)
y_pred=logreg.predict(X_test)

#Model Evaluation using Confusion Matrix

from sklearn import metrics
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
cnf_matrix

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))



## Naive Bayes
from sklearn.naive_bayes import MultinomialNB 
NBmodel = MultinomialNB()
# training
NBmodel.fit(X_train,y_train) 
y_pred_NB = NBmodel.predict(X_test)


print("Accuracy:",metrics.accuracy_score(y_test, y_pred_NB))

#SVM
from sklearn.svm import LinearSVC 
SVMmodel = LinearSVC()
# training
SVMmodel.fit(X_train,y_train) 
y_pred_SVM = SVMmodel.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred_SVM))


#Randome Forest and Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
DTmodel = DecisionTreeClassifier()
RFmodel = RandomForestClassifier(n_estimators=50, max_depth=3, bootstrap=True, random_state=0) ## number of trees and number of layers/depth

# training
DTmodel.fit(X_train,y_train) 
y_pred_DT = DTmodel.predict(X_test)
RFmodel.fit(X_train,y_train) 
y_pred_RF = RFmodel.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred_DT))
print("Accuracy:",metrics.accuracy_score(y_test, y_pred_RF))

#Deep Learning
from sklearn.neural_network import MLPClassifier
DLmodel = MLPClassifier(solver='lbfgs'
, hidden_layer_sizes=(3,2), random_state=1)
# training
DLmodel.fit(X_train,y_train)
y_pred_DL= DLmodel.predict(X_test)


print("Accuracy:",metrics.accuracy_score(y_test, y_pred_DL))


#Applying model on the test dataset

X_output_sample =df_3[feature_cols][2499:2984]

output = DLmodel.predict(X_output_sample)

result = np.array(output, dtype=np.uint8)
np.savetxt('output_result.csv', result)

