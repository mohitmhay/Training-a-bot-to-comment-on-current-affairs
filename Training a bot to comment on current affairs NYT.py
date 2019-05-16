
# coding: utf-8
Python package markovify is used for Markov chain generator. To improve the sentence structure for the generated comments, high performance NLP package spaCy is used for parts of speech tagging and functions from the package markovify are overriden.Steps:
Preparing text from comments for training the generator.
Training a simple Markov chain generator using the comments' text and using it to generate some comments.
Training an improved Markov chain generator with POS-Tagged text and using it to generate more comments.
# In[1]:


import pandas as pd
import markovify 
import spacy
import re

import warnings
warnings.filterwarnings('ignore')

from time import time
import gc


# In[2]:


curr_dir = r'D:\Python\MachineLearning\nyt-comments'
df1 = pd.read_csv(curr_dir + '\CommentsJan2017.csv')
df2 = pd.read_csv(curr_dir + '\CommentsFeb2017.csv')
df3 = pd.read_csv(curr_dir + '\CommentsMarch2017.csv')
df4 = pd.read_csv(curr_dir + '\CommentsApril2017.csv')
df5 = pd.read_csv(curr_dir + '\CommentsMay2017.csv')
df6 = pd.read_csv(curr_dir + '\CommentsJan2018.csv')
df7 = pd.read_csv(curr_dir + '\CommentsFeb2018.csv')
df8 = pd.read_csv(curr_dir + '\CommentsMarch2018.csv')
df9 = pd.read_csv(curr_dir + '\CommentsApril2018.csv')
comments = pd.concat([df1, df2, df3, df4, df5, df6, df7, df8, df9])
comments.drop_duplicates(subset='commentID', inplace=True)
comments.head(3)


# In[3]:


comments.shape


# In[4]:


comments.sectionName.value_counts()[:5]


# In[5]:


def preprocess(comments):
    commentBody = comments.loc[comments.sectionName=='Politics', 'commentBody']
    commentBody = commentBody.str.replace("(<br/>)", "")
    commentBody = commentBody.str.replace('(<a).*(>).*(</a>)', '')
    commentBody = commentBody.str.replace('(&amp)', '')
    commentBody = commentBody.str.replace('(&gt)', '')
    commentBody = commentBody.str.replace('(&lt)', '')
    commentBody = commentBody.str.replace('(\xa0)', ' ')  
    return commentBody


# In[6]:


commentBody = preprocess(comments)
commentBody.shape


# In[7]:


del comments, df1, df2, df3, df4, df5, df6, df7, df8
gc.collect()


# In[8]:


commentBody.sample().values[0]
#A random comment from the dataset:


# In[9]:


start_time = time()
comments_generator = markovify.Text(commentBody, state_size = 5)
print("Run time for training the generator : {} seconds".format(round(time()-start_time, 2)))

Generating comments:
# In[10]:


def generate_comments(generator, number=10, short=False):
    ginti = 0
    while ginti < number:
        if short:
            comment = generator.make_short_sentence(140)
        else:
            comment = generator.make_sentence()
        if comment:
            ginti += 1
            print("Comment {}".format(ginti))
            print(comment)
            print()


# In[11]:


generate_comments(comments_generator)

Improving Markov chain generator using spaCy for POS-Tagging:
The comments generated above are pretty good, but the sentence structure can be improved by using parts of speech tagging. Here we use high-performance library Spacy for this purpose and override the relevant functions in the markovify module.
# In[12]:


nlp = spacy.load('en_core_web_sm')

class POSifiedText(markovify.Text):
    def word_split(self, sentence):
        return ["::".join((word.orth_, word.pos_)) for word in nlp(sentence)]

    def word_join(self, words):
        sentence = " ".join(word.split("::")[0] for word in words)
        return sentence


# In[13]:


commentBody = preprocess(df9)
commentBody.shape

Freeing Memory
# In[14]:


del comments_generator, df9
gc.collect()


# In[15]:


#Getting More Comments
start_time = time()
comments_generator_POSified = POSifiedText(commentBody, state_size = 2)
print("Run time for training the generator : {} seconds".format(round(time()-start_time, 2)))


# In[16]:


generate_comments(comments_generator_POSified)

