#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# !pip install gensim
# !pip install python-Levenshtein


# # Reading and Exploring the Dataset
# The dataset we are using here is a subset of Amazon reviews from the Cell Phones & Accessories category. The data is stored as a JSON file and can be read using pandas.
# 
# Link to the Dataset: http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Cell_Phones_and_Accessories_5.json.gz

# In[ ]:


df = pd.read_json("Cell_Phones_and_Accessories_5.json", lines=True)
df


# # Simple Preprocessing & Tokenization
# The first thing to do for any data science task is to clean the data. For NLP, we apply various processing like converting all the words to lower case, trimming spaces, removing punctuations. This is something we will do over here too.
# 
# Additionally, we can also remove stop words like 'and', 'or', 'is', 'the', 'a', 'an' and convert words to their root forms like 'running' to 'run'.

# In[ ]:



review_text = df.reviewText.apply(gensim.utils.simple_preprocess)


# In[ ]:


review_text


# In[ ]:


review_text.loc[0]


# # Training the Word2Vec Model
# Train the model for reviews. Use a window of size 10 i.e. 10 words before the present word and 10 words ahead. A sentence with at least 2 words should only be considered, configure this using min_count parameter.
# 
# Workers define how many CPU threads to be used.

# ## Initialize the model

# In[ ]:


model = gensim.models.Word2Vec(
    window=10,
    min_count=2,
    workers=4,
)


# ## Build Vocabulary

# In[ ]:


model.build_vocab(review_text, progress_per=1000)


# ## Train the Word2Vec Model

# In[ ]:


model.train(review_text, total_examples=model.corpus_count, epochs=model.epochs)


# ## Save the Model
# Save the model so that it can be reused in other applications

# In[ ]:


model.save("./word2vec-amazon-cell-accessories-reviews-short.model")


# In[ ]:


Finding Similar Words and Similarity between words
https://radimrehurek.com/gensim/models/word2vec.html


# In[ ]:



model.wv.most_similar("bad")


# In[ ]:


model.wv.similarity(w1="cheap", w2="inexpensive")


# In[ ]:


model.wv.similarity(w1="great", w2="good")

