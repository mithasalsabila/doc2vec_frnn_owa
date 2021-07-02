#!/usr/bin/env python
# coding: utf-8

# # Load Data & Preprocessing

# In[1]:


import pandas as pd
import os


# In[2]:


parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
data_dir = os.path.join(parent_dir, 'data')
models_dir = os.path.join(parent_dir, 'models')
print('working directory: ', os.getcwd())
print('data directory:    ', data_dir, )
print('models directory:  ', models_dir)


# In[3]:


# change pandas column width so we can see the posts
pd.get_option('max_colwidth')
pd.set_option('max_colwidth', 2000)


# In[4]:


# read the data from excel 
train = pd.read_excel(r"C:\Users\ASUS\Downloads\New folder\train.xlsx", sep='\t')
test = pd.read_excel(r"C:\Users\ASUS\Downloads\New folder\test.xlsx", sep='\t')
dev = pd.read_excel(r"C:\Users\ASUS\Downloads\New folder\dev.xlsx", sep='\t')

#Mengisi NaN merged cells
train= train.fillna(method='ffill')
test = test.fillna(method='ffill')
dev = dev.fillna(method='ffill')


# In[5]:


train.head()


# In[6]:


#TEST : Menghapus data duplikat, Mengabungkan kolom excel yang merged, Menggabungkan test case-test step-expected result, Mengubah label jadi tipe integer

test_m = pd.DataFrame()
test_m = (test[['Test Case ID','Test Case','Label']]
                  .drop_duplicates('Test Case ID')
                  .set_index('Test Case ID'))

test.drop_duplicates('Test Case')
test.drop_duplicates('Test Step')
test.drop_duplicates('Expected Result')

test_m['Test_Steps'] = test.groupby('Test Case ID')['Test Step'].apply(' '.join)               
test_m['Expected_Results'] = test.groupby('Test Case ID')['Expected Result'].apply(' '.join)
test_m['Messages']=test_m['Test Case']+'_'+test_m['Test_Steps']+'_'+test_m['Expected_Results']

test_m.Label.replace("Dependen", 1, inplace = True)
test_m.Label.replace("Independen", 0, inplace = True)


# In[7]:


#DEV : Menghapus data duplikat, Mengabungkan kolom excel yang merged, Menggabungkan test case-test step-expected result, Mengubah label jadi tipe integer

dev_m = pd.DataFrame()
dev_m = (test[['Test Case ID','Test Case','Label']]
                  .drop_duplicates('Test Case ID')
                  .set_index('Test Case ID'))

dev.drop_duplicates('Test Case')
dev.drop_duplicates('Test Step')
dev.drop_duplicates('Expected Result')

dev_m['Test_Steps'] = test.groupby('Test Case ID')['Test Step'].apply(' '.join)               
dev_m['Expected_Results'] = test.groupby('Test Case ID')['Expected Result'].apply(' '.join)
dev_m['Messages']=dev_m['Test Case']+'_'+dev_m['Test_Steps']+'_'+dev_m['Expected_Results']

dev_m.Label.replace("Dependen", 1, inplace = True)
dev_m.Label.replace("Independen", 0, inplace = True)


# In[8]:


#TRAIN : Menghapus data duplikat, Mengabungkan kolom excel yang merged, Menggabungkan test case-test step-expected result, Mengubah label jadi tipe integer

train_m = pd.DataFrame()
train_m = (train[['Test Case ID','Test Case','Label']]
                  .drop_duplicates('Test Case ID')
                  .set_index('Test Case ID'))

train.drop_duplicates('Test Case')
train.drop_duplicates('Test Step')
train.drop_duplicates('Expected Result')

train_m['Test_Steps'] = train.groupby('Test Case ID')['Test Step'].apply(' '.join)               
train_m['Expected_Results'] = train.groupby('Test Case ID')['Expected Result'].apply(' '.join)
train_m['Messages']=train_m['Test Case']+'_'+train_m['Test_Steps']+'_'+train_m['Expected_Results']

train_m.Label.replace("Dependen", 1, inplace = True)
train_m.Label.replace("Independen", 0, inplace = True)

train_m.head()


# In[9]:


test_data = test_m.drop(columns=['Test Case','Test_Steps','Expected_Results'])
train_data = train_m.drop(columns=['Test Case','Test_Steps','Expected_Results'])
dev_data = dev_m.drop(columns=['Test Case','Test_Steps','Expected_Results'])
data = pd.concat([train_data, dev_data], ignore_index=True)

train_data.head()


# # Data Cleaning

# In[10]:


import nltk
import string
import re
from nltk.corpus import stopwords
from string import punctuation


# In[11]:


nltk.download('stopwords')
stop = set(stopwords.words('indonesian', 'english'))
punctuation = list(string.punctuation)
stop.update(punctuation)

def remove_stopwords(text):
    final_text = []
    for i in text.split():
        if i.strip().lower() not in stop and i.strip().lower().isalpha():
            final_text.append(i.strip().lower())
    return " ".join(final_text)


# In[12]:


def clean_text(text):
    text = remove_stopwords(text)
    return text

train_data['Messages'] = train_data['Messages'].apply(clean_text)
test_data['Messages'] = test_data['Messages'].apply(clean_text)
data['Messages'] = data['Messages'].apply(clean_text)

train_data.head()


# # WordCloud

# In[13]:


pip install wordcloud


# In[14]:


import matplotlib.pyplot as plt
import wordcloud
from wordcloud import WordCloud 

plt.figure(figsize = (20,20))
wc = WordCloud(max_words = 2000, width = 1600, height = 800).generate(" ".join(train_data[train_data.Label == 1].Messages))
plt.imshow(wc, interpolation = 'bilinear')


# # Embedding Using Doc2Vec

# In[15]:


from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument


# In[16]:


pip install sklearn


# In[17]:


from sklearn.model_selection import train_test_split

def label(corpus, label_type):
    labeled = []
    for i,v in enumerate(corpus):
        label = label_type + '_' + str(i)
        labeled.append(TaggedDocument(v.split(), [label]))
    return labeled


# In[18]:


X = train_data.Messages
y = train_data.Label


# In[19]:


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.2)

X_train = label(X_train, 'Train')
X_test = label(X_test, 'Test')
all_data = X_train + X_test


# In[20]:


import numpy as np
def get_vectors(model_, corpus_size, vectors_size, vectors_type):
    vectors = np.zeros((corpus_size, vectors_size))
    for i in range(0, corpus_size):
        prefix = vectors_type + '_' + str(i)
        vectors[i]= model_.dv[prefix]
    return vectors


# In[21]:


from sklearn import utils

model_dbow = Doc2Vec(dm=0, vector_size=300, negative = 5, min_count = 1, alpha = 0.065, min_alpha = 0.065)
model_dbow.build_vocab([x for x in all_data])

for epoch in range(50):
    model_dbow.train(utils.shuffle([x for x in all_data]), total_examples=len(all_data), epochs=1)
    model_dbow.alpha -= 0.002
    model_dbow.min_alpha = model_dbow.alpha


# In[22]:


train_vectors_dbow  = get_vectors(model_dbow, len(X_train), 300, 'Train')
test_vectors_dbow = get_vectors(model_dbow, len(X_test), 300, 'Test')


# # Imbalanced Classification Using IFROWANN

# In[23]:


from frlearn.neighbours import FRNN
from frlearn.neighbours.neighbour_search import KDTree, NNSearch
from frlearn.utils.owa_operators import OWAOperator, mean


# In[24]:


def least_indices_and_values(a, m, axis=-1):
    '''
    The helper-function for the Cosine relation class that sorts distances
    '''
    ia = np.argpartition(a, m - 1, axis=axis)
    a = np.take_along_axis(a, ia, axis=axis)
    take_this = np.arange(m)
    ia = np.take(ia, take_this, axis=axis)
    a = np.take(a, take_this, axis=axis)
    i_sort = np.argsort(a, axis=axis)
    ia = np.take_along_axis(ia, i_sort, axis=axis)
    a = np.take_along_axis(a, i_sort, axis=axis)
    return ia, a


# In[25]:


class Cosine(NNSearch):
    '''
    This class defines the cosine similarity relation that can be used into FRNN OWA classification method
    Cosine metric: cos(A, B) = (A * B)(||A|| x ||B||),
                    where A, B - embedding vectors of tweets, * is a scalar product, and ||.|| is the vector norm
    Cosine similarity: cos_similarity(A, B) = (1 + cos(A, B))/2.
    '''
    class Model(NNSearch):
        X_T: np.ndarray
        X_T_norm: np.ndarray
            
        def _query(self, X, m_int: int):
            distances = 1 - 0.5 * X @ self.X_T / np.linalg.norm(X, axis=1)[:, None] / self.X_T_norm
            return least_indices_and_values(distances, m_int, axis=-1)
        
    def construct(self, X) -> Model:
        model: NNSearch = super().construct(X)
        model.X_T = np.transpose(X)
        model.X_T_norm = np.linalg.norm(model.X_T, axis=0)
        return model


# In[26]:


def frnn_owa_method(train_data, y, test_data, vector_name, NNeighbours, lower, upper):
    '''
    This function implements FRNN OWA classification method
    Input:  train_data - train data in form of pandas DataFrame
            y - the list of train_data golden labels
            test_data - train data in form of pandas DataFrame
            vector_name - string, the name of vector with features in train_data and test_data
            NNeighbours - the int number of neighbours for FRNN OWA method
            lower, upper - lower and upper approximations, calculated with Ordered Weighted Average (OWA)
                aggregation operators possible options: strict(), exponential(), invadd(), mean(), additive()
    Output: conf_scores - the list of confidence scores, y_pred - the list of predicted labels
    '''
    def create_X(data, vector_name):
        # Helper function to create matrix X for train and test data
        indexes = list(data.index)
        vector_size = len(data[vector_name][indexes[0]])
        X = np.zeros((len(data), vector_size))
        for k in range(len(indexes)):
            for j in range(vector_size):
                X[k][j] = data[vector_name][indexes[k]][j]
        return X

    X_train = create_X(train_data, vector_name)
    X_test = create_X(test_data, vector_name)
    OWA = OWAOperator(NNeighbours)
    nn_search = Cosine()
    clf = FRNN(nn_search=nn_search, upper_weights=upper, lower_weights=lower, lower_k=NNeighbours, upper_k=NNeighbours)
    cl = clf.construct(X_train, y)
    # confidence scores
    conf_scores = cl.query(X_test)
    # labels
    y_pred = np.argmax(conf_scores, axis=1)
    return conf_scores, y_pred


# In[27]:


def weights_sum_test(conf_scores, alpha, classes = 2):
    '''
    This function performs rescaling and softmax transformation of confidence scores
    Input:  conf_scores - the list of confidence scores
            alpha - the int parameter used for confidence scores rescaling, by default it is equal to 0.8
            classes - the amount of classes for which confidence scores where calculated
    Output: the list of transformed confidence scores
    '''
    conf_scores_T = conf_scores.T
    conf_scores_T_rescale = [[(conf_scores_T[k][i]-0.5)/(alpha)
                              for i in range(len(conf_scores_T[k]))] for k in range(classes)]
    conf_scores_T_rescale_sum = [sum(conf_scores_T_rescale[k]) for k in range(classes)]
    res = [np.exp(conf_scores_T_rescale_sum[k])/sum([np.exp(conf_scores_T_rescale_sum[m])
                                                     for m in range(classes)]) for k in range(classes)]
    return res


# In[28]:


def test_ensemble_confscores(train_data, y, test_data, vector_names, NNeighbours, lower, upper, alpha):
    '''
    This function performs ensemble of FRNN OWA methods based on confidence scores outputs
    Input:  train_data - pandas DataFrame with train data that contains features 'vector_names'
            y - pandas Series or list, golden labels of train_data instances
            test_data - pandas DataFrame with test data that contains features 'vector_names'
            vector_names - the list of strings, names of features vectors in train_data and test_data
            NNeighbours - the list of int numbers, where each number is amount of neighbours 'k' that will be used
                to perform FRNN OWA classification method for the corresponded feature vector.
                Lenghts of 'vector_names' and 'NNeighbours' lists should be equal.
            lower, upper - lower and upper approximations, calculated with Ordered Weighted Average (OWA)
                aggregation operators with possible options: strict(), exponential(), invadd(), mean(), additive()
            alpha - the int parameter used for confidence scores rescaling, by default it is equal to 0.8
    Output: y_pred_res - the list of predicted labels
    '''
    # Create and fill 3D array with dimentions: number of vectors, number of test instances, number of classes
    conf_scores_all = np.zeros((len(vector_names), len(test_data), 2))
    for j in range(len(vector_names)):
        # Calculate confidence scores for each feature vector
        result = frnn_owa_method(train_data, y, test_data, vector_names[j], NNeighbours[j], lower, upper)[0]
        # Check for NaNs
        for k in range(len(result)):
            if np.any(np.isnan(result[k])):
                result[k] = [0, 0]
        conf_scores_all[j] = (result)
    # Rescale obtained confidence scores
    rescaled_conf_scores = np.array([weights_sum_test(conf_scores_all[:, k, :], alpha) 
                                     for k in range(len(conf_scores_all[0]))])
    # Use the mean voting function to obtain the predicted label
    y_pred_res = [np.round(6*np.average(k, weights=[0, 1])) for k in rescaled_conf_scores]
    return y_pred_res


# In[29]:


def test_ensemble_labels(train_data, y, test_data, vector_names, NNeighbours, lower, upper):
    '''
    This function performs ensemble of FRNN OWA methods based on labels as output
    Input:  train_data - pandas DataFrame with train data that contains features 'vector_names'
            y - pandas Series or list, golden labels of train_data instances
            test_data - pandas DataFrame with test data that contains features 'vector_names'
            vector_names - the list of strings, names of features vectors in train_data and test_data
            NNeighbours - the list of int numbers, where each number represent amount of neighbours 'k' that will
                be used to perform FRNN OWA classification method for the corresponded feature vector.
                Lenghts of 'vector_names' and 'NNeighbours' lists should be equal. 
            lower, upper - lower and upper approximations, calculated with Ordered Weighted Average (OWA)
                aggregation operators with possible options: strict(), exponential(), invadd(), mean(), additive()
    Output: y_pred_res - the list of predicted labels
    '''
    y_pred = []
    for j in range(len(vector_names)):
        y_pred.append(frnn_owa_method(train_data, y, test_data, vector_names[j], NNeighbours[j], lower, upper)[1])
    # Use voting function to obtain the ensembled label - we used mean
    y_pred_res = np.mean(y_pred, axis=0)
    return y_pred_res


# In[30]:


def cross_validation_ensemble_owa(df, vector_names, K_fold, NNeighbours, lower, upper, method, alpha=0.8):
    '''
    This function performs cross-validation evaluation for FRNN OWA ensemble
    Input:  df - pandas DataFrame with features to evaluate
            vector_names - the list of strings, names of features vectors in df
            K_fold - the number of folds of cross-validation, we used K_fold = 5
            NNeighbours - the list of int numbers, where each number represent amount of neighbours 'k' that
                will be used to perform FRNN OWA classification method for the corresponded feature vector.
                Lenghts of 'vector_names' and 'NNeighbours' lists should be equal.
            lower, upper - lower and upper approximations, calculated with Ordered Weighted Average (OWA)
                aggregation operators with possible options: strict(), exponential(), invadd(), mean(), additive()
            method - this string variable defines the output of FRNN OWA approach, it can be 'labels' or 'conf_scores'
            alpha - the int parameter used for confidence scores rescaling, by default it is equal to 0.8
    Output: Pearson Correlation Coefficient (PCC) as float number
            PCC = (sum_i(x_i-mean(x))(y_i-mean(y)))/(sqrt(sum_i(x_i-mean(x))^2*sum_i(y_i-mean(y))^2)),
            where x_i and y_i present the i-th components of vectors x and y
    '''
    frnn_acc = []
    random_indices = np.random.permutation(df.index)
    for i in range(K_fold):
        # Split df on train and test data
        test_data = df.loc[df.index.isin(random_indices[i*len(df.index)//K_fold:(i+1)*len(df.index)//K_fold])]
        train_data = df[~df.index.isin(test_data.index)]
        y = train_data['Label']
        y_true = test_data['Label']
        # Apply FRNN OWA method for each feature vector depends on specified output type
        if method == 'labels':
            # Solution for labels calculation
            y_pred_res = test_ensemble_labels(train_data, y, test_data, vector_names, NNeighbours, lower, upper)
        elif method == 'conf_scores':
            # Solution for confidence scores calculation
            y_pred_res = test_ensemble_confscores(train_data, y, test_data, vector_names, NNeighbours, lower, upper, alpha)
        else:
            print('Wrong output type was specified!')
        # Calculate PCC
        frnn_acc.append(accuracy_score(y_true, y_pred_res)[0])
        #pearson.append(pearsonr(y_true, y_pred_res)[0])
    #return np.mean(pearson)


# In[31]:


def namestr(obj, namespace):
    '''
    This function return the name of 'obj' variable
    Input: obj - any variable, namespace - the namespace setup, we used 'globals()'
    Output: string - the name of 'obj' variable
    '''
    return [name for name in namespace if namespace[name] is obj]


# In[32]:


# Datasets charachteristics

for dataset in [train_data]:
    print('Characteristics of ', namestr(dataset, globals())[0])
    print('Number of instances: ', len(dataset))
    print('Size of the smallest class: ', min([len(dataset[dataset.Label == i]) for i in range(2)]))
    print('Imbalance Ratio (IR): ', round(max([len(dataset[dataset.Label == i]) for i in range(2)])/min([len(dataset[dataset.Label == i]) for i in range(2)]), 1))
    print('\n')


# In[45]:


def get_vector_d2v(Messages):
    '''
    This function provides Word2Vec embedding for the tweet
    Input: Word2Vec model imported with KeyedVectors, tweet as a string
    Output: 300-dimentional vector as a list
    '''
    vectors = []
    for w in Messages.split():
        if w in model_dbow.dv:
            vectors.append(model_dbow.dv._g__getitem__(w))
        else:
            vectors.append(np.zeros(300))
    return np.mean(vectors, axis=0)


# In[46]:


data["Vector_d2v"] = data['Messages'].apply(lambda x: get_vector_d2v(x))
test_data["Vector_d2v"] = test_data['Messages'].apply(lambda x: get_vector_d2v(x))


# In[47]:


# The number of cross-validation folds
K_fold = 5


# In[48]:


from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from frlearn.utils.owa_operators import additive


# In[49]:


print('accuracy %s' % accuracy_score(cross_validation_ensemble_owa(data, ["Vector_d2v"], K_fold, [19], additive(), additive(), 'labels')))
print(classification_report(cross_validation_ensemble_owa(data, ["Vector_d2v"], K_fold, [19], additive(), additive(), 'labels')))


# In[ ]:




