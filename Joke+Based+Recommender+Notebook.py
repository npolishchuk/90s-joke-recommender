
# coding: utf-8

# In[1]:

cd C:\Users\npolishchuk\Pictures\DePaul\CSC 478\Assignment 4\JokeBasedRecommender


# In[2]:

import numpy as np


# ## Item-Based Joke Recommendation 

# ### a. Load in the joke ratings data and the joke text data

# In[3]:

parseJester = [line.strip().split(',') for line in open('modified_jester_data.csv').readlines()]
jester = []
for s in parseJester:
    t = [float(u) for u in s]
    jester.append(t)
print jester[0:5]


# In[4]:

jokes = np.genfromtxt('jokes.csv',delimiter=',',dtype=[('myint','<i8'),('mystring','S1000')])


# In[5]:

print jokes[0:5]


# ### b. Complete the definition for the function "test".  (Special note: this was a course assignment at DePaul.)

# In[6]:

# removed the test and print_most_similar_jokes functions
import itemBasedRec as ibr


# In[7]:

npjester = np.array(jester)
len(jester)


# In[8]:

def test(dataMat, test_ratio, estMethod):
    # Write this function to iterate over all users and for each perform cross-validation on items by calling
    # the above cross-validation function on each user.
    # MAE will be the ratio of total error across all test cases to the total number of test cases, for all users
    lendataMat = len(dataMat)
    cumError = 0
    cumCount = 0
    for j in range (0,lendataMat):
        ckError = ibr.cross_validate_user(dataMat, j, test_ratio, estMethod)
        cumError += ckError[0]
        cumCount += ckError[1]
    MAE = cumError / cumCount
    print 'Mean Absolute Error for ',estMethod,' : ', MAE


# In[ ]:

test(npjester, .2, ibr.standEst)


# In[ ]:

test(npjester, .2, ibr.svdEst)


# The SVD-based version had a slightly lower MAE, but took 5 times longer to run. This does not appear to be a better approach for this dataset.

# ### c. Write a new function "print_most_similar_jokes" which takes the joke ratings data, a query joke id, a parameter k for the number of nearest neighbors, and a similarity metric function, and prints the text of the query joke as well as the texts of the top k most similar jokes based on user ratings.

# In[99]:

def print_most_similar_jokes(dataMat, jokes, queryJoke, k, metric=ibr.pearsSim):
	# Write this function to find the k most similar jokes (based on user ratings) to a queryJoke
	# The queryJoke is a joke id as given in the 'jokes.csv' file (an corresponding to the a column in dataMat)
	# You must compare ratings for the queryJoke (the column in dataMat corresponding to the joke), to all
	# other joke rating vectors and return the top k. Note that this is the same as performing KNN on the 
    # columns of dataMat. The function must retrieve the text of the joke from 'jokes.csv' file and print both
	# the queryJoke text as well as the text of the returned jokes.
    n = len(jokes)
    simArray = np.zeros((n))
    for j in range(n):
        if j <> queryJoke:
            similarity = metric(npjester[:,j],npjester[:,queryJoke])
            simArray[j] =  similarity
    idx = np.argsort(simArray)[::-1]
    print 'Selected joke: \n'
    print jokes[queryJoke]
    print '\n Top {} Recommended Jokes Are: \n'.format(k)
    for i in range(k):
        print jokes[idx[i]]
        print '------------------'


# In[98]:

npjokes = np.array(jokes)


# In[100]:

print_most_similar_jokes(jester, npjokes, 15, 5, ibr.pearsSim)

