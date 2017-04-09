
# coding: utf-8

# # Personalized Re-ranking of URLs
# ### Solution to Kaggle's Personalized Web Search Challenge

# K Madhumathi, Tarun Tater, Sushravys G M

# Challenge description -- https://www.kaggle.com/c/yandex-personalized-web-search-challenge

# Data files may be found here -- https://www.kaggle.com/c/yandex-personalized-web-search-challenge/data

# For more details about solution, please refer massive_project_report.pdf

# In[ ]:

import graphlab as gl
import re
import graphlab.aggregate as agg
import time
import operator
import math
import numpy as np
import pandas as pd
from sklearn.decomposition import NMF


# # Converting the raw data into tabular format

# In[ ]:

fopen = open("train.csv", "r")
trans = fopen.readlines()
fopen.close()
outfile = open("trainRevised.csv", "w")
for i in range(1000000):
    
    trans[i] = re.sub("""\n""", "", trans[i],re.I|re.S)
    trans[i] = re.sub(""",""", "#", trans[i],re.I|re.S)
    if('M' in trans[i]):
        trans[i] = trans[i] + "NA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\n"
        outfile.write(trans[i])
    elif('C' in trans[i]):
        trans[i] = trans[i] + "\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\n"
        outfile.write(trans[i])
    else:
        trans[i] = trans[i] + "\n"
        outfile.write(trans[i])
outfile.close()


# In[ ]:

train = gl.SFrame.read_csv("train.csv", delimiter = '\t', header = False)


# # Getting and working on data of 1000 users

# In[ ]:

listofIndices = {}
lId = []
prevId = "0"
e1 = 0
l = [e1]
lStart = []
listOfTime = []

with open("/home/madhumathi/MassiveProject/MassiveData/1000Users.csv") as f:
    for i in xrange(1):
        f.next()
    for e, line in enumerate( f ):
        if(('M' in line) | ('Q' in line)):
            listOfTime.append(-1)
        else:
            listOfTime.append(int(line.split("\t")[1]))
        if(prevId == line.split("\t")[0]):
            pass
        else:
            l.append(e+1)
            listofIndices[int(prevId)] = l
            lId.append(int(prevId))
            prevId = line.split("\t")[0]
            e1 = e+1
            l = [e1]
            lStart.append(int(e1))
            
l = [e1, (e+2)]
listofIndices[int(line.split("\t")[0])] = l
lId.append(int(line.split("\t")[0]))
lastSession = int(line.split("\t")[0])


# # Calculating relevance scores of urls according to user satisfaction

# In[ ]:

relevanceList = []#-1 for query and session.
relevanceList.append('NA')
for i in range(len(listOfTime)-1):
    if((listOfTime[i] != -1) & (listOfTime[i+1] == -1)):
        relevanceList.append(str(3))
    elif((listOfTime[i] > -1) & (listOfTime[i+1] > -1)):
        if((listOfTime[i+1] - listOfTime[i]) < 50):
            relevanceList.append(str(1))
        elif((listOfTime[i+1] - listOfTime[i]) > 399):
            relevanceList.append(str(3))
        else:
            relevanceList.append(str(2))
    elif(listOfTime[i] == -1):
        relevanceList.append('NA')

if(listOfTime[i+1] != -1):
    relevanceList.append(str(2))
else:
    relevanceList.append('NA')


# # Converting data into readable table format with every Query followed by its 10 result URLs and their relevances

# In[ ]:

whichSession = 0 
checkNewSession = listofIndices[whichSession][1]#keeps track of when the new session starts
pages = []#the urls for a query
writeLines = [] #the lines which need to be written when a session completes
writeLine = ""#a string for each line
outfile = open("/home/madhumathi/MassiveProject/MassiveData/train1000Users.csv", "w")
header = 'userId' + ',' + 'day' + ',' + 'sessionId' + ',' + 'typeOfRecord' + ',' + 'queryId' + ','+ 'urlId' + ',' + 'relevance' + ',' + 'terms' + ',' + 'url1' + ',' + 'url2' + ',' + 'url3' + ',' + 'url4' + ',' + 'url5' + ',' + 'url6' + ',' + 'url7' + ',' + 'url8' + ',' + 'url9' + ',' + 'url10' + '\n'
outfile.write(header)#wrote the header
outfile.close()
outfile = open("/home/madhumathi/MassiveProject/MassiveData/train1000Users.csv", "a")
relevance = '0'
isPageClicked = False
shouldWrite = False

with open("/home/madhumathi/MassiveProject/MassiveData/1000Users.csv") as f: 
    
    for e, line in enumerate( f ):
        if(e < checkNewSession):#for each session
            line = line.split('\t')
            if(line[1] == 'M' ):
                userId = line[3]
                day = line[2]
                sessionId = line[0]
                writeLine = ""
                writeLines = []
                newSession = True
                isPageClicked = False
                timeSession = 0
                
            else:
                if(line[2] == 'Q'):
                    if((newSession == False) & (isPageClicked == True)):
                        for eachPageNotClicked in pages:
                            urlId = eachPageNotClicked
                            writeLine = userId + ',' + day + ',' + sessionId + ',' + typeOfRecord + ',' + queryId + ',' + urlId +',' + relevance + ',NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA\n'
                            writeLines.append(writeLine)
                    
                    isPageClicked = False
                    
                    newSession = False
                    newQuery = True
                    typeOfRecord = 'Q'
                    queryId = line[4]
                    urlId = 'NA'
                    terms = line[5]
                    terms = re.sub(""",""", "#", terms,re.I|re.S)
                    url1 = line[6].split(',')[0]
                    url2 = line[7].split(',')[0]
                    url3 = line[8].split(',')[0]
                    url4 = line[9].split(',')[0]
                    url5 = line[10].split(',')[0]
                    url6 = line[11].split(',')[0]
                    url7 = line[12].split(',')[0]
                    url8 = line[13].split(',')[0]
                    url9 = line[14].split(',')[0]
                    url10 = line[15].split(',')[0]
                    pages = [url1, url2, url3, url4, url5, url6, url7, url8, url9, url10]
                    writeLine = userId + ',' + day + ',' + sessionId + ',' + typeOfRecord + ',' + queryId + ',NA,' + 'NA' + ',' + terms + ',' + url1 + ',' + url2 + ',' + url3 + ',' + url4 + ',' + url5 + ',' + url6 + ',' + url7 + ',' + url8 + ',' + url9 + ',' + url10 + '\n'
                    
                else:
                    shouldWrite = True#atleast one page was clicked for at least one query in this session
                    if(newQuery == True):
                        isPageClicked = True
                        writeLines.append(writeLine)
                    typeOfRecord = 'C'
                    urlId = line[4]
                    urlId = re.sub("""\n""","", urlId, re.I|re.S)
                    try:
                        pages.remove(urlId)
                        writeLine = userId + ',' + day + ',' + sessionId + ',' + typeOfRecord + ',' + queryId + ',' + urlId +',' + relevanceList[e] + ',NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA\n'
                        newPageClick = True
                    except:
                        newPageClick = False
                        for j in range(len(writeLines)-1,-1,-1):
                            if(urlId in writeLines[j]):
                                if(writeLines[j].split(',')[6] < relevanceList[e]):
                                    writeLines[j] = userId + ',' + day + ',' + sessionId + ',' + typeOfRecord + ',' + queryId + ',' + urlId +',' + relevanceList[e] + ',NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA\n'
                                    break
                    newQuery = False
                    if(newPageClick == True):
                        writeLines.append(writeLine)

            if( e == (checkNewSession - 1)) :
                if(shouldWrite == True):
                    if(isPageClicked == True):
                        for eachPageNotClicked in pages:
                            urlId = eachPageNotClicked
                            writeLine = userId + ',' + day + ',' + sessionId + ',' + typeOfRecord + ',' + queryId + ',' + urlId +',' + relevance + ',NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA\n'
                            writeLines.append(writeLine)

                    for eachLine in writeLines:
                        outfile.write(eachLine)    
                if(whichSession != lastSession):
                    whichSession = whichSession + 1
                    checkNewSession = listofIndices[whichSession][1]#update session, 
                shouldWrite = False

outfile.close()      


# # Getting 24 days of Train Data

# In[ ]:

userIds = []
outfile = open("/home/madhumathi/MassiveProject/MassiveData/train24days.csv", "w")
header = 'userId' + ',' + 'day' + ',' + 'sessionId' + ',' + 'typeOfRecord' + ',' + 'queryId' + ','+ 'urlId' + ',' + 'relevance' + ',' + 'terms' + ',' + 'url1' + ',' + 'url2' + ',' + 'url3' + ',' + 'url4' + ',' + 'url5' + ',' + 'url6' + ',' + 'url7' + ',' + 'url8' + ',' + 'url9' + ',' + 'url10' + '\n'
outfile.write(header)#wrote the header
outfile.close()
outfile = open("/home/madhumathi/MassiveProject/MassiveData/train24days.csv", "a")

with open("/home/madhumathi/MassiveProject/MassiveData/train1000Users.csv") as f:
    
    for i in xrange(1):
        f.next()
    
    for e, line in enumerate( f ):
        if(int(line.split(',')[1]) < 24):
            outfile.write(line)


# # Getting 3 days of Test Data

# In[ ]:

userIds = []
outfile = open("/home/madhumathi/MassiveProject/MassiveData/test3days.csv", "w")
header = 'userId' + ',' + 'day' + ',' + 'sessionId' + ',' + 'typeOfRecord' + ',' + 'queryId' + ','+ 'urlId' + ',' + 'relevance' + ',' + 'terms' + ',' + 'url1' + ',' + 'url2' + ',' + 'url3' + ',' + 'url4' + ',' + 'url5' + ',' + 'url6' + ',' + 'url7' + ',' + 'url8' + ',' + 'url9' + ',' + 'url10' + '\n'
outfile.write(header)#wrote the header
outfile.close()
outfile = open("/home/madhumathi/MassiveProject/MassiveData/test3days.csv", "a")

with open("/home/madhumathi/MassiveProject/MassiveData/train1000Users.csv") as f:
    
    for i in xrange(1):
        f.next()
    
    for e, line in enumerate( f ):
        if(int(line.split(',')[1]) >= 24):
            if(int(line.split(',')[0]) in users):
                outfile.write(line)


# # Storing the train and test data into SFrames

# In[ ]:

train24days = gl.SFrame.read_csv("/home/madhumathi/MassiveProject/MassiveData/train24days.csv")


# In[ ]:

test3days = gl.SFrame.read_csv("/home/madhumathi/MassiveProject/MassiveData/test3days.csv")


# # Getting a list of all users, queries and URLs from train data

# In[ ]:

users = list(train24days['userId'].unique())
queries = list(train24days['queryId'].unique())
urls = list(train24days['urlId'].unique())


# # Building the User-URL matrix 

# In[ ]:

userUrlMatrix = np.array([[0.0 for i in range(len(urls))] for i in range(len(users))])


# In[ ]:

user = '0'
urlRelTemp = {}
urlCountTemp = {}
for i in urls:
    urlCountTemp[i] = 0
    urlRelTemp[i] = 0
with open("/home/madhumathi/MassiveProject/MassiveData/train24days.csv") as f: 
    for i in xrange(1):
        f.next()
    for e, line in enumerate( f ):
        lineRead = line.split(',')
        if(lineRead[0]!=user):
            for  i in urls:
                if (urlCountTemp[i] > 0):
                    #print user, i, urlRelTemp[i], urlCountTemp[i]
                    score = (urlRelTemp[i]*1.0)/urlCountTemp[i]
                else:
                    score = 0
                userUrlMatrix[users.index(int(user))][urls.index(i)] = score
            user = lineRead[0]
            urlRelTemp = {}
            urlCountTemp = {}
            for i in urls:
                urlCountTemp[i] = 0
                urlRelTemp[i] = 0
        else:
            if(lineRead[3] == 'C'):
                urlRelTemp[int(lineRead[5])] += int(lineRead[6])
                urlCountTemp[int(lineRead[5])] += 1
for  i in urls:
    if (urlCountTemp[i] > 0):
        #print user, i, urlRelTemp[i], urlCountTemp[i]
        score = (urlRelTemp[i]*1.0)/urlCountTemp[i]
    else:
        score = 0
    userUrlMatrix[users.index(int(user))][urls.index(i)] = score      


# # Factorizing the User-URL matrix into User-Topic and URL-Topic

# In[ ]:

model = NMF(n_components=500, init='nndsvd', random_state=0)
model.fit(userUrlMatrix)
urlTopic = model.components_
urlTopic = np.asmatrix(urlTopic)


# In[ ]:

try:
    binv = np.linalg.pinv(urlTopic)
except np.linalg.LinAlgError:
    # Not invertible. Skip this one.
    print "non invertible"


# In[ ]:

userTopic = np.array(userUrlMatrix*binv)
userTopic[userTopic < 0.000000000001] = 0.0


# # Building the Query-URL matrix

# In[ ]:

queryUrlMatrix = np.array([[0.0 for i in range(len(urls))] for j in range(len(queries))])
trainc = train24days[train24days['typeOfRecord'] == 'C']
train = trainc.groupby(key_columns = ['queryId','urlId'],operations = {'count' : agg.COUNT})
trainPandas = gl.SFrame.to_dataframe(train)


# In[ ]:

for index, row in trainPandas.iterrows():
    uindx = urls.index(row['urlId'])
    qindx = queries.index(row['queryId'])
    queryUrlMatrix[qindx][uindx] = row['count']


# # Factorizing the Query-URL matrix into Query-Topic and URL-Topic

# In[ ]:

model = NMF(n_components=500, init='nndsvd', random_state=0)
model.fit(queryUrlMatrix)
urlTopic = model.components_


# # TSPR matrix

# In[ ]:

tspr = userTopic*urlTopic
tspr = np.array(tspr)
# Normalizing the tspr matrix
tspr = (tspr - tspr.mean())/ tspr.var()


# # Giving ranks to URL positions returned by search engine

# In[ ]:

positionRanks = np.array([1,2,3,4,5,6,7,8,9,10])
positionRanks = (positionRanks - positionRanks.mean())/ positionRanks.var()
positionRanks = positionRanks+(-1*positionRanks.min()+1)


# # Getting Navigational Queries

# In[ ]:

listOfqids = {}
check = 0
numberOfuser = 0
with open("/home/madhumathi/MassiveProject/MassiveData/train.csv") as f:
    for e, line in enumerate( f ):
        if(line.split('\t')[1] == 'M'):
            if(int(line.split('\t')[2]) < 7):
                numberOfuser = numberOfuser + 1
                check = 1
                writeLine = line.split('\t')[0] + ',' + line.split('\t')[1] + ',' + line.split('\t')[2] + ',' + line.split('\t')[3] + '\n'
            else:
                check = 0
        elif(check == 1):
            if(line.split('\t')[2] == 'Q'):
                try:
                    listOfqids[int(line.split('\t')[4])] = listOfqids[int(line.split('\t')[4])] + 1
                except:
                    listOfqids[int(line.split('\t')[4])] = 1
                terms = line.split('\t')[5]
                terms = re.sub(""",""", "#", terms,re.I|re.S)
                
                writeLine = line.split('\t')[0] + ',' + line.split('\t')[1] + ',' + line.split('\t')[2] + ',' + line.split('\t')[3] + ',' + line.split('\t')[4] + ',' + terms + ',' + (line.split('\t')[6]).split(',')[0] + ',' + (line.split('\t')[7]).split(',')[0] + ',' + (line.split('\t')[8]).split(',')[0] + ',' + (line.split('\t')[9]).split(',')[0] + ',' + (line.split('\t')[10]).split(',')[0] + ',' + (line.split('\t')[11]).split(',')[0] + ',' + (line.split('\t')[12]).split(',')[0] + ',' + (line.split('\t')[13]).split(',')[0] + ',' + (line.split('\t')[14]).split(',')[0] + ',' + (line.split('\t')[15]).split(',')[0] + '\n'
            elif(line.split('\t')[2] == 'C'):
                writeLine = line.split('\t')[0] + ',' + line.split('\t')[1] + ',' + line.split('\t')[2] + ',' + line.split('\t')[3] + ',' + line.split('\t')[4]
            


# # Testing the 3 days data using NDCG measures

# In[ ]:

test = gl.SFrame.to_dataframe(test3days)


# In[ ]:

ndcg = []
count = 0
for index, row in test.iterrows():
    if (row['typeOfRecord'] == 'Q'):   
        count = 0
        ranks = []
        relevances = []
        relevanceHash = {}
        user = int(row['userId'])
        resultList = [row['url1'], row['url2'], row['url3'], row['url4'], row['url5'], row['url6'], row['url7'], row['url8'], row['url9'], row['url10']]
        resultList = [int(j) for j in resultList]
        dcg = 0
        idcg = 0
        if (int(row['queryId']) not in navigationalQueries):
            for i in range(0,len(resultList)):
                u = resultList[i]
                if(user not in users):
                    t1 = 0
                elif(u not in urls):
                    t1 = 0
                else :
                    t1 = tspr[users.index(user)][urls.index(u)]
                t2 = positionRanks[i]
                rank = (t1*1.0)/t2
                ranks.append((u, rank))
            sorted_urls = sorted(ranks, key=lambda x: x[1], reverse=True)
            personalisedResult = [k[0] for k in sorted_urls]
            
        else:
            personalisedResult = resultList
    else:
        count = count+1
        relevanceHash[int(row['urlId'])] = int(row['relevance'])
        relevances.append((int(row['urlId']), int(row['relevance'])))
        if(count == 10):
            sorted_relevance_urls = sorted(relevances, key=lambda x: x[1], reverse=True)
            relevanceResult = [k[0] for k in sorted_relevance_urls]
            for i in range(0,10):
                idcg = idcg+((pow(2, sorted_relevance_urls[i][1]) - 1)*1.0/math.log(i+2,2))
                dcg = dcg+((pow(2, relevanceHash[personalisedResult[i]]) - 1)*1.0/math.log(i+2,2))
            ndcg.append((dcg*1.0)/idcg)
            

