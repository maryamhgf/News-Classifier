import numpy as np
import math
import csv
import collections
import random
import pandas as pd
import os
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag, word_tokenize
from nltk.stem import PorterStemmer 
from nltk.tokenize import word_tokenize 
from nltk.stem.snowball import SnowballStemmer

#Lemmitization:
def normalizingDescriptonTolist(description):
    output = []
    #description is an string
    descriptionLow = description.lower()
    tokenizer = RegexpTokenizer(r'\w+')
    descriptionWithoutpounctuation = tokenizer.tokenize(descriptionLow)
    stopWords = set(stopwords.words('english')) 
    filteredDescriptionList = [w for w in descriptionWithoutpounctuation if not w in stopWords] 
    filteredDescription = " ".join(filteredDescriptionList)
    wnl = WordNetLemmatizer()
    
    for word, tag in pos_tag(word_tokenize(filteredDescription)):
        wntag = tag[0].lower()
        wntag = wntag if wntag in ['a', 'r', 'n', 'v'] else None
        if not wntag:
            lemma = word
        else:
            lemma = wnl.lemmatize(word, wntag)
        output.append(lemma)
    output2 = normalizingDescriptonTolistStemmingSnowBall(" ".join(output))
    return output2

def normalizingDescriptonTolistStemmingSnowBall(description):
    stemmer = SnowballStemmer("english")
    output = []
    #description is an string
    descriptionLow = description.lower()
    tokenizer = RegexpTokenizer(r'\w+')
    descriptionWithoutpounctuation = tokenizer.tokenize(descriptionLow)
    stopWords = set(stopwords.words('english')) 
    filteredDescriptionList = [w for w in descriptionWithoutpounctuation if not w in stopWords] 
    filteredDescription = " ".join(filteredDescriptionList)
    #print(pos_tag(word_tokenize(filteredDescription)))
    for w in word_tokenize(filteredDescription):
        output.append(stemmer.stem(w))
    return output 

def csvFileToListOfClasses(filename):
    fields = [] 
    classOfNews = ""
    travelWords = []
    businessWords = []
    styleWords = []
    with open(filename, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        fields = next(csvreader)  
        for row in csvreader: 
            numOfCol = 0
            for col in row:
                if(numOfCol == 2):
                    classOfNews = col
                if(numOfCol == 5):
                    link = col
                if(numOfCol == 4):
                    headline = col
                if(numOfCol == 6):
                    if(classOfNews == "TRAVEL"):
                        travelWords.append(" ".join((col, headline, link)))
                    elif(classOfNews == "BUSINESS"):
                        businessWords.append(" ".join((col, headline, link)))
                    elif(classOfNews == "STYLE & BEAUTY"):
                        styleWords.append(" ".join((col, headline, link)))
                numOfCol = numOfCol + 1
    return travelWords, businessWords, styleWords

def randomOverSampling(travelNews, businessNews, styleNews):
    numOfTravelNews = len(travelNews)
    numOfBusinessNews = len(businessNews)
    numOfStyleNews = len(styleNews)
    numOFMajority = max(numOfTravelNews, numOfBusinessNews, numOfStyleNews)
    if(numOFMajority == numOfTravelNews):
        numberOfDuplicate1 = int(numOfTravelNews / numOfBusinessNews)
        newBusinessNews = []
        newBusinessNews = newBusinessNews + businessNews * numberOfDuplicate1
        remaining = random.sample(businessNews, numOfTravelNews - len(newBusinessNews))
        newBusinessNews = newBusinessNews + remaining

        numberOfDuplicate2 = int(numOfTravelNews / numOfStyleNews)
        newStyleNews = []
        newStyleNews = newStyleNews + styleNews * numberOfDuplicate2
        remaining = random.sample(styleNews, numOfTravelNews - len(newStyleNews))
        newStyleNews = newStyleNews + remaining
        return travelNews, newBusinessNews, newStyleNews
    elif(numOFMajority == numOfBusinessNews):
        numberOfDuplicate1 = int(numOfBusinessNews / numOfTravelNews)
        newTravelNews = []
        newTravelNews = newTravelNews + travelNews * numberOfDuplicate1
        remaining = random.sample(travelNews, numOfBusinessNews - len(newTravelNews))
        newTravelNews = newTravelNews + remaining

        numberOfDuplicate2 = int(numOfBusinessNews / numOfStyleNews)
        newStyleNews = []
        newStyleNews = newStyleNews + styleNews * numberOfDuplicate2
        remaining = random.sample(styleNews, numOfBusinessNews - len(newStyleNews))
        newStyleNews = newStyleNews + remaining
        return newTravelNews, businessNews, newStyleNews
    elif(numOFMajority == numOfStyleNews):
        numberOfDuplicate1 = int(numOfStyleNews / numOfTravelNews)
        newTravelNews = []
        newTravelNews = newTravelNews + travelNews * numberOfDuplicate1
        remaining = random.sample(travelNews, numOfStyleNews - len(newTravelNews))
        newTravelNews = newTravelNews + remaining

        numberOfDuplicate2 = int(numOfStyleNews / numOfBusinessNews)
        newBusinessNews = []
        newBusinessNews = newBusinessNews + businessNews * numberOfDuplicate2
        remaining = random.sample(businessNews, numOfStyleNews - len(newBusinessNews))
        newBusinessNews = newBusinessNews + remaining
        return newTravelNews, newBusinessNews, styleNews

def randomOverSampling2(travelNews, businessNews, styleNews):
    numOfTravelNews = len(travelNews)
    numOfBusinessNews = len(businessNews)
    numOfStyleNews = len(styleNews)
    numOFMajority = max(numOfTravelNews, numOfBusinessNews, numOfStyleNews)
    newTravel = []
    newStyle = []
    newBusiness = []
    for i in range(numOFMajority):
        newTravel.append(travelNews[random.randint(0, numOfTravelNews -1)])
        newBusiness.append(businessNews[random.randint(0, numOfBusinessNews -1)])
        newStyle.append(styleNews[random.randint(0, numOfStyleNews -1)])
    return newTravel, newBusiness, newStyle

def exractTrainDataSet(travelWords, businessWords, styleWords):
    numOfTravelClass = len(travelWords)
    numOfBusinessClass = len(businessWords)
    numOfStyleClass = len(styleWords)
    numOfTravelTrain = int(0.8 * numOfTravelClass)
    numOfBusinessTrain = int(0.8 * numOfBusinessClass)
    numOfStyleTrain = int(0.8 * numOfStyleClass)
    travelTrainSet = []
    businessTrainSet = []
    styleTrainSet = []
    for i in range(numOfTravelTrain):
        travelTrainSet.append(travelWords[i])
    for i in range(numOfBusinessTrain):
        businessTrainSet.append(businessWords[i])
    for i in range(numOfStyleTrain):
        styleTrainSet.append(styleWords[i])
    travelTestSet = []
    businessTestSet = []
    styleTestSet = []
    for i in range(numOfTravelClass - numOfTravelTrain):
        travelTestSet.append(travelWords[numOfTravelTrain + i])
    for i in range(numOfBusinessClass - numOfBusinessTrain):
        businessTestSet.append(businessWords[numOfBusinessTrain + i])
    for i in range(numOfStyleClass - numOfStyleTrain):
        styleTestSet.append(styleWords[numOfStyleTrain + i])
    return travelTrainSet, businessTrainSet, styleTrainSet

def exractTestDataSet(numOfTravelClass, numOfTravelTrain, numOfBusinessClass, numOfBusinessTrain, 
       numOfStyleClass, numOfStyleTrain, travelWords, businessWords, styleWords):
    travelTestSet = []
    businessTestSet = []
    styleTestSet = []
    for i in range(numOfTravelClass - numOfTravelTrain):
        travelTestSet.append(travelWords[numOfTravelTrain + i])
    for i in range(numOfBusinessClass - numOfBusinessTrain):
        businessTestSet.append(businessWords[numOfBusinessTrain + i])
    for i in range(numOfStyleClass - numOfStyleTrain):
        styleTestSet.append(styleWords[numOfStyleTrain + i])
    return travelTestSet, businessTestSet, styleTestSet       

def newsToWords(listOfNews):
    allWords = []
    for news in listOfNews:
        words = normalizingDescriptonTolist(news)
        for word in words:
            allWords.append(word)
    return allWords

def laplacianSmootinig(numberOfAllWordsInClass, numberOfDuplication, numberOfFeatures):
    probability = (numberOfDuplication + 1) / (numberOfAllWordsInClass + numberOfFeatures)
    return probability

def getAllFeatures(featureTravel, featureBusiness, featureStyle):
    keyTravel = set(featureTravel.keys())
    keyBusiness = set(featureBusiness.keys())
    keyStyle = set(featureStyle.keys())
    allFeatures = set()
    allFeatures.update(keyTravel)
    allFeatures.update(keyBusiness)
    allFeatures.update(keyStyle)
    return allFeatures

def trainFeatures(travelTrainSet, businessTrainSet, styleTrainSet):
    #They are np.arrays
    allWordsInTravelTrain = newsToWords(travelTrainSet)
    allWordsInBusinessTrain = newsToWords(businessTrainSet)
    allWordsInStyleTrain = newsToWords(styleTrainSet)
    featuresInTravel = dict(collections.Counter(allWordsInTravelTrain))
    featuresInBusiness = dict(collections.Counter(allWordsInBusinessTrain))
    featuresInStyle = dict(collections.Counter(allWordsInStyleTrain))
    numOfAllFeaturesTravel = sum(featuresInTravel.values())
    numOfAllFeaturesBusiness = sum(featuresInBusiness.values())
    numOfAllFeaturesStyle = sum(featuresInStyle.values())
    allFeatures = getAllFeatures(featuresInTravel, featuresInBusiness, featuresInStyle)
    numOfAllFeatures = len(allFeatures)
    dataset = {}
    for feature in allFeatures:
        probabilities = []
        if(feature in featuresInTravel):
            countInTravel = featuresInTravel[feature]
        else:
            countInTravel = 0
        probabilities.append(laplacianSmootinig(numOfAllFeaturesTravel, countInTravel, numOfAllFeatures))
        if(feature in featuresInBusiness):
            countInBusiness = featuresInBusiness[feature]
        else:
            countInBusiness = 0
        probabilities.append(laplacianSmootinig(numOfAllFeaturesBusiness, countInBusiness, numOfAllFeatures))
        if(feature in featuresInStyle):
            countInStyle = featuresInStyle[feature]
        else:
            countInStyle = 0
        probabilities.append(laplacianSmootinig(numOfAllFeaturesStyle, countInStyle, numOfAllFeatures))
        dataset.update({feature : probabilities})
    return dataset

def creatModel(filename):
    print("Start to creat model(Phase2 for balanced data(after oversampling)): ")
    travelNews, businessNews, styleNews = csvFileToListOfClasses(filename)
    travelTrainSetBefore, businessTrainSetBefore, styleTrainSetBefore = exractTrainDataSet(travelNews, businessNews, styleNews)
    travelTrainSet, businessTrainSet, styleTrainSet = randomOverSampling(travelTrainSetBefore, businessTrainSetBefore, styleTrainSetBefore )
    features = trainFeatures(travelTrainSet, businessTrainSet, styleTrainSet)
    PTravel = len(travelTrainSet) / (len(travelTrainSet) + len(businessTrainSet)+ len(styleTrainSet))
    PBusiness = len(businessTrainSet) / (len(travelTrainSet) + len(businessTrainSet) + len(styleTrainSet))
    PStyle = len(styleTrainSet) / (len(travelTrainSet) + len(businessTrainSet) + len(styleTrainSet))
    print("PTravel: ", PTravel, "PBusiness: ", PBusiness, "PStyle: ", PStyle)
    return features, PTravel, PBusiness,PStyle, travelNews, businessNews, styleNews, len(travelTrainSetBefore), len(businessTrainSetBefore), len(styleTrainSetBefore)

def classifyNews(news, features, PTravel, PBusiness, PStyle):
    wordsOfNews = normalizingDescriptonTolist(news)
    pClassTravel = 1
    pClassBusiness = 1
    pClassStyle = 1
    for word in wordsOfNews:
        if(word in features):
            pClassTravel = pClassTravel * features[word][0]
            pClassBusiness = pClassBusiness * features[word][1]
            pClassStyle = pClassStyle * features[word][2]
    pClassTravel = pClassTravel * PTravel
    pClassBusiness = pClassBusiness * PBusiness
    pClassStyle = pClassStyle * PStyle
    if(pClassBusiness < pClassTravel):
        category = "TRAVEL"
        winner = pClassTravel
    if(pClassBusiness >= pClassTravel):
        winner = pClassBusiness
        category = "BUSINESS"
    if(winner < pClassStyle):
        winner = pClassStyle
        category = "STYLE & BEAUTY"
    return category        

def classifyValidationData(testTravel, testBusiness, testStyle, features, PTravel, PBusiness, PStyle):
    numberOfCorrectTravel = 0
    numberOfCorrectBusiness = 0
    numberOfCorrectStyle = 0
    allDetectedTravel = 0
    allDetectedBusiness = 0
    allDetectedStyle = 0
    allTravelDetectedBusiness = 0
    allTravelDetectedStyle = 0
    allBusinessDetectedTravel = 0
    allBusinessDetectedStyle = 0
    allStyleDetectedTravel = 0
    allStyleDetectedBusiness = 0
    for news in testTravel:
        classOfNews = classifyNews(news, features, PTravel, PBusiness, PStyle)
        if(classOfNews == "TRAVEL"):
            numberOfCorrectTravel = numberOfCorrectTravel + 1
            allDetectedTravel = allDetectedTravel + 1
        if(classOfNews == "BUSINESS"):
            allTravelDetectedBusiness = allTravelDetectedBusiness + 1
            allDetectedBusiness = allDetectedBusiness + 1
        if(classOfNews == "STYLE & BEAUTY"):
            allTravelDetectedStyle = allTravelDetectedStyle + 1
            allDetectedStyle = allDetectedStyle + 1
    for news in testBusiness:
        classOfNews = classifyNews(news, features, PTravel, PBusiness, PStyle)
        if(classOfNews == "BUSINESS"):
            numberOfCorrectBusiness = numberOfCorrectBusiness + 1
            allDetectedBusiness = allDetectedBusiness + 1
        if(classOfNews == "TRAVEL"):
            allBusinessDetectedTravel = allBusinessDetectedTravel + 1
            allDetectedTravel = allDetectedTravel + 1
        if(classOfNews == "STYLE & BEAUTY"):
            allBusinessDetectedStyle = allBusinessDetectedStyle + 1
            allDetectedStyle = allDetectedStyle + 1
    for news in testStyle:
        classOfNews = classifyNews(news, features, PTravel, PBusiness, PStyle)
        if(classOfNews == "BUSINESS"):
            allStyleDetectedBusiness = allStyleDetectedBusiness + 1
            allDetectedBusiness = allDetectedBusiness + 1
        if(classOfNews == "TRAVEL"):
            allStyleDetectedTravel = allStyleDetectedTravel + 1
            allDetectedTravel = allDetectedTravel + 1
        if(classOfNews == "STYLE & BEAUTY"):
            numberOfCorrectStyle = numberOfCorrectStyle + 1
            allDetectedStyle = allDetectedStyle + 1

    #For confusion matrix:
    print("Travel: \n", "travel detected travel: ", numberOfCorrectTravel, "travel detected business: ", allTravelDetectedBusiness, "travel detected style: ", allTravelDetectedStyle)
    print("Business: \n", "business detected travel: ", allBusinessDetectedTravel, "business detected business: ", numberOfCorrectBusiness, "business detected style: ", allBusinessDetectedStyle)
    print("Style: \n", "style detected travel: ", allStyleDetectedTravel, "style detected business: ", allStyleDetectedBusiness, "style detected style: ", numberOfCorrectStyle)
    return numberOfCorrectTravel, numberOfCorrectBusiness, numberOfCorrectStyle, allDetectedTravel, allDetectedBusiness, allDetectedStyle

def evaluateComputation(numberOfCorrectTravel, numberOfCorrectBusiness, numberOfCorrectStyle, numberOfTravelInTest, 
         numberOfBusinessInTest, numberOfStyleInTest, allDetectedTravel, allDetectedBusiness, allDetectedStyle, allTest):
    recall = numberOfCorrectTravel / numberOfTravelInTest
    recall2 = numberOfCorrectBusiness / numberOfBusinessInTest
    recall3 = numberOfCorrectStyle / numberOfStyleInTest
    percision = numberOfCorrectTravel / allDetectedTravel
    percision2 = numberOfCorrectBusiness / allDetectedBusiness
    percision3 = numberOfCorrectStyle / allDetectedStyle
    accuracy = (numberOfCorrectBusiness + numberOfCorrectTravel + numberOfCorrectStyle) / allTest
    print("Recall1: \n", recall * 100)
    print("Recall2: \n", recall2 * 100)
    print("Recall3: \n", recall3 * 100)
    print("Percision: \n", percision * 100)
    print("Percision2: \n", percision2 * 100)
    print("Percision3: \n", percision3 * 100)
    print("Accuracy: \n", accuracy * 100)

def evaluateModel( numOfTravelTrain, numOfBusinessTrain, numberOfStyleTrain, travelWords,
         businessWords, styleWords, features, PTravel, PBusiness, PStyle):
    travelTest, businessTest, styleTest = exractTestDataSet(len(travelWords), numOfTravelTrain, len(businessWords), numOfBusinessTrain,
             len(styleWords), numberOfStyleTrain, travelWords, businessWords, styleWords)
    numberOfCorrectTravel, numberOfCorrectBusiness, numberOfCorrectStyle, allDetectedTravel, allDetectedBusiness, allDetectedStyle = classifyValidationData(travelTest, businessTest, styleTest, features, PTravel, PBusiness, PStyle)
    evaluateComputation(numberOfCorrectTravel, numberOfCorrectBusiness, numberOfCorrectStyle, len(travelTest), 
            len(businessTest) ,len(styleTest), allDetectedTravel, allDetectedBusiness, allDetectedStyle, len(travelTest) + len(businessTest) + len(styleTest))


def classifyTest(filename, features, PTravel, PBusiness, PStyle):
    #print("INSIDE CLASSIFY::::::::::::::")
    fields = []
    output= pd.DataFrame()
    classes = []
    with open(filename, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        fields = next(csvreader)
        index = 0
        for nameOfColomn in fields:
            if(nameOfColomn == "headline"):
                headlineCol = index
            if(nameOfColomn == "link"):
                linkCol = index
            if(nameOfColomn == "short_description"):
                descriptionCol = index
            index = index + 1 
        for row in csvreader:
            numOfCol = 0
            for col in row:
                if(numOfCol == linkCol):
                    link = col
                    #print("LINK: ", link)
                if(numOfCol == headlineCol):
                    headline = col
                    #print("HEADLINE: ", headline)
                if(numOfCol == descriptionCol):
                    #print("DESCRIPTION: ", col)
                    news = " ".join((col, headline, link))
                    classes.append(classifyNews(news, features, PTravel, PBusiness, PStyle))
                numOfCol = numOfCol + 1
    for i in range(len(classes)):
	    output.at[i,0] = int(i)
	    output.at[i,1] = classes[i]
    o = pd.DataFrame(output.values.tolist() , columns  = ['index' , 'category'])
    script_dir = os.path.dirname(__file__)
    o.to_csv(path_or_buf= script_dir + '/Attachment/output-phase2-oversampled.csv', index= False)

script_dir = os.path.dirname(__file__)
filename = script_dir + "/Attachment/data.csv"
filenameTest = script_dir + "/Attachment/test.csv"
features, PTravel, PBusiness, PStyle, travelNews, businessNews, styleNews, numOfTravelTrain, numOfBusinessTrain, numberOfStyleTrain = creatModel(filename)
print("MODEL CREATED")
classifyTest(filenameTest, features, PTravel, PBusiness, PStyle)
evaluateModel(numOfTravelTrain, numOfBusinessTrain, numberOfStyleTrain, travelNews, businessNews, styleNews, features, PTravel, PBusiness, PStyle)