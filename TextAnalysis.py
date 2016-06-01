"""
Restaurant Review Analysis
"""

import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from nltk.stem.porter import PorterStemmer
#from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
#from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from numpy import array
from sklearn.externals import joblib
import os.path

# Read the reviews and their polarities
def loadReviews(fname):
    
    reviews=[]
    polarities=[]
    f=open(fname)
    for line in f:
        try:
            review,rating=line.strip().split('\t')  
        except:
            print(line)
            break
        reviews.append(review.lower())    
        polarities.append(int(rating))
    f.close()

    return reviews,polarities


def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

def tokenize(text):
    tokens = nltk.word_tokenize(text)
    stemmer=PorterStemmer()    
    stems = stem_tokens(tokens, stemmer)
    return stems    

    
rev_train,pol_train=loadReviews('Reviews.txt')
rev_test,pol_test=loadReviews('TestFile.txt')


vectorizer = TfidfVectorizer(min_df=3,
                                 max_df = 0.7,
                                 sublinear_tf=True,
                                 use_idf=True, ngram_range=(1, 2),
                    tokenizer=tokenize, 
                    token_pattern=u'(?u)\b\w\w+\b') #, tokenizer=tokenize, token_pattern=u'(?u)\b\w\w+\b'
  

# Generates train and test vectors for Training and Test Dataset
train_vectors = vectorizer.fit_transform(rev_train)
test_vectors = vectorizer.transform(rev_test)

# This is the function which model the dataset using SVM
# This function also predicts the accuracy of the model when fitted on test data.
def SVM_Classifier():
    classifier_svm = svm.LinearSVC()
    if os.path.exists('filename.pkl'):
       classifier_svm = joblib.load('filename.pkl')
    else:
        classifier_svm.fit(train_vectors, pol_train)
        joblib.dump(classifier_svm, 'filename.pkl')
    prediction_svm = classifier_svm.predict(test_vectors)
    return prediction_svm
    
# This is the function which models the dataset using Multinomial Naive Bayes
# This function also predicts the accuracy of the model when fitted on test data.
def MNB_Classifier():
    classifier_MNB = MultinomialNB()
    if os.path.exists('filename1.pkl'):
       classifier_MNB = joblib.load('filename1.pkl')
    else: 
        classifier_MNB.fit(train_vectors, pol_train)
        joblib.dump(classifier_MNB, 'filename1.pkl')
    prediction_MNB = classifier_MNB.predict(test_vectors)
    return prediction_MNB
    
# This is the function which models the dataset using Logistic Rehression
# This function also predicts the accuracy of the model when fitted on test data.
def LR_Classifier():
    classifier_LR = LogisticRegression()
    if os.path.exists('filename2.pkl'):
        classifier_LR = joblib.load('filename2.pkl')
    else:
        classifier_LR.fit(train_vectors, pol_train)
        joblib.dump(classifier_LR, 'filename2.pkl')
    prediction_LR = classifier_LR.predict(test_vectors)
    return prediction_LR

#This fuction is used for printing the final project
def Print_Results():
    print("Results for LinearSVC")
    prediction_svm = SVM_Classifier()
    print('ACCURACY:\t',accuracy_score(pol_test, prediction_svm))
    print('PREDICTED:',prediction_svm)
    print('CORRECT:\t', array(pol_test))
    
    print("Results for MultinomialNB")
    prediction_MNB = MNB_Classifier()
    print('ACCURACY:\t',accuracy_score(pol_test, prediction_MNB))
    print('PREDICTED:',prediction_MNB)
    print('CORRECT:\t', array(pol_test))
    
    print("Results for Logistic_Regression")
    prediction_LR = LR_Classifier()
    print('ACCURACY:\t',accuracy_score(pol_test, prediction_LR))
    print('PREDICTED:',prediction_LR)
    print('CORRECT:\t', array(pol_test))


def main():
    Print_Results()
    
if __name__ == "__main__":
   main()
    




