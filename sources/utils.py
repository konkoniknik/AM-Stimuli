import sys,os
import tensorflow as tf
import numpy as np
import time
import argparse
import random
import time
import matplotlib.pyplot as plt
import help_functions as h


#Show some stimuli:    
def show_image(X,Y,s="stimuli"):
    fig = plt.figure(s)
    for i in range(25):
        plt.subplot(5,5,i+1)
        plt.tight_layout()
        n=random.randint(0,len(Y)-1)
        pic=np.reshape(X[n,:],newshape=[28,28])
        plt.imshow(pic, cmap='gray', interpolation='none')
        plt.title("{}".format(np.argmax(Y[n,:])))
        plt.xticks([])
        plt.yticks([])
    plt.show()

def Train(model,trD,trL,tstD,tstL,valD,valL,N=20,evaluate=False):

    model.fit(trD, trL, epochs=N,batch_size=128,verbose=2)
    if(evaluate==True):
        preds_tst=model.predict(tstD)
        h.print_scores(np.argmax(preds_tst,1),np.argmax(tstL,1))


def init_weights(model, init,weights=None):
    if weights is None:
        weights = model.get_weights()
    weights = [init(w.shape).numpy() for w in weights]
    model.set_weights(weights)


#Calculates thresholds for each class based on the expectation (average) for the real data
#and mins per-class
def calculate_class_thresholds(model, X_Data, l_soft, n_classes=10):


    res = l_soft(model(X_Data)).numpy()
    a = [[] for _ in range(n_classes)]
    indx = np.argmax(res, 1)
    for i in range(len(indx)): a[indx[i]].append(res[i,indx[i]])

    print("Lengths and avgs:")
    means = []
    for i in range(n_classes): means.append(sum(a[i])/len(a[i]))
    

    return means



#plot histograms for datsapoints that belong to a given class
#(training data and stimuli)
def plot_belief_distr(model,l_soft,X_Data,y_Synth,cl=0,n_classes=10):
    
    y_Data = l_soft(model(X_Data)).numpy()
    
    print(y_Data)

    buckets_Real = [[] for _ in range(n_classes)]
    indx = np.argmax(y_Data, 1)
    for i in range(len(indx)): buckets_Real[indx[i]].append(y_Data[i,indx[i]])


    buckets_Synth = [[] for _ in range(n_classes)]
    indx = np.argmax(y_Synth, 1)
    for i in range(len(indx)): buckets_Synth[indx[i]].append(y_Synth[i,indx[i]])

    
    plt.hist(buckets_Real[cl],label="Real", bins=50)
    plt.hist(buckets_Synth[cl],label="Stimuli",bins=50)
    plt.xlabel("Max. output probability")
    plt.ylabel("# of samples")
    plt.title("Max. Belief Distribution for Class {}".format(cl))

    plt.legend(loc='upper left')
    plt.show()




def drop_high_belief(X_Synth,y_Synth,T_drop):

    T_drop=T_drop/10000
    print("T drop is:",T_drop)
    Xx,yy=[],[]
    for x,y in zip(X_Synth,y_Synth):
        if(np.max(y)<T_drop):
            Xx.append(x)
            yy.append(y)
            print("drop high belief: ",y)
    return np.array(Xx),np.array(yy)
