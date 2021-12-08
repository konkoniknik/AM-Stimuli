import numpy as np
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score,precision_score,roc_curve,auc, accuracy_score,confusion_matrix,cohen_kappa_score
import matplotlib
import matplotlib.pyplot as plt
import random

#Most of these are legacy functions which arent being used
def show_Data(X,y):

    x1 = np.linspace(0, X.shape[1], X.shape[1])

    fig,axes = plt.subplots(nrows=10 ,ncols=10,figsize=(10,10))
    plt.setp(axes,yticks=[],xticks=[],xlabel="ssss")
    for i in range(10):
        for j in range(10):
            q=random.randint(0,X.shape[0])
            axes[i, j].plot(x1, X[q,])
            plt.sca(axes[i,j])
            plt.xlabel(str(y[q]))
            plt.ylabel(str(np.amin(X[q]))[:5]+" "+str(np.amax(X[q]))[:4])

    fig.tight_layout()
    plt.show()



def num_to_cat(y,sz):
    temp=np.zeros([len(y),sz])
    for i in range(len(y)):
        for j in range(sz):
            if(j==y[i]):
                temp[i,j]=1

    return temp


def plot_auc(test_L,y_scores):
	fpr, tpr, _ = roc_curve(test_L, y_scores)
	roc_auc = auc(fpr, tpr)
	plt.figure()
	lw = 2
	plt.plot(fpr, tpr, color='darkorange',lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
	plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Receiver operating characteristic example')
	plt.legend(loc="lower right")
	plt.show()


def kappa(t_L,preds):
	a=sum(t_L)
	ca=sum(preds)

	na=len(t_L)-a
	cna=len(preds)-ca

	aca=a*ca
	nacna=na*cna


	acaDnl = aca/ len(t_L)
	nacnaDnl = nacna/len(t_L)

	pexp=(acaDnl+nacnaDnl)/float(len(t_L))


	kappa=(accuracy_score(preds,t_L) -pexp)/(1-pexp)

	return kappa



def get_Batch(X,Y,sz):
    perm=[(random.randint(0,X.shape[0])-1) for j in range(sz)]
    batch_x=X[perm,:]
    batch_y=Y[perm]

    return batch_x,batch_y

def print_scores(preds,labels,prnt='on'):
    
    if(np.amax(labels)<=1):
        con=confusion_matrix(labels,preds)
        spec=con[0,0]/(con[0,1]+con[0,0])
        sens=con[1,1]/(con[1,0]+con[1,1])
        acc=accuracy_score(preds,labels)
        prec=precision_score( labels,preds)
        kappa1=kappa(labels,preds)
        if(prnt=='on'):print("Binary: Spec:",spec,"Sens:",sens,"Prec:", prec,"Acc:",acc,"Kappa:",kappa1)
        return spec,sens, prec,acc,kappa1
    
    else:
    
        k=cohen_kappa_score(labels,preds)
        acc=accuracy_score(labels,preds)
        #kappa1=kappa(labels,preds)
        if(prnt=='on'):print("Multi: Acc:",acc,"k:",k)
        return  acc,0,0,0,k


def shuffle(X,Y,Z=[]):
    perm=np.random.permutation(X.shape[0])
    X=X[perm]
    Y=Y[perm]
    if(Z==[]):
        return X,Y
    else:
        Z=Z[perm]
        return X,Y,Z




def transf(x):
    try:
        return float(x)
    except ValueError:
        return 0  # np.nan


def clearNaN(x):
	 # <TODO>: change  the NaN values with the average of the previous and the next of the array instead of putting zeros on transf
    return 0


def normalize(X):
    x_mean = np.mean(X)
    x_std = np.std(X)
    return (X - x_mean) / x_std


def scale(x):
    xmin = x.min()
    xmax = x.max()

    x_tmp = (x ) / (xmax - xmin)
    return x_tmp



def downsample(inp_col, f, N):
    inp_col_float = [transf(i) for i in inp_col]
    inp_matr = np.reshape(inp_col_float, (-1, f))

    t = np.mean(inp_matr, axis=1)

    t1 = np.reshape(t, (N, -1))

    # print np.size(t1,0),np.size(t1,1),t1[0,:]
    return t1


def nextBatch(Data, start, finish):
    return Data[start:finish, :]

def split_TT(X,Y,r):
        len_X=np.shape(X)[0]
        n=int(r*len_X)
        print("Rows in X, n:",len_X,n)
        train_X,train_Y=X[:n,:],Y[:n]
        test_X,test_Y=X[n:,:],Y[n:]
        return train_X,train_Y,test_X,test_Y




#generate a random subsample based only on apneic events (this is why we need the labels Y)
def RandomSubsampleCond(X,Y,size1,c):
    Y_Appos=[i for i in range(len(Y)) if(Y[i]==c)]
    #print("yapos:",Y_Appos)
    X_Appos=X[Y_Appos,:]

    return X_Appos[np.random.choice(len(Y_Appos),size=size1),:]

def RandomSubsample(X,Y,size1,Z=np.zeros(1)):
    perm=np.random.permutation(len(Y))
    #print(*perm)
    perm_test=perm[:size1]
    perm_train=perm[size1:]

    #print((perm_train),len(perm_test))

    tr_Data=X[perm_train]
    tr_Data_L=Y[perm_train]

    tst_Data=X[perm_test]
    tst_Data_L=Y[perm_test]

    if(Z.shape[0]>1):
        print("RandomSubsample: Using Z...")
        tr_Z=Z[perm_train]
        tst_Z=Z[perm_test]
        return  tr_Data,tr_Data_L,tst_Data,tst_Data_L,tr_Z,tst_Z
    else:
        return tr_Data,tr_Data_L,tst_Data,tst_Data_L




