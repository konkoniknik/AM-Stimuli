import sys,os
import tensorflow as tf
import numpy as np
import time
from sources.help_functions import print_scores, num_to_cat
from sources.models import create_Classifier, create_generator
from sources.utils import Train, plot_belief_distr, init_weights, calculate_class_thresholds, drop_high_belief, show_image

import argparse
import random
import time
import matplotlib.pyplot as plt

n_classes=10
def G_loss(s_c_e,givenClass,output):
      
    return s_c_e(givenClass, output)



def AM_Sample(model, generator,cl,dT,l_soft,sce,optimizer,b_G=32):

    batch_yy=np.array([cl for _ in range(b_G)]) 
    pos=-1

    noise = tf.random.normal([b_G, 512])

    for cnt in range(50):
        with tf.GradientTape() as gen_tape:

            generated_samples = generator(noise, training=True)

            output = model(generated_samples, training=False)
            
            gen_loss = G_loss(sce,batch_yy,output)
           


        gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
        optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))

        l=l_soft(model(generator(noise))).numpy()

        #print("l:",l.shape,l)
        #if we find one candidate we break
        for i in range(b_G):
            if(l[i,cl]>dT):
                pos=i
                break

        if(pos>=0): break

    #print("cnt,pos",cnt,pos,"l",l[pos,cl],end=" ")
    return l[pos,:],generator(noise).numpy()[pos,:]


#T_min=7500#9600# Or we can use something like Calculate_ClassThresholds(Teacher,X_train[:10000],l_softmx)
# for a much stricter per-class threshold. 
# Default T_max used =9999
def generate_stimuli(Teacher,generator,l_softmx,sce,g_optimizer,initializer,Stimuli_Count,batch_Gen=1,T_min=7500,T_max=9999):
    print("T min/max:",T_min,T_max, "Stimuli Count:",Stimuli_Count) 
    X_S,y_S=[],[]
    cnt_f,cnt_s=0,0
    success_str=""

    start_time=time.time()
    while cnt_s < Stimuli_Count:
        
        #How often shall we reinit?
        q=random.randint(0,500)
       
            
        if((success_str=="FAIL")):
            print("REINIT")
            init_weights(generator, initializer)
            

        random_class=random.randint(0,n_classes-1)

        #print(int(1000*T_max[cl]),cl)
        #T=random.randint(T_min,int(1000*T_max[random_class]))/1000
        
        T=random.randint(T_min,T_max)/10000

        ls,gen_datum=AM_Sample(Teacher,generator,random_class,T,l_softmx,sce,g_optimizer,batch_Gen)
        

        if(ls[random_class]>T):
            success_str="SUCCESS"
            X_S.append(gen_datum)
            y_S.append(ls)
            cnt_s+=1
        else:
            success_str="FAIL"
            cnt_f+=1

        print(cnt_s,random_class,np.argmax(ls),success_str,np.amax(ls),T,np.amax(gen_datum))
        


    X_S=np.array(X_S)
    y_S=np.array(y_S)
    print("Synth.shape:",X_S.shape,"Time:",time.time()-start_time,"Fails:",cnt_f)

    return X_S,y_S

   
 


def run_test(N_T,Stimuli_N,T_min,T_max,N_S,T_drop):
    #Extract MNIST
    (X_train, y_train), (X_test, y_test)=tf.keras.datasets.mnist.load_data(path='mnist.npz')

    #Rescale to 0-1 and reshape
    X_train=X_train/255
    X_test=X_test/255

    X_train=np.reshape(X_train,[-1,28,28,1])
    X_test=np.reshape(X_test,[-1,28,28,1])


    print("Data shape",X_train.shape,np.amax(X_train),np.amin(X_train))


    #Create validation set after corruption: Assumes corrupted val set
    X_val,y_val=X_train[55000:,],y_train[55000:]
    X_train,y_train=X_train[:55000,],y_train[:55000]
 
    #Transform to one hot vectos
    train_Categories = num_to_cat(y_train,n_classes)
    test_Categories = num_to_cat(y_test,n_classes)
    val_Categories = num_to_cat(y_val,n_classes)

    print("Create and Train a  Teacher Classifier....")
    Teacher = create_Classifier()
    Teacher.compile(optimizer='adam',
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


    #Set up the initializer. We ll need it for restarting our model
    initializer = tf.keras.initializers.GlorotUniform()
    l_softmx=tf.keras.layers.Softmax()

    #Train the teacher
    Train(Teacher,X_train,train_Categories,None,None,X_val,val_Categories,N=N_T)
    Teacher.summary()


    #Create the AM-generator
    AM_generator = create_generator()
    AM_generator.summary()
    g_optimizer = tf.keras.optimizers.Adam(0.001)#0.001)

  
    sce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
   
    T_mean_train=calculate_class_thresholds(Teacher,X_train[:10000],l_softmx)
    print("Mean train data thresholds per class are:",T_mean_train)    

    #Generate the stimuli
    X_Synth,y_Synth = generate_stimuli(Teacher,AM_generator,l_softmx,sce,g_optimizer,initializer,Stimuli_N,T_min=T_min,T_max=T_max)
   
    #Do we want to drop "high belief" data (i.e., low entropy or high softmax)?
    if(T_drop!=-1): X_Synth,y_Synth = drop_high_belief(X_Synth,y_Synth,T_drop)
    print("X, y shapes:",X_Synth.shape,y_Synth.shape)

    
    #Plot the belief distribution of stimuli and real data for a given class (class 0 in this case).
    #Same number of data for real and synth data
    plot_belief_distr(Teacher,l_softmx,X_train[:len(y_Synth)],y_Synth,cl=0)


    print("Create and Train a  Student Classifier....")
    Student = create_Classifier()
    y_preds = Student.predict(X_test)
    
    #Just a dummy test to verify that student is new and untrained
    print_scores(np.argmax(y_preds,1),y_test)

    Student.compile(optimizer='adam',
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
    #Train the Student on the stimuli
    Train(Student,X_Synth,y_Synth,X_test,test_Categories,X_val,val_Categories,N=N_S,evaluate=True)
    #Student.summary()

 
    show_image(X_Synth,y_Synth)



############################################################################

parser = argparse.ArgumentParser(description='Configs for the AM stimuli generation and training')
parser.add_argument('--GPU', metavar='int', required=False,
                        help='which GPU shall we run?(starting from 0)')


parser.add_argument('--N_T', metavar='int', required=False,
                        help='how many epochs should we train the Teacher (#epochs)')
parser.add_argument('--Stimuli_N', metavar='path', required=False,
                        help='how many stimuli should we generate?')

parser.add_argument('--T_min', metavar='int', required=False,
                        help='Minimum Threshold (times 10000.. for example if we want a threshold of 0.98345, we give 9834 or 9835)')

parser.add_argument('--T_max', metavar='int', required=False,
                        help='Max Threshold (times 10000.. for example if we want a threshold of 0.98345, we give 9834 or 9835)')

parser.add_argument('--N_S', metavar='int', required=False,
        help='how many epochs should we train the Student? (Optional, Default: ~8K batch iters, #epochs)')


parser.add_argument('--T_drop', metavar='int', required=False,
        help='Do we want to drop stimuli above a certain threshold? If we want that specify threshold (times 10000.. for example if we want a threshold of 0.98345, we give 9834 or 9835)')


args = parser.parse_args()

if args.GPU!=None:
    os.environ["CUDA_VISIBLE_DEVICES"]=args.GPU


N_T=3 if args.N_T==None else int(args.N_T)
Stimuli_N=1000 if args.Stimuli_N==None else int(args.Stimuli_N)# We used stimuli N of 1000 and 7500 for the code of this experiments
T_min= 7500 if args.T_min==None else int(args.T_min)
T_max= 9999 if args.T_max==None else int(args.T_max)
N_S=130 if args.N_S==None else int(args.N_S)#~equivalent steps as 1000 epochs of 1000 stimuli dataset for batch_size=128
T_drop=-1 if args.T_drop==None else int(args.T_drop)


run_test(N_T,Stimuli_N,T_min,T_max,N_S,T_drop)

