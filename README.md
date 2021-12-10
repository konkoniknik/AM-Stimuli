# AM-Stimuli
Some Proof-of-Concept code for AM stimuli generation on MNIST (JAIR 2021, Volume 72, "Learning Realistic Patterns from Visually Unrealistic Stimuli: Generalization and Data Anonymization", https://arxiv.org/abs/2009.10007).  We extract information from a Teacher network in the form of a stimuli dataset, and  we then train a student network with this dataset.
 
![Image 1](/assets/images/adversarial_v5.png)


## Stimuli 

We perform several proof-of-concept experiments to evaluate the approach. We use similar, but deeper, Student/Teacher architectures to the ones we used for our paper experiments, and extract stimuli from the Teacher h_T. We use various threshold values for the maximum probability of the output of h_T and evaluate with an identical Student architeture. 
