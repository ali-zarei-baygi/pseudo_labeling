import json
import random

import oyaml as yaml
import pandas as pd

from helper import (augmentation_char, augmentation_ppdb, augmentation_T5,
                    ave_embedding_per_class_MiniLM, check_cosine,
                    compare_cosine, convert_psuedo_to_dict, dataframe_to_dict,
                    dict_to_yaml, extract_psuedo_ave_class_threshold,
                    finetune_classifier, m_random_from_dict,
                    merge_two_dictionary, new_samples_for_each_class,
                    remove_psuedo_from_unlabeled, remove_value_from_dict,
                    test_accuracy, test_classifier, train_classifier)

"""
################################################################################################
#######################################  Approach 1  ###########################################

In this approach, we add a number of pseudo labeled samples to the initial training data and 
finetune the classifier using this new training set
We are using two methods for selecting these pseudo labeled samples:
Method 1: will select the pseudo labeled samples only if their confidence is more than a 
          Threshold
Method 2: will select the pseudo labeled samples if their confidence is more than the average 
          confidence of their class

************************************************************************************************
#######################################  Approach 2  ###########################################

In this approach, we use "paraphrase-MiniLM-L6-v2" model to generate sentence embedding for each 
selected pseudo labeled sample. We will add a selected pseudo labeled sample to a class in the 
initial training data only if its sentence embedding is similar to the average sentence embedding
of the initial training data of that class. We are using cosine similarity to compare the sentence 
embeddings. We are using two methods for checking the cosine similarity and selecting new pseudo 
labeled samples:
Method 1: will select the pseudo labeled samples only if the cosine similarity is larger than a 
          Threshold
Method 2: will compare the cosine similarity between the pseudo labeled sample and all classes in 
          the initial training data, and only if its cosine similarity with its assigned class
          is the largest, it will be selected to be added to the initial training data

************************************************************************************************
#######################################  Approach 3  ###########################################

In this approach, which is taken from FixMatch algorithm, we augment the selected pseudo labeled
samples, and then add them to the initial training data, and finetune the classifier using this
new training set. We used three different methods for augmenting the selected pseudo samples:
Method 1: Character Augmentation
Method 2: Synonym Replacement
Method 3: Paraphrase Generation Using T5
"""
################################################################################################
#######################################  Method Selection  #####################################
"""
You can use a combination of different approaches. To select a method, we start from Approach 1. 
This approach must be selected. You can choose between "Approach1_Method1" and "Approach1_Method2"
You can then, combine Approach 1 with Approach 2 and/or Approach 3. 
"""
# Approach 1: Selecting Pseudo Labeled Samples
Approach1_Method1 = True    # based on Threshold
Approach1_Method2 = False   # based on average class confidence

# Approach 2: Check for Cosine Similarity
Approach2_Method1 = False    # based on Threshold
Approach2_Method2 = True   # based on comparison

# Approach 3: Augmentation of Selected Pseudo Samples
Approach3_Method1 = False   # based on Character Augmentation
Approach3_Method2 = True    # based on Synonym Replacement
Approach3_Method3 = False   # based on Paraphrase Generation Using T5

################################################################################################
################################################################################################

######################################## hyperparameters #######################################
number_training_data = 12 # number of initial samples per class for training
loops = 20 # number of loops to finetune the model and add pseudo labels to the initial training data
epoch_fraction = 0.3 # the model will be traind in 100 epoches. But it will be finetune in a feaction 
                     # of 100 epochs (e.g., if epoch fraction is 0.5, the number of epochs during
                     # finetuning will be 50)
num_new_sample_per_class = 5 # number of pseudo labeled samples per class which will be added to
                             # training data in each loop
confidence_threshold = 0.9 # only if the confidence of a psuedo label is more than the confidence
                            # threshold it will be added to the training data
cosine_threshold = 0.7 # only if the cosine similarity is more than the cosine threshold it will be
                       # added to the training data
################################################################################################

# the directory where all files are located
directory = '/Users/saramohamadi/ali/rasa/test_helper'

# read all labeled data from a dataframe(csv file) and save them in a dictionary
dict_all_labeled_data = dataframe_to_dict(directory + '/all_labeled_data.csv')

# split labeled data to train and test set (for each class we will use m (number_training_data) sample
# for training data and keep the rest for the test set)
dict_training_data = m_random_from_dict(dict_all_labeled_data, number_training_data)
dict_test_data = remove_value_from_dict(dict_training_data, dict_all_labeled_data)

# convert training and test set dictionary to yaml file (for RASA) and asve them in the directory
dict_to_yaml(dict_training_data, save_to = directory + '/training_data.yml')
dict_to_yaml(dict_test_data, save_to = directory + '/test_data.yml')

# read all unlabeled data from a dataframe (csv file) and convert them to a dictionary
dict_unlabeled_data = dataframe_to_dict(directory + '/unlabeled_data.csv')

# train the calssifier with initial training data
train_classifier(directory)

# test the primary classifier
test_classifier(directory, on_unlabel_data=False)

# print the test accuracy
print('##############################################################################')
print('The primary test accuracy is ', test_accuracy(directory + '/test_results'))
print('##############################################################################')

if Approach2_Method1==True or Approach2_Method2==True:
    ave_emb_per_class = ave_embedding_per_class_MiniLM(dict_training_data)

# psuedo labeling loop
for i in range(loops):

    # convert the dictionary of unlabeled data from to a yaml file and save it in the directory
    dict_to_yaml(dict_unlabeled_data, save_to = directory + '/unlabeled_data.yml')

    # test classifier with unlabeled data to give them pseudo labels
    test_classifier(directory, on_unlabel_data=True)

    # select pseudo labels based on Approach 1 method 1 or 2 and save them in a dictionary
    if Approach1_Method1==True:
        psuedo_dict = convert_psuedo_to_dict(directory, confidence_threshold)
    if Approach1_Method2==True:
        psuedo_dict = extract_psuedo_ave_class_threshold(directory)
    
    # select a number of (num_new_sample_per_class) pseudo labeled samples with highest confidence
    # from each class 
    dict_new_samples = new_samples_for_each_class(psuedo_dict, num_new_sample_per_class)

    # check for the cosine similarity using Approach 2 Method 1 or 2 (samples that cannot pass the
    # cosine check will be removed)
    if Approach2_Method1==True:
        check_cosine(dict_new_samples, ave_emb_per_class, cosine_threshold)
    if Approach2_Method2==True:
        compare_cosine(dict_new_samples, ave_emb_per_class)

    # remove the selected psuedo label samples form unlabeled data (this will automatically remove
    # selected data from unlabeled data dictionary)
    remove_psuedo_from_unlabeled(dict_new_samples, dict_unlabeled_data)

    # augment selected pseudo labeled samples using Approach 3 Method 1, 2, or 3
    if Approach3_Method1==True:
        augmentation_char(dict_new_samples)
    if Approach3_Method2==True:
        augmentation_ppdb(dict_new_samples)
    if Approach3_Method3==True:
        augmentation_T5(dict_new_samples)

    # merge selected pseudo labeled samples and initial training data to generate the new training set
    dict_new_training_data = merge_two_dictionary(dict_new_samples, dict_training_data)

    # convert the new training set from a dictionary to yaml file and save it in the directory
    dict_to_yaml(dict_new_training_data, save_to= directory + '/training_data.yml')
    
    # finetune the model using the new training set in a epoch_fraction number of epochs
    finetune_classifier(directory, epoch_fraction)

    # test the finetuned classifier to check how it is performing
    test_classifier(directory, on_unlabel_data=False)

    # print the test accuracy
    print('##############################################################################')
    print('The test accuracy is ', test_accuracy(directory + '/test_results'))
    print('##############################################################################')
