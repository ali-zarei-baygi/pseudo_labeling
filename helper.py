import asyncio
import json
import random
from collections import defaultdict

import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
import numpy as np
import oyaml as yaml
import pandas as pd
from rasa.model_testing import test_nlu
from rasa.model_training import train_nlu
from scipy.spatial.distance import cosine
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
                          


def train_classifier(directory):
    # training the classifier on the data
    print("\n******* Start training classifier...")
    train_nlu(config=directory + '/config.yml',
              nlu_data=directory + '/training_data.yml',
              output=directory,
              fixed_model_name='nlu',
              persist_nlu_training_data=False,
              additional_arguments=None,
              domain=directory +'/domain.yml',
              model_to_finetune=None,
              finetuning_epoch_fraction=1.0)

def finetune_classifier(directory, epoch_fraction):
    # finetne a model on the data
    print("\n******* Start finetuning classifier with new training data...")
    train_nlu(config=directory + '/config.yml',
              nlu_data=directory + '/training_data.yml',
              output=directory,
              fixed_model_name='nlu',
              persist_nlu_training_data=False,
              additional_arguments=None,
              domain=directory +'/domain.yml',
              model_to_finetune= directory + '/nlu.tar.gz',
              finetuning_epoch_fraction=epoch_fraction)

def test_classifier(directory, on_unlabel_data):
    if on_unlabel_data:
        path = '/unlabeled_data.yml'
        results_dir = directory
        print("\n******* Start generating Psuedo labels...")
    else:
        path = '/test_data.yml'
        results_dir = directory + '/test_results'
        print("\n******* Start testing classifier...")
    loop = asyncio.get_event_loop()
    loop.run_until_complete(test_nlu(
        model=directory + '/nlu.tar.gz',
        nlu_data=directory + path,
        output_directory=results_dir,
        additional_arguments={'intent_ranking': True,  # If true successful predictions are written to a file
                              'errors': True,  # If true incorrect predictions are written to a file
                              'disable_plotting': False,
                              # If true confusion matrix and histogram will not be rendered
                              'report_as_dict': True  # if true the evaluation report will be returned as dict
                              }
        ))

def convert_psuedo_to_dict(directory, threshold):
    with open(directory + '/intent_errors.json') as file:
        contents = json.load(file)
    contents = sorted(contents, key=lambda contents:contents['intent_prediction']['confidence'], reverse=True)
    intent_dict = {}
    for item in contents:
        if item['intent_prediction']['name'] not in intent_dict:
            intent_dict[item['intent_prediction']['name']] = []
        if item['intent_prediction']['confidence']>threshold:
            intent_dict[item['intent_prediction']['name']].append(item['text'])
    return intent_dict

def new_samples_for_each_class(psuedo_label_dict, number_of_new_samples):
    new_samples = {}
    for intent in psuedo_label_dict:
        l = len(psuedo_label_dict[intent])
        if l>=number_of_new_samples:
            n = number_of_new_samples
        else:
            n = l
        new_samples[intent] = psuedo_label_dict[intent][:n]
    return new_samples

def compute_average_confidence(directory):
    with open(directory + '/intent_errors.json') as file:
        contents = json.load(file)
    intent_avg_confidence_dict = {}
    intent_count_dict = {}
    sum_confidence = 0
    num_items = 0
    for item in contents:
        if item['intent_prediction']['name'] not in intent_avg_confidence_dict:
            intent_avg_confidence_dict[item['intent_prediction']['name']] = 0
            intent_count_dict[item['intent_prediction']['name']] = 0
        intent_avg_confidence_dict[item['intent_prediction']['name']] += item['intent_prediction']['confidence']
        intent_count_dict[item['intent_prediction']['name']] += 1
        sum_confidence += item['intent_prediction']['confidence']
        num_items += 1
    for intent in intent_avg_confidence_dict:
        intent_avg_confidence_dict[intent] /= intent_count_dict[intent]
    ave_confidence = sum_confidence / num_items
    return intent_avg_confidence_dict, ave_confidence

def test_accuracy(directory):
    with open(directory + '/intent_report.json') as file:
        contents = json.load(file)
    return contents['accuracy']
    
def extract_new_sample_threshold(directory, threshold):
    with open(directory + '/intent_errors.json') as file:
        contents = json.load(file)
    new_samples = {}
    for item in contents:
        if item['intent'] != 'default2':
            if item['intent_prediction']['confidence']>threshold:
                if item['intent_prediction']['name'] not in new_samples:
                    new_samples[item['intent_prediction']['name']]=[]
                new_samples[item['intent_prediction']['name']].append(item['text'])
    return new_samples

def extract_new_sample_rating(directory, sampling_rate):
    with open(directory + '/intent_errors.json') as file:
        contents = json.load(file)
    contents = sorted(contents, key=lambda contents:contents['intent_prediction']['confidence'], reverse=True)
    len_new_samples = int(len(contents) * sampling_rate)
    new_samples = {}
    for i in range(len_new_samples):
        if contents[i]['intent_prediction']['name'] not in new_samples:
            new_samples[contents[i]['intent_prediction']['name']]=[]
        new_samples[contents[i]['intent_prediction']['name']].append(contents[i]['text'])
    return new_samples

def remove_psuedo_from_unlabeled(psuedo, unlabeled):
    for key in psuedo:
        for utterance in psuedo[key]:
            if utterance in unlabeled['default']:
                unlabeled['default'].remove(utterance)

def remove_value_from_dict(dict1, dict2):
    for key in dict1:
        for utterance in dict1[key]:
            if utterance in dict2[key]:
                dict2[key].remove(utterance)
    return dict2

def m_random_from_dict(dict, m):
    random.seed(1)
    selected_samples = {}
    for key in dict:
        selected_samples[key] = random.sample(dict[key], m)
    return selected_samples

def dataframe_to_dict(directory):
    data = pd.read_csv(directory)
    utterances=data['utterances'].tolist()
    labels=data['labels'].tolist()
    dict_samples = {}
    for i in range(len(labels)):
        if labels[i] not in dict_samples:
            dict_samples[labels[i]]=[]
        dict_samples[labels[i]].append(utterances[i])
    return dict_samples

def dict_to_yaml(dict, save_to):
    yaml_data = {'version': '3.0', 'nlu':[]}
    j = 0
    for key in dict:
        yaml_data['nlu'].append({'intent':key, 'examples':''})
        for i in range(len(dict[key])):
            yaml_data['nlu'][j]['examples'] += ("- " + dict[key][i] + "\n")
        j += 1
    ##########################  yaml Format  #####################################
    def str_presenter(dumper, data):
        if len(data.splitlines()) > 1 or '\n' in data:  
            text_list = [line.rstrip() for line in data.splitlines()]
            fixed_data = "\n".join(text_list)
            return dumper.represent_scalar('tag:yaml.org,2002:str', fixed_data, style='|')
        return dumper.represent_scalar('tag:yaml.org,2002:str', data)
    yaml.add_representer(str, str_presenter)
    # to use with safe_dump:
    yaml.representer.SafeRepresenter.add_representer(str, str_presenter)
    ###############################################################################
    with open(save_to, 'w') as file:
        yaml.dump(yaml_data, file)

def merge_two_dictionary(dict_1, dict_2):
    merged_dict = {}
    for key in dict_2:
        merged_dict[key] = []
        for utterance in dict_2[key]:
            merged_dict[key].append(utterance)
    for intent in dict_1:
        if intent in merged_dict:
            for utterance in dict_1[intent]:
                merged_dict[intent].append(utterance)
    return merged_dict

def extract_psuedo_ave_class_threshold(directory):
    ave_conf_per_class, _ = compute_average_confidence(directory)
    with open(directory + '/intent_errors.json') as file:
        contents = json.load(file)
    contents = sorted(contents, key=lambda contents:contents['intent_prediction']['confidence'], reverse=True)
    intent_dict = {}
    for item in contents:
        if item['intent_prediction']['name'] not in intent_dict:
            intent_dict[item['intent_prediction']['name']] = []
        if item['intent_prediction']['confidence']>ave_conf_per_class[item['intent_prediction']['name']]:
            intent_dict[item['intent_prediction']['name']].append(item['text'])
    return intent_dict


def ave_embedding_per_class_MiniLM(dict):
    dict_ave_embedding = {}
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')  
    for intent in dict:
        dict_ave_embedding[intent] = np.mean(model.encode(dict[intent]), axis=0)
    return dict_ave_embedding

def cosine_similarity(vec1, vec2):
    return (1 - cosine(vec1, vec2))

def check_cosine(to_be_checked, against, threshold):
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    count_all = 0
    count_removed = 0
    for intent in to_be_checked:
        for utterance in to_be_checked[intent]:
            count_all += 1
            utter_emb = model.encode(utterance)
            if cosine_similarity(utter_emb, against[intent])<threshold:
                to_be_checked[intent].remove(utterance)
                count_removed += 1
    print(count_removed, ' out of ', count_all, 'utterances were removed from the new psuedo samples because of low similarity')

def compare_cosine(to_be_checked, against):
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    count_all = 0
    count_removed = 0
    for intent in to_be_checked:
        for utterance in to_be_checked[intent]:
            count_all += 1
            utter_emb = model.encode(utterance)
            cos_sim_to_all_intent = []
            for key in against:
                cos_sim_to_all_intent.append(cosine_similarity(utter_emb, against[key]))
            if cosine_similarity(utter_emb, against[intent])!=max(cos_sim_to_all_intent):
                to_be_checked[intent].remove(utterance)
                count_removed += 1
    print(count_removed, ' out of ', count_all, 'utterances were removed from the new psuedo samples because of low similarity')


def augmentation_ppdb(dict_data):
    print("\n******* Start generating augmented data...")
    aug = naw.SynonymAug(aug_src='ppdb', model_path='/Users/saramohamadi/ali/rasa/test_helper/ppdb-2.0-s-all')
    for intent in dict_data:
        dict_data[intent] = aug.augment(dict_data[intent])

def augmentation_T5(dict_data):
    print("\n******* Start generating augmented data...")
    tokenizer = AutoTokenizer.from_pretrained("Vamsi/T5_Paraphrase_Paws")  
    model = AutoModelForSeq2SeqLM.from_pretrained("Vamsi/T5_Paraphrase_Paws")
    print_count = 0
    for intent in dict_data:
        for utterance in dict_data[intent]:
            text = "paraphrase: " + utterance + " </s>"
            encoding = tokenizer.encode_plus(text,padding='longest', return_tensors="pt")
            input_ids, attention_masks = encoding["input_ids"].to("cpu"), encoding["attention_mask"].to("cpu")
            outputs = model.generate(
            input_ids=input_ids, attention_mask=attention_masks,
            max_length=256,
            do_sample=True,
            top_k=120,
            top_p=0.95,
            early_stopping=True,
            num_return_sequences=1
            )
            for output in outputs:
                utterance = tokenizer.decode(output, skip_special_tokens=True,clean_up_tokenization_spaces=True)

def augmentation_char(dict_data):
    print("\n******* Start generating augmented data...")
    aug = nac.KeyboardAug()
    for intent in dict_data:
        dict_data[intent] = aug.augment(dict_data[intent])
