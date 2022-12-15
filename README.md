# pseudo_labeling

One of the main challenges of creating a chatbot in real world is to annotate the data. This data is usually collected from the conversations that happens between two persons: one as a user and the other one as an agent. After collecting these dialogues, they need to be annotated by humans. The intent of each dialogue is one of the most important annotation which is used for intent classification; an essential portion of chatbot creation.<br>
In this project, we used MultiWOZ 2.2 dataset [1], which is a large-scale human-human conversational corpus spanning over eight domains: Hotel, Restaurant, Taxi, Train, Attraction, Hospital, Bus, and Police. Although this dataset is annotated, the intents (active intent in MultiWOZ 2.2 dataset) are assigned widely on the conversations. For example, all sentences in a conversation around “booking table in a restaurant” are assigned with the same intent (Appendix-Table 1). Here, we used different pseudo labeling algorithms (semi-supervised) to assign more accurate intent to each sentence of the MultiWOZ 2.2 dataset. This can significantly help the natural language understanding processes which would improve the NLU portion of chatbot.<br>
The first step for assigning accurate intents to utterances was to extract them from the MultiWOZ 2.2 dataset, and to label a portion of them manually. 847 utterances were labeled with 21 different intents (see **Table 1** for an example). Form these 847 labeled utterances, 12 samples per each class were randomly selected as the training set to develop the intent classification model. The rest of labeled utterances were used for testing the model. The goal here was to develop a model that can accurately assign pseudo labels to a set of unlabeled data. For this task, a set of unlabeled data with 3783 utterances was extracted from the MultiWOZ 2.2 dataset. We used RASA DIET Classifier (a Dual Intent and Entity Transformer) as our intent classification algorithm [2]. The DIET Classifier (consisted of one layer transformer with the size of 128, and 4 attention head) was trained with the initial training data (12 samples per classes). The test accuracy was 69.42% (the run time was ~ 1 min). Then, three main approaches were developed to improve the performance of this model leveraging the unlabeled data.<br>

**Table 1.** Comparing the old and new labels for a conversation around Restaurant
| Utterances                                                   | Old Label	        | New Label                   |
| ------------------------------------------------------------ |:------------------:| ---------------------------:|
| I want to find a cheap restaurant in the south part of town. | Find_Restaurant	| Find + Restaurant + Details |
| ------------------------------------------------------------ |:------------------:| ---------------------------:|
| Are there any other places?	                               | Find_Restaurant	| Request_Info                |
| ------------------------------------------------------------ |:------------------:| ---------------------------:|
| sorry what is the food type of that restaurant ?             | Find_Restaurant	| Restaurant + Request_info   |
| ------------------------------------------------------------ |:------------------:| ---------------------------:|
| what is the price range ?	                                   | Find_Restaurant	| Request_info                |
| ------------------------------------------------------------ |:------------------:| ---------------------------:|
| What is their address and phone number?                      | Find_Restaurant	| Request_info                |
| ------------------------------------------------------------ |:------------------:| ---------------------------:|
| Please go ahead and book a table for me                      | Book_Restaurant	| Book + Restaurant           |
| ------------------------------------------------------------ |:------------------:| ---------------------------:|
| Book it for 4 persons. Give me the reference number          | Book_Restaurant	| Book + Request_info         |
| ------------------------------------------------------------ |:------------------:| ---------------------------:|
| outside please!                                              | Book_Restaurant	| Details                     |
| ------------------------------------------------------------ |:------------------:| ---------------------------:|
| No, thanks.                                                  | Book_Restaurant	| Deny + Thanks               |





The first approach was to generate pseudo labels [3] for unlabeled data (using the model that was trained on initial training data), add them directly to the initial training data, and then finetune the model using this new training set. To improve the performance of the model, we used a loop, and added a certain number of pseudo-labeled samples to the initial training data in each iteration. Then, the model, was finetuned in each iteration, using the new training set. The idea was that the model is getting better and better in each iteration, therefore, it can generate better pseudo labels. Two methods were employed for selecting pseudo-labeled samples. In Method 1, we sorted pseudo labeled samples based on their confidence in each individual class and selected a few of them (hyperparameter) from each class to be added to the initial training data. These selected pseudo-labeled samples must have a confidence more than a threshold (hyperparameter) to be considered for adding to the initial training data. Using this method, the test accuracy was increased from 69.42% to 75.19%. Method 2 was similar to Method 1 with one major difference. In Method 2, we calculated the average confidence of pseudo-labeled samples per each class and used those as the threshold for the classes. The reason was that for some classes, the confidence of samples was always low, which resulted in addition of no new sample to the training set for those classes. Method 2 performance was lower than Method 1, where it increased the test accuracy to 74.49%.<br>

The second approach was to use a cosine similarity check before adding the pseudo-labeled samples to the initial training data. To implement this approach, a contextual sentence embedding model, called “paraphrase-MiniLM-L6-v2” (which is based on Sentence-BERT) [4] was used to map each sentence to a 384-dimensional dense vector space. An average sentence embedding vector for each individual class was generated based on the initial training data. Here, we used two methods for checking the cosine similarity. In Method 1, before adding a pseudo-labeled sample to its assigned class in the initial training set, the cosine similarity between the embedding of pseudo-labeled sample and average embedding of that class was calculated, and only if this cosine similarity was able to pass a threshold (hyperparameter), it was added to the initial training set. Using this method, the test accuracy was increased from 69.42% to 76.16%. In Method 2, we calculated the cosine similarity between the embedding of pseudo-labeled sample and average embedding of all classes in the initial training data. A pseudo-labeled sample was added to the initial training data only if the cosine similarity between this pseudo-labeled sample and its assigned class was the largest among all calculated cosine similarities. Method 2 was able to slightly outperform Method 1 and increased the test accuracy to 76.61%.<br>

The third approach was derived from FixMatch algorithm [5]. The only difference was that rather than using a weekly and a strongly augmented samples, we used the original and an augmented sample. Three augmentation methods including character replacement (using NLPAug library), synonym replacement (using ppdb [6] from NLPAug library), and paraphrase generation using T5 was tested to find the one that improve the model performance better. Here, before adding selected pseudo-labeled samples to the initial training data, we augmented the utterances while using the pseudo labels that were generated for non-augmented utterances. The character augmentation, synonym replacement, and paraphrase generation were able to increase the test accuracy from 69.42% to 74.75%, 76.81%, and 73.33%, respectively.<br>

At the end, we tried a combination of different approaches, to see if combining these approaches can help with improving the performance of the model. Since augmentation using synonym replacement had a larger boost in the performance of the model compared to the character augmentation and paraphrase generation, we selected to continue with synonym replacement for Approach 3. **Table 2** shows the result. The combination of 3 approaches significantly improved the performance of the model (the run time for all combination was approximately the same ~20 min). This implies that text augmentation (if an appropriate augmentation method is selected) can significantly help with improving intent classification performance when combined with efficient pseudo labeling method.<br>



**Table 2.** The accuracy of intent classification model: Supervised vs Semi-Supervised (leveraging unlabeled data through pseudo labeling and data augmentation)
| Models                                                                    | Accuracy % |
| ------------------------------------------------------------------------- | ----------:|
| Initial Model Trained on Labeled Data (supervised)	                    |  69.54     |
| Method 1 from Approach 1 + Method 1 from Approach 2 + synonym replacement | 80.23      |
| Method 1 from Approach 1 + Method 2 from Approach 2 + synonym replacement | 79.56      |
| Method 2 from Approach 1 + Method 1 from Approach 2 + synonym replacement | 79.05      |
| Method 2 from Approach 1 + Method 2 from Approach 2 + synonym replacement | 78.54      |




**References**:<br>
**[1]** Multiwoz 2.2: A dialogue dataset with additional annotation corrections and state tracking baselines.<br>
**[2]** “DIET: Lightweight Language Understanding for Dialogue Systems,”<br>
**[3]** Pseudo-label: The simple and efficient semi-supervised learning method for deep neural networks.<br>
**[4]** Sentence-bert: Sentence embeddings using siamese bert-networks.<br>
**[5]** Fixmatch: Simplifying semi-supervised learning with consistency and confidence. Advances in neural information processing systems.<br>
**[6]** PPDB 2.0: Better paraphrase ranking, fine-grained entailment relations, word embeddings, and style classification.<br>

