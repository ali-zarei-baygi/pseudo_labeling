# Using pseudo_labeling to improve performance of text classification


In this project, we used MultiWOZ 2.2 dataset [1], which is a large-scale human-human conversational corpus spanning over eight domains: Hotel, Restaurant, Taxi, Train, Attraction, Hospital, Bus, and Police. Although this dataset is annotated, the intents (active intent in MultiWOZ 2.2 dataset) are assigned widely on the conversations. Here, we used different pseudo labeling algorithms (semi-supervised) to assign more fine-grained intent to each sentence of the MultiWOZ 2.2 dataset (Table 1).<br>

**Table 1.**
| Utterances                                                   | Old Label	        | New Label                   |
| ------------------------------------------------------------ |:------------------:| ---------------------------:|
| I want to find a cheap restaurant in the south part of town. | Find_Restaurant	| Find + Restaurant + Details |
| Are there any other places?	                               | Find_Restaurant	| Request_Info                |
| sorry what is the food type of that restaurant ?             | Find_Restaurant	| Restaurant + Request_info   |
| what is the price range ?	                                   | Find_Restaurant	| Request_info                |
| What is their address and phone number?                      | Find_Restaurant	| Request_info                |
| Please go ahead and book a table for me                      | Book_Restaurant	| Book + Restaurant           |
| Book it for 4 persons. Give me the reference number          | Book_Restaurant	| Book + Request_info         |
| outside please!                                              | Book_Restaurant	| Details                     |
| No, thanks.                                                  | Book_Restaurant	| Deny + Thanks               |



We used RASA DIET Classifier (a Dual Intent and Entity Transformer) as our intent classification algorithm [2]. The DIET Classifier was trained with the initial training data (12 samples per classes). Then, three main approaches were developed to improve the performance of this model leveraging the unlabeled data.<br>

* **Approach 1: Simple Pseudo Labeling**<br>

* **Approach 2: Similarity Check Between sentence embedding of Pseudo Labeles and Initial Training Data**<br>

* **Approach 3: Data Augmentation (FixMatch)**<br>


At the end, we tried a combination of different approaches. **Table 2** shows the result. 



**Table 2.**
| Models                                                                    | Accuracy % |
| ------------------------------------------------------------------------- | ----------:|
| Initial Model Trained on Labeled Data (supervised)	                    |  69.54     |
| Method 1 from Approach 1 + Method 1 from Approach 2 + synonym replacement | 80.23      |
| Method 1 from Approach 1 + Method 2 from Approach 2 + synonym replacement | 79.56      |
| Method 2 from Approach 1 + Method 1 from Approach 2 + synonym replacement | 79.05      |
| Method 2 from Approach 1 + Method 2 from Approach 2 + synonym replacement | 78.54      |
* please see the main.py file for the detail information about each approach and method

-------------------------------------------------------------------------------------------------------------

**References**:<br>
**[1]** Multiwoz 2.2: A dialogue dataset with additional annotation corrections and state tracking baselines.<br>
**[2]** “DIET: Lightweight Language Understanding for Dialogue Systems,”<br>
**[3]** Pseudo-label: The simple and efficient semi-supervised learning method for deep neural networks.<br>
**[4]** Sentence-bert: Sentence embeddings using siamese bert-networks.<br>
**[5]** Fixmatch: Simplifying semi-supervised learning with consistency and confidence. Advances in neural information processing systems.<br>
**[6]** PPDB 2.0: Better paraphrase ranking, fine-grained entailment relations, word embeddings, and style classification.<br>

