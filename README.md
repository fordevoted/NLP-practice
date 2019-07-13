# NLP-practice
 this is the NLP practice contains predict word and `NER` task. both of them are using python and package `NLTK` to program.
 
 ## OverView
 In predict word task, we need to predict the word afer given sentence; and in NER task, we need to do ner for given corpus
 the predict wrod task end of accuracy 0.83333 in given corpus; NER task end of f1 score 0.685 in given corpus which provided by Ministry of Education, Taiwan, AICUP contest.
 ## Usage
  download source code & training data, then run the code directly. 
 ## Feature
 In predict word task, I used `Bigram` to be the language model,and do the preprocessing of input data, after read the input line by line , I change all of character into lowercase and remove all of digit by RE, and the punctuations are remove also. After that, I shrink the multiple space into single space, and tokenization ,then put all of token in to a list.
 
 In NER task, I use conditional random field (CRF) method to achieve the goal, and use `sklearn_crfsuite` to implement it with algorithms `lbfgs`,c1 =0.1 ,c2=0.1,max_iterations=3000,linesearch='StrongBacktracking',min_freq=0.<br>
 In part of word feature, I add some feature, to improve the perform, include 
 * word uppercase/lowercase
 * number
 * character
 * captial or not
 * contain space or not
 * contain'-' or not
 * bias
 * the last 3 character of word
 * the last 2 character of word

I update the feature of uppercase/lowercase, contain space/'-' or not, title, number, character, word length etc.. If bias is 1, it means everyword will ralate to context before and after，which equal to window size = 3 。<br>
 <br>
 Besides, I do some trial include:<br>
 * add prefix and suffix: this change make better perform in mt trial, and it may result in the special word in Biomedical Science.
 * update perfix and suffix while update feature: This change make better perform in my trial.
 * add `Part of Speech tag(POS)` in to feature: this change make significantly beetter perform in my tial.
 
 ## License
 ##### Fordevoed
  NCU CSIE 105802015 陳昱瑋
  
 ## Contact
  210509fssh@gmail.com
