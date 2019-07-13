import nltk
import sklearn
import sklearn_crfsuite
from sklearn_crfsuite import metrics
import os 
nltk.download('averaged_perceptron_tagger')
def read_iob2_sents(in_iob2_file):
    sents = []
    with open(in_iob2_file, 'r', encoding = 'utf8') as f:
        for sent_iob2 in f.read().split('\n\n'):
            sent = []
            for raw in sent_iob2.split('\n'):
                if raw == '':
                    continue
                columns = raw.split('\t')
                sent.append(tuple(columns))
            sents.append(sent)
    return sents
train_sents = read_iob2_sents('train.iob2')
train_sents[0]

def sent2labels(sent): return [label for token, pos,label in sent]
def word2features(sent, i):
    word = sent[i][0]
    postag = sent[i][1]
    features = {
        'bias' : 1.0,
        'word': word,
        'word_len':len(word),
        'word[0:3]':word[0:3],
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.islower()': word.islower(),
        'word.isalnum()': word.isalnum(),
        'word.isupper()': word.isupper(),
        'word.isdigit()': word.isdigit(), 
        'word.isalpha()': word.isalpha(),
        'word.istitle()': word.istitle(),
        'word.containspace()':' ' in word==True,
        'word.containdash()':'-' in word==True,
        'postag': postag,
        'postag[:2]': postag[:2],}
    if i > 0:
        word1 = sent[i-1][0]
        postag1= sent[i-1][1]
        features.update({
            '-1:word': word1,
            '-1:word_len':len(word1),
            '-1:word[0:3]':word1[0:3],
            '-1:word[-3:]': word1[-3:],
            '-1:word[-2:]': word1[-2:],
            '-1:word.islower()': word1.islower(),
            '-1:word.isalnum()': word1.isalnum(), 
            '-1:word.isupper()': word1.isupper(),
            '-1:word.isdigit()': word1.isdigit(),
            '-1:word.isalpha()': word1.isalpha(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.containspace()':' ' in word==True,
            '-1:word.containdash()':'-' in word==True,
            '-1:postag': postag1,
            '-1:postag[:2]': postag1[:2]})
    else:
        features['BOS'] = True
    if i < len(sent)-1:
        word1 = sent[i+1][0]
        postag1 = sent[i+1][1]
        features.update({
            '+1:word': word1,
            '+1:word_len':len(word1),
            '+1:word[0:3]':word1[0:3],
            '+1:word[-3:]': word1[-3:],
            '+1:word[-2:]': word1[-2:],
            '+1:word.islower()': word1.islower(),
            '+1:word.isalnum()': word1.isalnum(), 
            '+1:word.isupper()': word1.isupper(),
            '+1:word.isdigit()': word1.isdigit(),
            '+1:word.isalpha()': word1.isalpha(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.containspace()':' ' in word==True,
            '+1:word.containdash()':'-' in word==True,
            '+1:postag': postag1,
            '+1:postag[:2]': postag1[:2]})
    else:
        features['EOS'] = True
    return features

def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]
def appendpostagged(sent):
    data=[]
    for i, s in enumerate(sent):
        # Obtain the list of tokens in the document
        tokens = [t for t, label in s]
        # Perform POS tagging
        for token in tokens :
            if(token==''):
                tokens.remove('')
        tagged = nltk.pos_tag(tokens)
        print(tagged)
        # Take the word, POS tag, and its label
        data.append([(w, pos, label) for (w, label), (word, pos) in zip(s, tagged)])
    return data

def write_iob2(data, pred, out_iob2_file):
    with open(out_iob2_file, 'wb') as iob2_writer:
        for _data, _pred in zip(data, pred):
            for _tuple, _pred in zip(_data, _pred):
                iob2_writer.write(bytes('\t'.join(_tuple) + '\t' + _pred + '\n', encoding = 'utf8'))
            iob2_writer.write(bytes('\n', encoding = 'utf8'))


#main
print("go")
train_sents = read_iob2_sents('train.iob2')
train_sents = appendpostagged(train_sents)
#print(train_sents)
test_sents = read_iob2_sents('test.iob2')
test_sents = appendpostagged(test_sents)

X_train = [sent2features(s) for s in train_sents]
y_train = [sent2labels(s) for s in train_sents]

X_test = [sent2features(s) for s in test_sents]
y_test = [sent2labels(s) for s in test_sents]
print("openfile success")

crf = sklearn_crfsuite.CRF(
    algorithm='lbfgs',
    all_possible_states=True,
    all_possible_transitions=True,
    c1 =0.1,
    c2=0.1,
    linesearch='StrongBacktracking',
    max_iterations=8000,
    min_freq=0
)
print("start train")
crf.fit(X_train, y_train)
print("end and start pred")
y_pred = crf.predict(X_test)
print(y_pred)

labels = list(crf.classes_)
labels.remove('O')
sorted_labels = sorted(
    labels,
    key=lambda name: (name[1:], name[0])
)
print(metrics.flat_classification_report(y_test, y_pred, labels=sorted_labels, digits=3))
write_iob2(test_sents, y_pred, 'answer.iob2')
 




