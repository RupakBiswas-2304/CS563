import nltk
from dataloader import DataLoader

c_treebank, c_brown, c_conll2000, tagged_sentence = None, None, None, None
# download necessary nltk data

def load_data():
    global c_treebank, c_brown, c_conll2000, tagged_sentence
    c_treebank = nltk.corpus.treebank.tagged_sents(tagset='universal')
    c_brown = nltk.corpus.brown.tagged_sents(tagset='universal')
    c_conll2000 = nltk.corpus.conll2000.tagged_sents(tagset='universal')
    tagged_sentence = c_treebank + c_brown + c_conll2000


try:
    load_data()
except:
    nltk.download('treebank')
    nltk.download('universal_tagset')
    nltk.download('brown')
    nltk.download('conll2000')
    load_data()

class RNNPostagger:
    def __init__(self) -> None:
        pass

    def train(self, data):
        pass

    def _predict(self, data):
        pass

    def test(self, data):
        pass

if __name__ == '__main__':
    d = DataLoader()
    d.load('Brown_train.txt')
    for (train, test) in d.n_fold(5):
        print(len(train), len(test))