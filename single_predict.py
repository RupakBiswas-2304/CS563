from dataloader import Dataloader, Data
from hmm import HMM, ModelType

if __name__ == '__main__':

    d = Dataloader()
    d.load('Brown_train.txt')
    hmm = HMM(ModelType.Bigram)
    hmm.train(d.train)
    while True:
        s = input()
        if s == 'exit':
            break
        data = Data(s.encode('utf-8').decode('utf-8'))
        print(hmm.predict(data.tokens[2:]))
        print(data.tags[2:])

