import random

class Data:
    def __init__(self, line):
        line = line.split(" ")
        self.tokens = ["<START>", "<START>"]
        self.tags = ["START", "START"]

        # remove '/r/n' or '/n' from the last word
        if line[-1][-2:] == '\r\n' or len(line[-1]) == 0  or line[-1][-1] == '\n':
            line.pop(-1)

        for word in line:
            word = word.split("/")
            self.tokens.append("/".join(word[:-1]))
            self.tags.append(word[-1])

    def __str__(self):
        f_str = ""
        for i in range(len(self.tokens)):
            f_str += f"{self.tokens[i]}[{self.tags[i]}] "
        return f_str
    
class Dataloader:
    def __init__(self) -> None:
        self.data = []
        self.train = []
        self.train_size = 70

    def load(self, filename: str, train_size: int = 70) -> None:
        self.train_size = train_size
        with open(filename, 'rb') as f:
            for line in f:
                self.data.append(Data(line.decode('utf-8')))

        # random.shuffle(self.data)
        t = len(self.data)
        t = t*self.train_size//100
        self.train = self.data[:t]

        # suffle the data
    def n_fold(self, n: int):
        # random.shuffle(self.data)
        t = len(self.data)
        test_size = t//n
        start_index = 0

        for i in range(n):
            data_test = self.data[start_index:start_index+test_size]
            data_train = self.data[:start_index]+self.data[start_index+test_size:]
            start_index += test_size
            yield (data_train, data_test)



if __name__ == '__main__':
    d = Dataloader()
    d.load('Brown_train.txt')
    for (train, test) in d.n_fold(5):
        print(len(train), len(test))
        print(train[0])
        print(test[0])
        print()