import enum
from dataloader import Dataloader
import multiprocessing
import argparse

class ModelType(enum.Enum):
    Bigram = 1
    Trigram = 2


class HMM:
    def __init__(self, type: ModelType) -> None:
        """
        Initialize the HMM class.

        Args:
            type (ModelType): The type of HMM model (Bigram or Trigram).
            filename (str): The filename of the data to be loaded.

        Returns:
            None
        """
        self.type = type
        self.EP = {}  # emission probability
        self.TP = {}  # transition probability
        self.SP = {}  # start probability
        self.GEP = {}  # Global emission probability

    def train(self, train_data, normalize=True) -> None:
        """
        Train the HMM model.

        Args:
            normalize (bool, optional): Whether to normalize the probabilities. Defaults to True.

        Returns:
            None
        """
        for data in train_data:

            # start probability
            if data.tags[2] not in self.SP:
                self.SP[data.tags[2]] = 0
            self.SP[data.tags[2]] += 1

            for i in range(2, len(data.tags)):

                # transition probability
                if data.tags[i] not in self.TP:
                    self.TP[data.tags[i]] = {}
                if data.tags[i - 1] not in self.TP[data.tags[i]]:

                    if self.type == ModelType.Bigram:
                        self.TP[data.tags[i]][data.tags[i - 1]] = 0
                    else:
                        self.TP[data.tags[i]][data.tags[i - 1]] = {}

                if self.type == ModelType.Bigram:
                    self.TP[data.tags[i]][data.tags[i - 1]] += 1
                else:
                    if data.tags[i - 2] not in self.TP[data.tags[i]][data.tags[i - 1]]:
                        self.TP[data.tags[i]][data.tags[i - 1]][data.tags[i - 2]] = 0
                    self.TP[data.tags[i]][data.tags[i - 1]][data.tags[i - 2]] += 1

                # emission probability
                if data.tokens[i] not in self.EP:
                    self.EP[data.tokens[i]] = {}

                if data.tags[i] not in self.EP[data.tokens[i]]:
                    self.EP[data.tokens[i]][data.tags[i]] = 0

                if data.tags[i] not in self.GEP:
                    self.GEP[data.tags[i]] = 0

                self.GEP[data.tags[i]] += 1
                self.EP[data.tokens[i]][data.tags[i]] += 1

        self.states = list(self.SP.keys())

        if not normalize:
            print("Training done. [Not normalized]")
            return
        # normalize
        for tag in self.SP:
            self.SP[tag] /= len(train_data)

        s = sum(self.GEP.values())
        for tag in self.GEP:
            self.GEP[tag] /= s

        for tp in self.TP.values():
            if self.type == ModelType.Bigram:
                s = sum(tp.values())
                for tag in tp:
                    tp[tag] /= s
            else:
                s = 0
                for tpp in tp.values():
                    s += sum(tpp.values())

                for _, value in tp.items():
                    for k, _ in value.items():
                        value[k] /= s

        for ep in self.EP.values():
            s = sum(ep.values())
            for tag in ep:
                ep[tag] /= s

        print("Training done. [Normalized]")

    def predict(self, tokens: list) -> list:
        """
        Predict the tags for the given tokens.

        Args:
            tokens (list): The list of tokens to predict the tags for.

        Returns:
            list: The predicted tags for the given tokens.
        """
        if self.type == ModelType.Bigram:
            return self._predict_bigram(tokens)
        else:
            return self._predict_trigram(tokens)

    def _predict_trigram(self, tokens: list) -> list:
        """
        Viterbi algorithm
        """
        # tokens = tokens[2:]
        v = [{}]
        for st in self.states:
            v[0][st] = {
                "prob": self.SP[st] * self.EP.get(tokens[0], self.GEP).get(st, 0),
                "prev": "START",
            }

        for t in range(1, len(tokens)):
            v.append({})
            for st in self.states:
                max_tr_prob = 0
                prev_st_selected = self.states[0]
                for prev_st in self.states:
                    for pp_st in self.states:
                        tr_prob = (
                            v[t - 1][prev_st]["prob"]
                            * self.TP[st].get(prev_st, {}).get(pp_st, 0)
                            * self.EP.get(tokens[t], self.GEP).get(st, 0)
                        )
                        if tr_prob > max_tr_prob:
                            max_tr_prob = tr_prob
                            prev_st_selected = prev_st
                max_prob = max_tr_prob
                v[t][st] = {"prob": max_prob, "prev": prev_st_selected}

        opt = []
        max_prob = 0.0
        best_st = "X"
        for st, data in v[-1].items():
            if data["prob"] > max_prob:
                max_prob = data["prob"]
                best_st = st
        opt.append(best_st)
        previous = best_st
        for t in range(len(v) - 2, -1, -1):
            opt.insert(0, v[t + 1][previous]["prev"])
            previous = v[t + 1][previous]["prev"]
        return opt

    def _predict_bigram(self, tokens: list) -> list:
        """
        Viterbi algorithm
        """
        # tokens = tokens[2:]
        v = [{}]
        for st in self.states:
            v[0][st] = {
                "prob": self.SP[st] * self.EP.get(tokens[0], self.GEP).get(st, 0),
                "prev": "START",
            }

        for t in range(1, len(tokens)):
            v.append({})
            for st in self.states:
                max_tr_prob = (
                    v[t - 1][self.states[0]]["prob"]
                    * self.TP[st][self.states[0]]
                    * self.EP.get(tokens[t], self.GEP).get(st, 0)
                )
                prev_st_selected = self.states[0]
                for prev_st in self.states[1:]:
                    tr_prob = (
                        v[t - 1][prev_st]["prob"]
                        * self.TP[st].get(prev_st, 0)
                        * self.EP.get(tokens[t], self.GEP).get(st, 0)
                    )
                    if tr_prob > max_tr_prob:
                        max_tr_prob = tr_prob
                        prev_st_selected = prev_st
                max_prob = max_tr_prob
                v[t][st] = {"prob": max_prob, "prev": prev_st_selected}

        opt = []
        max_prob = 0.0
        best_st = None
        for st, data in v[-1].items():
            if data["prob"] > max_prob:
                max_prob = data["prob"]
                best_st = st
        opt.append(best_st)
        previous = best_st
        for t in range(len(v) - 2, -1, -1):
            opt.insert(0, v[t + 1][previous]["prev"])
            previous = v[t + 1][previous]["prev"]
        return opt


    def test(self, test_data):
        report = {}
        for st in self.states:
            report[st] = {"TP": 0, "FP": 0, "FN": 0, "TN": 0}

        total_pred, total_tp = 0, 0
        
        for data in test_data:
            pred = self.predict(data.tokens[2:])
            total_pred += len(pred)
            for i in range(len(data.tags[2:])):
                if data.tags[i+2] == pred[i]:
                    report[data.tags[i+2]]["TP"] += 1
                    total_tp += 1
                else:
                    report[data.tags[i+2]]["FN"] += 1
                    report[pred[i]]["FP"] += 1

                for st in self.states:
                    if st != data.tags[i+2] and st != pred[i]:
                        report[st]["TN"] += 1

        for st in report:
            report[st]["accuracy"] = (report[st]["TP"] + report[st]["TN"]) / (report[st]["TP"] + report[st]["TN"] + report[st]["FP"] + report[st]["FN"])
            report[st]["precision"] = report[st]["TP"] / (report[st]["TP"] + report[st]["FP"])
            report[st]["recall"] = report[st]["TP"] / (report[st]["TP"] + report[st]["FN"])
            report[st]["f1"] = 2 * report[st]["precision"] * report[st]["recall"] / (report[st]["precision"] + report[st]["recall"])
        return report, total_tp/total_pred


def worker_b(data):
    worker(ModelType.Bigram, data)

def worker_t(data):
    worker(ModelType.Trigram, data)

def worker(model_type, data):
    worker_id = multiprocessing.current_process().name
    train_data, test_data = data
    model = HMM(model_type)
    model.train(train_data)

    report, gloabl_accurecy = model.test(test_data)
    # print(report)
    print("State\t\tAccuracy\tPrecision\tRecall\t\tF1 Score")
    for st in report:
        print(f"{st}\t\t{report[st]['accuracy']:.6f}\t{report[st]['precision']:.6f}\t{report[st]['recall']:.6f}\t{report[st]['f1']:.6f}")
    print(f"Worker {worker_id} done. Total Accurecy : {gloabl_accurecy}\n\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HMM")
    parser.add_argument("--n", type=int, default=5, help="Number of workers")
    parser.add_argument("--filename", type=str, default="Brown_train.txt", help="The filename of the data to be loaded.")
    parser.add_argument("--mode", type=str, default="Bigram", help="The type of HMM model (Bigram or Trigram).")

    args = parser.parse_args()

    data = Dataloader()
    data.load(args.filename, 80)
    n = args.n

    pool = multiprocessing.Pool(n)
    train_test_itreable = data.n_fold(n)

    if args.mode == "Bigram":
        pool.map(worker_b, train_test_itreable)
    else:
        pool.map(worker_t, train_test_itreable)
    
    pool.close()
    pool.join()

        
