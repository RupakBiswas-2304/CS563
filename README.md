# How run 
- Install dependencies
```bash
pip install -r requirements.txt
```

#### Run the hmm model :
```
hmm.py --help
```
```
usage: hmm.py [-h] [--n N] [--filename FILENAME] [--mode MODE] [--show_matrix SHOW_MATRIX]

HMM

options:
  -h, --help            show this help message and exit
  --n N                 Number of workers
  --filename FILENAME   The filename of the data to be loaded.
  --mode MODE           The type of HMM model (Bigram or Trigram).
  --show_matrix SHOW_MATRIX
                        Show confusion matrix
```

Example: `python3 hmm.py --mode Bigram --show_matrix true`

#### Run the rnn model :
```
python3 rnn.py 
```