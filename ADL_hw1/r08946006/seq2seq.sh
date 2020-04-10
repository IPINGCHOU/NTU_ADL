python3.6 ./src/preprocess_seq2seq.py . $1 # transfering test.jsonl to test.pkl(tokenized), 1 for input data
python3.6 seq2seq.py ./test_seq2seq.pkl $2 # start predicting, 2 for output path
