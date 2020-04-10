python3.6 ./src/preprocess_seq_tag.py . $1 # transfering test.jsonl to test.pkl(tokenized), 1 for input data
python3.6 extractive.py ./test_tag.pkl $2 # start predicting, 2 for output path
