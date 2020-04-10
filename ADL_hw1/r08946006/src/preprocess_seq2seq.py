import os
import sys
sys.path.append(os.getcwd()+"/src/")
import argparse
import logging
import os
import json
import pickle
from pathlib import Path
from utils import Tokenizer, Embedding
from dataset import Seq2SeqDataset
from tqdm import tqdm

def main(args):
    # Read test file
    with open(args.input_dataname) as f:
        test = [json.loads(line) for line in f]
    # Read embedding
    with open(str(args.output_dir) + '/embedding_seq2seq.pkl', 'rb') as f:
        embedding = pickle.load(f)
        
    tokenizer = Tokenizer(lower=True)
    tokenizer.set_vocab(embedding.vocab)

    logging.info('Creating test dataset...')
    
    create_seq2seq_dataset(
        process_samples(tokenizer, test),
        args.output_dir / 'test_seq2seq.pkl',
        tokenizer.pad_token_id
    )


def process_samples(tokenizer, samples):
    bos_id = tokenizer.bos_token_id
    eos_id = tokenizer.eos_token_id
    processeds = []
    for sample in tqdm(samples):
        processed = {
            'id': sample['id'],
            'text': tokenizer.encode(sample['text']) + [eos_id],
        }
        if 'summary' in sample:
            processed['summary'] = (
                [bos_id]
                + tokenizer.encode(sample['summary'])
                + [eos_id]
            )
        processeds.append(processed)

    return processeds


def create_seq2seq_dataset(samples, save_path, padding=0):
    dataset = Seq2SeqDataset(
        samples, padding=padding,
        max_text_len=300,
        max_summary_len=80
    )
    with open(save_path, 'wb') as f:
        pickle.dump(dataset, f)


def _parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('output_dir', type=Path)
    parser.add_argument('input_dataname', type=Path)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = _parse_args()
    loglevel = os.environ.get('LOGLEVEL', 'INFO').upper()
    logging.basicConfig(format='%(asctime)s | %(levelname)s | %(message)s',
                        level=loglevel, datefmt='%Y-%m-%d %H:%M:%S')
    main(args)
