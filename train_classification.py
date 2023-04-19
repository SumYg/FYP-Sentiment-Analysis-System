import os
import json
import torch
from torch import nn
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader

from setup import load
load()
# from ptb import PTB
# from input_dataset import InputDataset
from input_dataset_simcse import InputDataset
# from model import VAEDecoder, VAEEncoder
# from model2 import VAEDecoder, VAEEncoder, get_bert_embedding
from model3 import VAEDecoder, VAEEncoder
# from utils import to_var, idx2word2 as idx2word, interpolate
from utils import to_var, idx2word, interpolate

import pandas as pd



def main(args):
    
    dataset = InputDataset(
            data_dir=args.data_dir,
            # raw_data_filename='sentence_split_9999_skip_first_100.pickle',
            # raw_data_filename='999_skip_first_100.pickle',
            # raw_data_filename='twitter_100.pickle',
            raw_data_filename='sentence_split_99999_skip_first_0',
            split='valid',
            create_data=False,
            max_sequence_length=args.max_sequence_length
        )

    tokenizer = dataset.tokenizer

    # pretrained_name = 'bert-base-uncased'
    # embedding_size, bert_embedding = get_bert_embedding(pretrained_name)
    embedding_size = args.embedding_size

    encoder = VAEEncoder(
        vocab_size=dataset.vocab_size,
        sos_idx=dataset.sos_idx,
        eos_idx=dataset.eos_idx,
        pad_idx=dataset.pad_idx,
        unk_idx=dataset.unk_idx,
        max_sequence_length=args.max_sequence_length,
        embedding_size=embedding_size,  # args.embedding_size
        rnn_type=args.rnn_type,
        hidden_size=args.hidden_size,
        word_dropout=args.word_dropout,
        embedding_dropout=args.embedding_dropout,
        latent_size=args.latent_size,
        num_layers=args.num_layers,
        bidirectional=args.bidirectional,
        # embedding=bert_embedding
        )

    decoder = VAEDecoder(
        vocab_size=dataset.vocab_size,
        sos_idx=dataset.sos_idx,
        eos_idx=dataset.eos_idx,
        pad_idx=dataset.pad_idx,
        unk_idx=dataset.unk_idx,
        max_sequence_length=args.max_sequence_length,
        embedding_size=embedding_size,  # args.embedding_size
        rnn_type=args.rnn_type,
        hidden_size=args.hidden_size,
        word_dropout=args.word_dropout,
        embedding_dropout=args.embedding_dropout,
        latent_size=args.latent_size,
        num_layers=args.num_layers,
        bidirectional=args.bidirectional,
        # embedding=bert_embedding
        embedding=encoder.encoder.embeddings.word_embeddings
        )

    # encoder = encoder
    # encoder = nn.DataParallel(encoder)
    # decoder = decoder
    # decoder = nn.DataParallel(decoder)


    if not os.path.exists(args.load_checkpoint):
        raise FileNotFoundError(args.load_checkpoint)

    # model.load_state_dict(torch.load(args.load_checkpoint))
    model_dict = torch.load(args.load_checkpoint)
    for k, v in model_dict.items():
        print(k)
    encoder.load_state_dict(model_dict['encoder_state_dict'])
    decoder.load_state_dict(model_dict['decoder_state_dict'])
    print("Model loaded from %s" % args.load_checkpoint)

    if torch.cuda.is_available():
        encoder = encoder.cuda()
        decoder = decoder.cuda()
    
    encoder.eval()
    decoder.eval()

    print(torch.cuda.is_available())

    data = []
    with open('../NLI/snli_1.0/snli_1.0_train.jsonl', 'r') as f:
        line = f.readline()
        while line:
            instance = json.loads(line)
            data.append([instance['gold_label'], instance['sentence1'], instance['sentence2']])
            line = f.readline()

    df = pd.DataFrame(data, columns=['Label', 'Sentence1', 'Sentence2'])
    training_df = df[df['Label'] != '-']

    # batch_size = 1000
    batch_size = 30

    tensor = torch.Tensor
    long_tensor = torch.LongTensor
    def make_batch_tensor(tensors):
        return tensor(torch.stack(tensors, dim=0)).cuda()

    latent1s = []
    latent2s = []
    for i in tqdm(range(0, len(training_df), batch_size)):
        training_df_batch = training_df[i:i+batch_size]
        # get the latent code
        

        batch_input, batch_target, batch_attention_mask, batch_length = list(zip(*(dataset._tokenize_sentence(s) for s in training_df_batch['Sentence1'])))

        batch_input, batch_target, batch_attention_mask, batch_length = make_batch_tensor(batch_input), make_batch_tensor(batch_target), make_batch_tensor(batch_attention_mask), long_tensor(batch_length).cuda()

        # _, _, _, latent, reversed_idx, _ = encoder(batch_input, batch_length)
        _, _, latent1 = encoder(batch_input, batch_attention_mask)
        # latent1 = latent[reversed_idx]

        # batch_input, batch_target, batch_length = list(zip(*(dataset._tokenize_sentence(s) for s in training_df_batch['Sentence2'])))
        batch_input, batch_target, batch_attention_mask, batch_length = list(zip(*(dataset._tokenize_sentence(s) for s in training_df_batch['Sentence2'])))

        # batch_input, batch_target, batch_length = make_batch_tensor(batch_input), make_batch_tensor(batch_target), long_tensor(batch_length).cuda()
        batch_input, batch_target, batch_attention_mask, batch_length = make_batch_tensor(batch_input), make_batch_tensor(batch_target), make_batch_tensor(batch_attention_mask), long_tensor(batch_length).cuda()
        # _, _, _, latent, reversed_idx, _ = encoder(batch_input, batch_length)
        _, _, latent2 = encoder(batch_input, batch_attention_mask)
        # latent2 = latent[reversed_idx]

        # detach and save
        latent1s.extend(latent1.detach().cpu().numpy())
        latent2s.extend(latent2.detach().cpu().numpy())

    training_df['latent1'] = latent1s
    training_df['latent2'] = latent2s
    
    # save dataframe to csv
    training_df.to_csv('training_df.csv', index=False)
    # 0/0

    # data_loader = DataLoader(
    #             dataset=dataset,
    #             batch_size=100,
    #             shuffle=False,
    #             pin_memory=True
    #         )
    
    # batch = list(data_loader)[0]
    
    # for k, v in batch.items():
    #     if torch.is_tensor(v):
    #         batch[k] = to_var(v)
                        
    # print(batch['input'].device)
    # print(batch['input'].shape)

    # batch_size, sorted_idx, mean, logv, latent, reversed_idx, _ = encoder(batch['input'], batch['length'])
    # latent = latent[reversed_idx]
    # print(latent)
    
    # class i2wClass():
    #     def __getitem__(self, id):
    #         return tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens([id]))
    
    # i2w = i2wClass()
    # draw_dist(latent.cpu().detach().numpy())
    
    # samples, z, padded_outputs = decoder.inference(z=latent)

    # print("----------------------")
    # print(*idx2word(batch['target'], i2w=i2w, pad_idx=dataset.pad_idx), sep='\n')
    # print("=================")
    # print(*idx2word(samples, i2w=i2w, pad_idx=dataset.pad_idx), sep='\n')
    # print("----------------------")
   
    

    # samples, z, _ = decoder.inference(n=args.num_samples)
    # print('----------SAMPLES----------')
    # print(*idx2word(samples, i2w=i2w, pad_idx=dataset.pad_idx), sep='\n')

    # z1 = torch.randn([args.latent_size]).numpy()
    # z2 = torch.randn([args.latent_size]).numpy()
    # z = to_var(torch.from_numpy(interpolate(start=z1, end=z2, steps=8)).float())
    # samples, z, _ = decoder.inference(z=z)
    # print('-------INTERPOLATION-------')
    # print(*idx2word(samples, i2w=i2w, pad_idx=dataset.pad_idx), sep='\n')

    

# def draw_dist(M):
#     import numpy as np
#     import matplotlib.pyplot as plt
#     from sklearn.decomposition import PCA
#     pca = PCA(n_components=2)
#     reduced = pca.fit_transform(M)

#     # We need a 2 x 944 array, not 944 by 2 (all X coordinates in one list)
#     t = reduced.transpose()

#     plt.scatter(t[0], t[1])
#     plt.savefig('dist.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-c', '--load_checkpoint', type=str)
    parser.add_argument('-n', '--num_samples', type=int, default=10)

    parser.add_argument('-dd', '--data_dir', type=str, default='dataset')
    parser.add_argument('-ms', '--max_sequence_length', type=int, default=50)
    parser.add_argument('-eb', '--embedding_size', type=int, default=300)
    parser.add_argument('-rnn', '--rnn_type', type=str, default='gru')
    parser.add_argument('-hs', '--hidden_size', type=int, default=256)
    parser.add_argument('-wd', '--word_dropout', type=float, default=0)
    parser.add_argument('-ed', '--embedding_dropout', type=float, default=0.5)
    parser.add_argument('-ls', '--latent_size', type=int, default=16)
    parser.add_argument('-nl', '--num_layers', type=int, default=1)
    parser.add_argument('-bi', '--bidirectional', action='store_true')

    args = parser.parse_args()

    args.rnn_type = args.rnn_type.lower()

    assert args.rnn_type in ['rnn', 'lstm', 'gru']
    assert 0 <= args.word_dropout <= 1

    main(args)
