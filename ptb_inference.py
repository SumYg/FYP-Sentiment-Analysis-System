import os
import json
import torch
import argparse

from torch.utils.data import DataLoader

from setup import load
load()
from ptb import PTB
# from input_dataset import InputDataset
from model import VAEDecoder, VAEEncoder
from utils import to_var, idx2word2 as idx2word, interpolate


def main(args):
    with open(args.data_dir+'/ptb.vocab.json', 'r') as file:
        vocab = json.load(file)

    w2i, i2w = vocab['w2i'], vocab['i2w']
    # print(i2w)
    
    dataset = PTB(
            data_dir=args.data_dir,
            # raw_data_filename='sentence_split_9999_skip_first_100.pickle',
            # raw_data_filename='999_skip_first_100.pickle',
            # raw_data_filename='twitter_100.pickle',
            # raw_data_filename='sentence_split_9999_skip_first_100',
            split='valid',
            create_data=False,
            max_sequence_length=args.max_sequence_length
        )
    
    # tokenizer = dataset.tokenizer

    encoder = VAEEncoder(
        vocab_size=dataset.vocab_size,
        sos_idx=dataset.sos_idx,
        eos_idx=dataset.eos_idx,
        pad_idx=dataset.pad_idx,
        unk_idx=dataset.unk_idx,
        max_sequence_length=args.max_sequence_length,
        embedding_size=args.embedding_size,
        rnn_type=args.rnn_type,
        hidden_size=args.hidden_size,
        word_dropout=args.word_dropout,
        embedding_dropout=args.embedding_dropout,
        latent_size=args.latent_size,
        num_layers=args.num_layers,
        bidirectional=args.bidirectional
        )

    decoder = VAEDecoder(
        vocab_size=dataset.vocab_size,
        sos_idx=dataset.sos_idx,
        eos_idx=dataset.eos_idx,
        pad_idx=dataset.pad_idx,
        unk_idx=dataset.unk_idx,
        max_sequence_length=args.max_sequence_length,
        embedding_size=args.embedding_size,
        rnn_type=args.rnn_type,
        hidden_size=args.hidden_size,
        word_dropout=args.word_dropout,
        embedding_dropout=args.embedding_dropout,
        latent_size=args.latent_size,
        num_layers=args.num_layers,
        bidirectional=args.bidirectional,
        embedding=encoder.embedding
        )


    if not os.path.exists(args.load_checkpoint):
        raise FileNotFoundError(args.load_checkpoint)

    # model.load_state_dict(torch.load(args.load_checkpoint))
    model_dict = torch.load(args.load_checkpoint)
    encoder.load_state_dict(model_dict['encoder_state_dict'])
    decoder.load_state_dict(model_dict['decoder_state_dict'])
    print("Model loaded from %s" % args.load_checkpoint)

    if torch.cuda.is_available():
        encoder = encoder.cuda()
        decoder = decoder.cuda()
    
    encoder.eval()
    decoder.eval()
    # dataset = PTB(
    #         data_dir=args.data_dir,
    #         split='self',
    #         create_data=True,
    #         max_sequence_length=args.max_sequence_length,
    #     )
    print(torch.cuda.is_available())
    data_loader = DataLoader(
                dataset=dataset,
                batch_size=100,
                shuffle=False,
                pin_memory=True
            )
    
    batch = list(data_loader)[0]
    
    for k, v in batch.items():
        if torch.is_tensor(v):
            batch[k] = to_var(v)
                        
    print(batch['input'].device)
    print(batch['input'].shape)
    # print(next(model.parameters()).is_cuda, batch['input'].is_cuda)
    # batch_size, sorted_idx, mean, logv, latent, reversed_idx, _, _ = encoder(batch['input'], batch['length'])
    batch_size, sorted_idx, mean, logv, latent, reversed_idx, _ = encoder(batch['input'], batch['length'])
    latent = latent[reversed_idx]
    print(latent)
    
    # class i2wClass():
    #     def __getitem__(self, id):
    #         return tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens([id]))
    
    # i2w = i2wClass()
    draw_dist(latent.cpu().detach().numpy())
    
    samples, z, padded_outputs = decoder.inference(z=latent)
    # print(padded_outputs)
    # exit()
    print("----------------------")
    print(*idx2word(batch['target'], i2w=i2w, pad_idx=dataset.pad_idx), sep='\n')
    print("=================")
    print(*idx2word(samples, i2w=i2w, pad_idx=dataset.pad_idx), sep='\n')
    print("----------------------")
    
    # exit()
    

    samples, z, _ = decoder.inference(n=args.num_samples)
    print('----------SAMPLES----------')
    print(*idx2word(samples, i2w=i2w, pad_idx=dataset.pad_idx), sep='\n')

    z1 = torch.randn([args.latent_size]).numpy()
    z2 = torch.randn([args.latent_size]).numpy()
    z = to_var(torch.from_numpy(interpolate(start=z1, end=z2, steps=8)).float())
    samples, z, _ = decoder.inference(z=z)
    print('-------INTERPOLATION-------')
    print(*idx2word(samples, i2w=i2w, pad_idx=dataset.pad_idx), sep='\n')

    # seems no problem
    # output = decoder.generate_seq(batch_size, batch['length'], latent)
    # print(len(output))
    # print(output)
    # output = torch.cat(output)
    # print(*idx2word(output, i2w=i2w, pad_idx=dataset.pad_idx), sep='\n')

    

def draw_dist(M):
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(M)

    # We need a 2 x 944 array, not 944 by 2 (all X coordinates in one list)
    t = reduced.transpose()

    plt.scatter(t[0], t[1])
    plt.savefig('dist.png')


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
