import os
import json
import time
import torch
import argparse
import numpy as np
from multiprocessing import cpu_count
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from collections import OrderedDict, defaultdict

from transformers import BertModel

from setup import load
load()
# from ptb import PTB
from input_dataset import InputDataset
from utils import to_var, idx2word, expierment_name
# from model import VAEDecoder, VAEEncoder
from model2 import VAEDecoder, VAEEncoder, get_bert_embedding
from torch import nn
import random
from math import ceil

def create_batch(texts, teacher_forcing_prob):
    input_ids = []
    attention_masks = []
    target_ids = []

    for text in texts:
        # Tokenize the input text
        encoded = tokenizer.encode_plus(text, padding="max_length", truncation=True, max_length=128, return_tensors="pt")

        # Add the input_ids and attention_mask to the batch
        input_ids.append(encoded["input_ids"].squeeze(0))
        attention_masks.append(encoded["attention_mask"].squeeze(0))

        # Add the target_ids to the batch, with a chance of not using teacher forcing
        if random.random() < teacher_forcing_prob:
            target_ids.append(encoded["input_ids"].squeeze(0))
        else:
            # Use the model's predictions as the target
            with torch.no_grad():
                target = model.generate(input_ids, attention_masks)
            target_ids.append(target.squeeze(0))

    # Convert the batch to tensors
    input_ids = torch.stack(input_ids, dim=0).to(device)
    attention_masks = torch.stack(attention_masks, dim=0).to(device)
    target_ids = torch.stack(target_ids, dim=0).to(device)

    return input_ids, attention_masks, target_ids


def main(args):
    ts = time.strftime('%Y-%b-%d-%H:%M:%S', time.gmtime())
    st_time = time.time()

    # splits = ['train', 'valid'] + (['test'] if args.test else [])
    splits = ['train', 'valid']

    datasets = OrderedDict()
    for split in splits:
        datasets[split] = InputDataset(
            data_dir=args.data_dir,
            # raw_data_filename='sentence_split_full_7061004_skip_first_0',
            raw_data_filename='sentence_split_99999_skip_first_0',
            split=split,
            create_data=args.create_data,
            max_sequence_length=args.max_sequence_length,
            min_occ=args.min_occ
        )

    pretrained_name = 'bert-base-uncased'
    embedding_size, bert_embedding = get_bert_embedding(pretrained_name)

    params = dict(
        vocab_size=datasets['train'].vocab_size,
        sos_idx=datasets['train'].sos_idx,
        eos_idx=datasets['train'].eos_idx,
        pad_idx=datasets['train'].pad_idx,
        unk_idx=datasets['train'].unk_idx,
        max_sequence_length=args.max_sequence_length,
        embedding_size=embedding_size,  # args.embedding_size
        embedding=bert_embedding,
        rnn_type=args.rnn_type,
        hidden_size=args.hidden_size,
        word_dropout=args.word_dropout,
        embedding_dropout=args.embedding_dropout,
        latent_size=args.latent_size,
        num_layers=args.num_layers,
        bidirectional=args.bidirectional
    )
    sos_idx = datasets['train'].sos_idx  # eos_idx = datasets['train'].eos_idx

    # devices = [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    encoder = VAEEncoder(**params)
    # encoder.to(devices)
    # decoder = VAEDecoder(**params, bert=encoder.bert)
    decoder = VAEDecoder(**params)
    # decoder.to(devices)
    if torch.cuda.device_count() > 1:
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        encoder = nn.DataParallel(encoder)
        encoder.to(device)
        decoder = nn.DataParallel(decoder)
        decoder.to(device)
    elif torch.cuda.device_count() == 1:
        encoder = encoder.to(device)
        decoder = decoder.to(device)


    # decoder = VAEDecoder(**params, embedding=encoder.embedding)

    teacher_forcing_ratio = args.teacher_forcing
    
    # model.load_state_dict(torch.load("bin/2023-Jan-06-continue/E49.pytorch"))
    print("Number of trainable parameters:", sum(p.numel() for p in encoder.parameters() if p.requires_grad) + sum(p.numel() for p in decoder.parameters() if p.requires_grad))
    print("Number of all parameters:", sum(p.numel() for p in encoder.parameters()) + sum(p.numel() for p in decoder.parameters()))
    # if torch.cuda.is_available():
    #     encoder = encoder.cuda()
    #     decoder = decoder.cuda()
    #     print("Gpus:")
    #     for i in range(torch.cuda.device_count()):
    #         print(torch.cuda.get_device_name(0))
    #     print("--------------------------------------")

    print(encoder)
    print(decoder)

    if args.tensorboard_logging:
        writer = SummaryWriter(os.path.join(args.logdir, expierment_name(args, ts)))
        writer.add_text("encoder", str(encoder))
        writer.add_text("decoder", str(decoder))
        writer.add_text("args", str(args))
        writer.add_text("ts", ts)

    save_model_path = os.path.join(args.save_model_path, ts)
    os.makedirs(save_model_path)

    params['embedding'] = pretrained_name

    with open(os.path.join(save_model_path, 'model_params.json'), 'w') as f:
        json.dump(params, f, indent=4)

    def kl_anneal_function(anneal_function, step, k, x0):
        # return .5
        if anneal_function == 'logistic':
            # return float(1/(1+np.exp(-x0*(step-x0)/1000000)))
            return 1/(1+np.exp(-k*(step-x0)))
            # return float(0) if step < 600 else float(1/(1+np.exp(-k*(step-x0))))
        elif anneal_function == 'linear':
            return min(1, step/x0)

    NLL = torch.nn.NLLLoss(ignore_index=datasets['train'].pad_idx, reduction='sum')

    # criterion = nn.NLLLoss()
    
    def loss_fn(nll_loss, length, mean, logv, anneal_function, step, k, x0):

        # cut-off unnecessary padding from target, and flatten
        # target = target[:, :torch.max(length).item()].contiguous().view(-1)
        # target = target[:, :torch.max(length).item()].contiguous().view(-1)
        # # target = target.view(-1)
        # logp = logp[:, :torch.max(length).item()].contiguous().view(-1, logp.size(2))
        # # logp = logp.view(-1, logp.size(2))
        # # print(logp.shape)
        # # print(target.shape)
        # # Negative Log Likelihood
        # # print(logp.shape)
        # # print(target.shape)
        # # exit()
        
        # NLL_loss = NLL(logp, target)
        
        # print(logv.shape, mean.shape)
        # KL Divergence
        KL_loss = -0.5 * torch.sum(1 + logv - mean.pow(2) - logv.exp())
        KL_weight = kl_anneal_function(anneal_function, step, k, x0)
        
        # print(KL_loss)
        # exit()

        return nll_loss, KL_loss, KL_weight

    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=args.learning_rate) #, weight_decay=1e-3)
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=args.learning_rate) #, weight_decay=3e-3)

    tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor
    step = 0

    total_steps = args.epochs * ceil(len(datasets['train'])/ args.batch_size)

    x0 = total_steps/ 2
    # replaced args.x0 with x0
    # min_kl_loss = 20

    for epoch in range(args.epochs):

        for split in splits:
            # print(datasets[split][97])
            # print("----------------")
            data_loader = DataLoader(
                dataset=datasets[split],
                batch_size=args.batch_size,
                shuffle=split=='train',
                num_workers=8,
                pin_memory=torch.cuda.is_available()
            )

            tracker = defaultdict(tensor)

            # Enable/Disable Dropout
            if split == 'train':
                encoder.train()
                decoder.train()
            else:
                encoder.eval()
                decoder.eval()

            for iteration, batch in enumerate(data_loader):
                # print(batch['input'])
                # print(batch)
                batch_size = batch['input'].size(0)
                # print("Batch size", batch_size)
                # exit()
                for k, v in batch.items():
                    if torch.is_tensor(v):
                        batch[k] = to_var(v)

                # Forward pass
                # batch_size, sorted_idx, mean, logv, z, reversed_idx, input_embedding, sorted_lengths = encoder(batch['input'], batch['length'])  # use different embedding
                # print(batch['input'], batch['length'])
                sorted_idx, mean, logv, z, reversed_idx, sorted_lengths = encoder(batch['input'], batch['length'])
                use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
                # use_teacher_forcing = True
                # print(use_teacher_forcing)
                if use_teacher_forcing:
                    params = batch['input'], batch['length'], sorted_idx, mean, logv, z, reversed_idx, sorted_lengths
                    logp, _ = decoder(use_teacher_forcing, params)
                    # print("logp shape", logp.shape)
                    # print("batch target shape", batch['target'].shape)

                    target = batch['target'][:, :torch.max(batch['length']).item()].contiguous().view(-1)
                    # print("target shape", target.shape)
                    # target = target.view(-1)
                    logp = logp[:, :torch.max(batch['length']).item()].contiguous().view(-1, logp.size(2))
                    # print("logp shape", logp.shape)
                    nll_loss = NLL(logp, target)
                else:
                    input_sequence = to_var(torch.Tensor(batch_size).fill_(sos_idx).long())  # newly change from eos to sos
                    params = input_sequence, z, True
                    nll_loss = 0
                    target_tensor = batch['target'][sorted_idx]
                    # print(batch['length'])
                    
                    # ended_sequence_indices_set = set()
                    # running_indices = torch.arange(batch_size)

                    # hidden = None
                    # print("EOS index", eos_idx)
                    # print(datasets['train'].pad_idx)
                    # print(batch['target'])
                    # 1/0
                    for di in range(max(batch['length'])):
                        logp, hidden = decoder(use_teacher_forcing, params)
                        # decoder_output, decoder_hidden = decoder(use_teacher_forcing, params)

                        # # select top 1 word from output
                        # topv, topi = logp.topk(1)
                        # decoder_input = topi.squeeze()  # detach from history as input

                        # sample output
                        probs = logp.exp().squeeze()
                        m = torch.distributions.Categorical(probs)
                        decoder_input = m.sample()

                        # print(8, target_tensor[:, di].shape, target_tensor[:, di])
                        # print(8, target_tensor[:, di].shape)
                        # print(9, target_tensor.shape)
                        logp = logp.squeeze(1)
                        # local_loss = NLL(logp, target_tensor[:, di])
                        # print("criterion(logp, target_tensor[di])", local_loss.shape)
                        # print(local_loss)
                        # 1/0
                        # print(reversed_idx.shape, logp.shape)
                        nll_loss += NLL(logp, target_tensor[:, di])
                        
                        # for seq_index, decoder_next_input in enumerate(decoder_input):
                        #     if decoder_next_input.item() == eos_idx:
                        #         ended_sequence_indices_set.add(seq_index)

                        # print(params[0], decoder_input)
                        params = decoder_input, hidden, False
                    # print(ended_sequence_indices_set)
                    # 0/0

                # loss calculation
                NLL_loss, KL_loss, KL_weight = loss_fn(nll_loss,
                    batch['length'], mean, logv, args.anneal_function, step, args.k, args.x0)  # original x0

                loss = (NLL_loss + KL_weight * KL_loss) / batch_size
                # loss = (NLL_loss + max(min_kl_loss, KL_weight * KL_loss)) / batch_size
                # print(loss)
                # 0/0
                # backward + optimization
                if split == 'train':
                    encoder_optimizer.zero_grad()
                    decoder_optimizer.zero_grad()
                    loss.backward()
                    encoder_optimizer.step()
                    decoder_optimizer.step()
                    step += 1

                # bookkeepeing
                tracker['ELBO'] = torch.cat((tracker['ELBO'], loss.data.view(1, -1)), dim=0)

                if args.tensorboard_logging:
                    writer.add_scalar("%s/ELBO" % split.upper(), loss.item(), epoch*len(data_loader) + iteration)
                    writer.add_scalar("%s/NLL Loss" % split.upper(), NLL_loss.item() / batch_size,
                                      epoch*len(data_loader) + iteration)
                    writer.add_scalar("%s/KL Loss" % split.upper(), KL_loss.item() / batch_size,
                                      epoch*len(data_loader) + iteration)
                    writer.add_scalar("%s/KL Weight" % split.upper(), KL_weight,
                                      epoch*len(data_loader) + iteration)

                if iteration % args.print_every == 0 or iteration+1 == len(data_loader):
                    print("%s Batch %04d/%i, Loss %9.4f, NLL-Loss %9.4f, KL-Loss %9.4f, KL-Weight %6.3f"
                          % (split.upper(), iteration, len(data_loader)-1, loss.item(), NLL_loss.item()/batch_size,
                          KL_loss.item()/batch_size, KL_weight))

                # if split == 'valid':
                #     if 'target_sents' not in tracker:
                #         tracker['target_sents'] = list()
                #     tracker['target_sents'] += idx2word(batch['target'].data, i2w=datasets['train'].get_i2w(),
                #                                         pad_idx=datasets['train'].pad_idx)
                #     tracker['z'] = torch.cat((tracker['z'], z.data), dim=0)

            print("%s Epoch %02d/%i, Mean ELBO %9.4f" % (split.upper(), epoch, args.epochs, tracker['ELBO'].mean()))
            print(f"Elapsed Time: {time.time() - st_time}s")

            if args.tensorboard_logging:
                writer.add_scalar("%s-Epoch/ELBO" % split.upper(), torch.mean(tracker['ELBO']), epoch)

            # # save a dump of all sentences and the encoded latent space
            # if split == 'valid':
            #     dump = {'target_sents': tracker['target_sents'], 'z': tracker['z'].tolist()}
            #     if not os.path.exists(os.path.join('dumps', ts)):
            #         os.makedirs('dumps/'+ts)
            #     with open(os.path.join('dumps/'+ts+'/valid_E%i.json' % epoch), 'w') as dump_file:
            #         json.dump(dump,dump_file)

            # save checkpoint
            if split == 'train' and (epoch == args.epochs - 1 or epoch% args.save_every == (args.save_every - 1)):
                checkpoint_path = os.path.join(save_model_path, "E%i.pytorch" % epoch)
                torch.save({
                    'encoder_state_dict': encoder.state_dict(),
                    'decoder_state_dict': decoder.state_dict(),
                    'encoder_optimizer_state_dict': encoder_optimizer.state_dict(),
                    'decoder_optimizer_state_dict': decoder_optimizer.state_dict(),
                }, checkpoint_path)
                print("Model saved at %s" % checkpoint_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, default='dataset')
    parser.add_argument('--create_data', action='store_true')
    parser.add_argument('--max_sequence_length', type=int, default=100)
    parser.add_argument('--min_occ', type=int, default=1)
    parser.add_argument('--test', action='store_true')

    parser.add_argument('-ep', '--epochs', type=int, default=10)
    parser.add_argument('-bs', '--batch_size', type=int, default=32)
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001)

    parser.add_argument('-eb', '--embedding_size', type=int, default=300)
    parser.add_argument('-rnn', '--rnn_type', type=str, default='gru')
    parser.add_argument('-hs', '--hidden_size', type=int, default=256)
    parser.add_argument('-nl', '--num_layers', type=int, default=1)
    parser.add_argument('-bi', '--bidirectional', action='store_true')
    parser.add_argument('-ls', '--latent_size', type=int, default=16)
    parser.add_argument('-wd', '--word_dropout', type=float, default=0.62)
    parser.add_argument('-tf', '--teacher_forcing', type=float, default=0.8)
    parser.add_argument('-ed', '--embedding_dropout', type=float, default=0.5)

    parser.add_argument('-af', '--anneal_function', type=str, default='logistic')
    parser.add_argument('-k', '--k', type=float, default=0.0025)
    parser.add_argument('-x0', '--x0', type=int, default=2500)  # now should be only for log folder name

    parser.add_argument('-v', '--print_every', type=int, default=50)
    parser.add_argument('-tb', '--tensorboard_logging', action='store_true')
    parser.add_argument('-log', '--logdir', type=str, default='logs')
    parser.add_argument('-bin', '--save_model_path', type=str, default='bin')

    parser.add_argument('-se', '--save_every', type=int, default=20)


    args = parser.parse_args()

    args.rnn_type = args.rnn_type.lower()
    args.anneal_function = args.anneal_function.lower()

    assert args.rnn_type in ['rnn', 'lstm', 'gru']
    assert args.anneal_function in ['logistic', 'linear']
    assert 0 <= args.word_dropout <= 1

    main(args)
