import os
import json
import torch
from torch import nn
import argparse
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from multiprocessing import cpu_count
import time

from setup import load
load()
from utils import partial_experiment_name
from model2 import VAEDecoder, VAEEncoder, get_bert_embedding
from model_classification import SimilarClassifier, SentimentClassifier, EntailmentClassifier

from transformers import AutoModel, AutoTokenizer

from public_dataset import STSDataset, SST2Dataset, SNLIDataset
from collections import OrderedDict, defaultdict

def to_var(x, device='cuda:1', requires_grad=False):
    if torch.cuda.is_available():
        x = x.to(device)
    return x.requires_grad_(requires_grad)

def main(args):
    # pretrained_model_name = "princeton-nlp/sup-simcse-bert-base-uncased"
    pretrained_model_name = 'bert-base-uncased'
    model_path = 'bin/2023-Apr-13-07:33:29/E29.pytorch'
    
    def process_two_sentences(batch):
        
        sentence1 = batch['sentence1']
        
        for k, v in sentence1.items():
            # print(k, type(v))
            # print(len(v), type(v[0]))
            # print(len(v[0]))
            # print(len(v[0][0]), type(v[0][0]))
            # print(v[0][0].shape)
            sentence1[k] = to_var(torch.stack(v)).T
            # print(sentence1[k].shape)
            # if torch.is_tensor(v):
            #     sentence1[k] = to_var(v)

        sentence1_representation = model(**sentence1, output_hidden_states=False, return_dict=True).pooler_output

        sentence2 = batch['sentence2']
        
        for k, v in sentence2.items():
            sentence2[k] = to_var(torch.stack(v)).T
            # if torch.is_tensor(v):
            #     sentence2[k] = to_var(v)

        sentence2_representation = model(**sentence2, output_hidden_states=False, return_dict=True).pooler_output


        # sentence2_representation = model(**tokenizer(batch['sentence2'], padding=True, truncation=True, return_tensors='pt'), output_hidden_states=False, return_dict=True).pooler_output

        # get the similarity score from classifier
        return classifier(sentence1_representation.detach().to('cuda:0'), sentence2_representation.detach().to('cuda:0'))

    def process(batch):
        sentence = batch['sentence']

        for k, v in sentence.items():
            sentence[k] = to_var(torch.stack(v)).T
            # print(k, type(v))
            # if torch.is_tensor(v):
            #     print(v.shape)
            # else:
            #     sentence[k] = to_var(torch.stack(v).T)
            #     print(sentence[k].shape)
            #     sentence1[k] = to_var(torch.stack(*v)).T
                # sentence[k] = to_var(v)
        

        sentence_representation = model(**sentence, output_hidden_states=False, return_dict=True).pooler_output
        return classifier(sentence_representation.detach().to('cuda:0'))
    
    ts = time.strftime('%Y-%b-%d-%H:%M:%S', time.localtime())
    st_time = time.time()

    splits = ['train', 'validation']
    datasets = OrderedDict()

    if args.task == "similarity":
        dataset = STSDataset
        classifier_model = SimilarClassifier
        foward_func = process_two_sentences
        criterion = lambda x, l: nn.MSELoss()(x, l.float())
        loss_name = 'MSE'
    elif args.task == 'sentiment':
        dataset = SST2Dataset
        classifier_model = SentimentClassifier
        foward_func = process
        criterion = lambda x, l: nn.BCELoss()(x.squeeze(), l.float())
        loss_name = 'BCE'
    elif args.task == 'entailment':
        dataset = SNLIDataset
        classifier_model = EntailmentClassifier
        foward_func = process_two_sentences
        criterion = nn.CrossEntropyLoss()
        loss_name = 'CrossEntropy'
    else:
        raise ValueError("Task not supported")

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)

    for split in splits:
        datasets[split] = dataset(tokenizer, split)



    # model = AutoModel.from_pretrained(pretrained_model_name)

    with open(os.path.dirname(model_path) + '/model_params.json', 'r') as f:
        model_params = json.load(f)
    embedding_size, bert_embedding = get_bert_embedding(pretrained_model_name)
    model_params['embedding_size'] = embedding_size
    model_params['embedding'] = bert_embedding
    model = VAEEncoder(**model_params)

    with open(model_path, 'rb') as f:
        model_dict = torch.load(f)

    model.load_state_dict(model_dict['encoder_state_dict'])

    # no backward pass
    for param in model.parameters():
        param.requires_grad = False

    params = {
        'task': args.task,
        'pretrained_model': pretrained_model_name,
        'model_path': model_path,
        'model_type': 'LSTM-VAE',
        'learning_rate': args.learning_rate,
        'batch_size': args.batch_size,
        'hidden_size': model.config.hidden_size,
        'epochs': args.epochs,
        'loss': loss_name,
        'ts': ts
    }

    classifier = classifier_model(**params)

    print(classifier)
    
    if args.tensorboard_logging:
        writer = SummaryWriter(os.path.join(args.logdir, partial_experiment_name(args, ts)))
        writer.add_text("classifier", str(classifier))
        writer.add_text("args", str(args))
        writer.add_text("ts", ts)
    save_model_path = os.path.join(args.save_model_path, ts)
    os.makedirs(save_model_path)

    with open(os.path.join(save_model_path, 'model_params.json'), 'w') as f:
        json.dump(params, f, indent=4)

    if torch.cuda.is_available():
        model = model.to('cuda:1')
        classifier = classifier.to('cuda:0')
        tensor = lambda: torch.cuda.FloatTensor().to('cuda:0')
        
    else:
        tensor = torch.Tensor

    
    optimizer = torch.optim.Adam(classifier.parameters(), lr=args.learning_rate)
    
    model.eval()

    print(f"Cuda available: {torch.cuda.is_available()}")

    step = 0

    

    for epoch in range(args.epochs):
        for split in splits:
            data_loader = DataLoader(
                datasets[split]
                , batch_size=args.batch_size
                , shuffle=split=='train'
                , num_workers=4  # cpu_count() // 2
                , pin_memory=torch.cuda.is_available()
            )
            tracker = defaultdict(tensor)
            
            # Enable/Disable Dropout
            if split == 'train':
                classifier.train()
            else:
                classifier.eval()

            for iteration, batch in enumerate(data_loader):
                # print(batch['sentence1'])
                # get setence1 and convert to tensor
                score = foward_func(batch)

                # calculate the loss
                l = to_var(batch['label'], device=score.device)
                # print(score, l)
                loss = criterion(score, l)

                # backward + optimization
                if split == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    step += 1
                
                tracker[loss_name] = torch.cat((tracker[loss_name], loss.data.view(1, -1)), dim=0)

                if args.tensorboard_logging:
                    writer.add_scalar("%s/%s" % (split.upper(), loss_name), loss.item(), epoch*len(data_loader) + iteration)


                if iteration % args.print_every == 0 or iteration+1 == len(data_loader):
                    print("%s Batch %04d/%i, %s Loss %9.4f"
                          % (split.upper(), iteration, len(data_loader)-1, loss_name, loss.item()))
            
            print("%s Epoch %02d/%i, Mean %s %9.4f" % (split.upper(), epoch, args.epochs, loss_name, tracker[loss_name].mean()))
            print(f"Elapsed Time: {time.time() - st_time}s")

            
            if args.tensorboard_logging:
                writer.add_scalar("%s-Epoch/%s" % (split.upper(), loss_name), torch.mean(tracker[loss_name]), epoch)


            # save checkpoint
            if split == 'train' and (epoch == args.epochs - 1 or epoch% args.save_every == (args.save_every - 1)):
                checkpoint_path = os.path.join(save_model_path, "E%i.pytorch" % epoch)
                torch.save({
                    'classifier_state_dict': classifier.state_dict(),
                    'classifier_optimizer_state_dict': optimizer.state_dict(),
                }, checkpoint_path)
                print("Model saved at %s" % checkpoint_path)
                

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-task', '--task', type=str, default='similarity')

    parser.add_argument('-log', '--logdir', type=str, default='logs')
    parser.add_argument('-bin', '--save_model_path', type=str, default='bin')

    parser.add_argument('-ep', '--epochs', type=int, default=10)
    parser.add_argument('-bs', '--batch_size', type=int, default=32)
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001)

    parser.add_argument('-v', '--print_every', type=int, default=50)
    parser.add_argument('-se', '--save_every', type=int, default=20)
    parser.add_argument('-tb', '--tensorboard_logging', action='store_true')

    args = parser.parse_args()

    main(args)
