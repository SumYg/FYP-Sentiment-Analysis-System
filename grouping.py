
from transformers import AutoModel, AutoTokenizer
import torch
import os, json


from setup import load
load()
from model_classification import SimilarClassifier, SentimentClassifier, EntailmentClassifier
from utils import to_var

import numpy as np
import logging

class OpinionRepresenter:
    def __init__(self, text, pretrained_model_name, batch_size=128, local_model_type=None, local_model_path=None):
        if local_model_type:
            if local_model_type != 'lstm':
                raise NotImplementedError
            from model2 import VAEEncoder, get_bert_embedding
            from input_dataset import VAETokenizer
            
            with open(os.path.dirname(local_model_path) + '/model_params.json', 'r') as f:
                model_params = json.load(f)
            embedding_size, bert_embedding = get_bert_embedding(pretrained_model_name)
            model_params['embedding_size'] = embedding_size
            model_params['embedding'] = bert_embedding
            model_params['tensor_device'] = 'cuda:0'  # no cuda 1 available
            encoder = VAEEncoder(**model_params)

            with open(local_model_path, 'rb') as f:
                encoder.load_state_dict(torch.load(f, map_location=torch.device('cuda:0'))['encoder_state_dict'])
            if torch.cuda.is_available():
                encoder.cuda()
            # no backward pass
            for param in encoder.parameters():
                param.requires_grad = False

            class pooler:
                pass
            
            def model(**batch):
                # batch = VAE_dataset._construct_data(sentence_batch)
                _, _, _, z, _, _ = encoder(batch['input'], batch['length'])
                r = pooler()
                r.pooler_output = z
                return r
            tokenizer = VAETokenizer()

            
        else:
            model = AutoModel.from_pretrained(pretrained_model_name)
            tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)

            # Set the gradient computation behavior for the model
            for param in model.parameters():
                param.requires_grad = False

            if torch.cuda.is_available():
                model = model.cuda()

        self.representations = []
        for i in range(0, len(text), batch_size):
            # report progress
            print(f"Representing {i}/{len(text)}")

            batch = text[i:i+batch_size]
            # convert text to representations
            tokens = tokenizer(batch, return_tensors='pt', padding=True, truncation=True)
            # print(tokens)
            for k, v in tokens.items():
                tokens[k] = to_var(v)
            
            with torch.no_grad():
                representations = model(**tokens).pooler_output
                # print(representations)
                self.representations.extend(representations.cpu())
    
    def get_sentence_representation(self, index):
        return self.representations[index].cuda()

    def get_sentence_representations(self, indices):
        sliced = []
        for i in indices:
            if i < len(self.representations):
                sliced.append(self.representations[i])
        return torch.stack(sliced).cuda() if sliced else torch.empty(0).cuda()  # when 
    
    def __len__(self):
        return len(self.representations)

class OpinionGrouper:
    def __init__(self, classifier_path, batch_size=128, score_threshold=.7) -> None:
        self.batch_size = batch_size
        self.score_threshold = score_threshold
        # get the parent directory of the classifier
        parent_dir = os.path.dirname(classifier_path)
        # get the model params
        with open(os.path.join(parent_dir, 'model_params.json'), 'r') as f:
            # load the model params
            self.model_params = json.load(f)
        self.pretrained_model_name = self.model_params['pretrained_model']

        if self.model_params['task'] == "similarity":
            self.classifier = SimilarClassifier
            # foward_func = process_two_sentences
        elif self.model_params['task'] == 'sentiment':
            self.classifier = SentimentClassifier
            # foward_func = process
        elif self.model_params['task'] == 'entailment':
            self.classifier = EntailmentClassifier
            # foward_func = process_two_sentences
        else:
            raise ValueError("Task not supported")
        self.classifier = self.classifier(self.model_params['hidden_size'])
        model_dict = torch.load(classifier_path)
        self.classifier.load_state_dict(model_dict['classifier_state_dict'])

        if torch.cuda.is_available():
            self.classifier = self.classifier.cuda()

        
        # Set the gradient computation behavior for the model
        for param in self.classifier.parameters():
            param.requires_grad = False
        
        self.classifier.eval()


    # def binary_score(self, text):
    #     """
    #     text: list of strings
    #     """
        
    #     score = []
    #     for i in range(0, len(text), self.batch_size):
    #         batch = text[i:i+self.batch_size]
    #         # convert text to representations
    #         tokens = self.tokenizer(batch, return_tensors='pt', padding=True)

    #         for k, v in tokens.items():
    #             tokens[k] = to_var(v)

    #         with torch.no_grad():
    #             representations = self.model(**tokens).pooler_output
    #             score.extend(self.classifier(representations).cpu().numpy().tolist())

    #     return score
        
        # classification_threshold = 0.5
        # # get the binary classification
        # classification = torch.cat(score, dim=0).cpu().numpy() > classification_threshold
        # return classification

    # def get_sentence_representations(self, text):
    #     # convert text to representations
    #     tokens = self.tokenizer(text, return_tensors='pt', padding=True, return_dict=True).pooler_output

    #     for k, v in tokens.items():
    #         tokens[k] = to_var(v)

    #     with torch.no_grad():
    #         return self.model(**tokens)

    def get_unordred_pairs_score(self, representations):
        """
        text: list of strings
        """

        scores = []
        # indices = []

        # representations = OpinionRepresenter(text, self.tokenizer, self.model, batch_size=self.batch_size)

        for i in range(len(representations)):
            representation_1 = representations.get_sentence_representation(i)
            # print(representation_1.shape)
            temp_scores = torch.empty((0,))
            # temp_indices = []
            for j in range(i+1, len(representations), self.batch_size):
                # print("Memory Usage:", torch.cuda.memory_allocated(), 'bytes')

                # selected_indices = [k for k in range(j, j+self.batch_size)]
                # temp_indices.extend(selected_indices)
                # others = representations.get_sentence_representations(selected_indices)
                others = representations.get_sentence_representations(range(j, j+self.batch_size))

                # duplicate representation_1 to match the length of representations
                representation_1_batch = representation_1.repeat(len(others), 1)

                # process the batch
                # temp_scores.append(self.classifier(representation_1_batch, others).cpu())
                temp_scores = torch.cat((temp_scores, self.classifier(representation_1_batch, others).cpu()))
                # # free up memory
                # del others
                # del representation_1_batch
            
            # scores.append(torch.cat(temp_scores))
            scores.append(temp_scores)
            # scores.append(torch.cat(temp_scores) if temp_scores else torch.Tensor([]))
            # indices.append(temp_indices)

        return scores
        # return scores, indices

    def get_unodered_pairs_index(self, i, j):
        return i + j + 1
        
    def get_ordered_pairs_score(self, representations):
        """
        text: list of strings
        """

        scores = []

        # representations = OpinionRepresenter(text, self.tokenizer, self.model, batch_size=self.batch_size)
        for i in range(len(representations)):
            representation_1 = representations.get_sentence_representation(i)

            temp_scores = torch.empty((0,))
            for j in range(0, len(representations), self.batch_size):
                # print("Memory Usage:", torch.cuda.memory_allocated(), 'bytes')
                others = representations.get_sentence_representations((k for k in range(j, j+self.batch_size) if k != i))
                if others.nelement() == 0:
                    continue
                representation_1_batch = representation_1.repeat(len(others), 1)
                # process the batch
                # temp_scores.append(self.classifier.inference(representation_1_batch, others).cpu())
                temp_scores = torch.cat((temp_scores, self.classifier.inference(representation_1_batch, others).cpu()))
                # # free up memory
                # del others
                # del representation_1_batch
            # scores.append(torch.cat(temp_scores) if temp_scores else torch.Tensor([]))
            # scores = torch.cat((scores, temp_scores))
            scores.append(temp_scores)

        return scores
    
    def get_sentiment_score(self, representations):
        scores = torch.empty((0, 1))
        # representations = OpinionRepresenter(text, self.tokenizer, self.model, batch_size=self.batch_size)
        print(scores.shape)
        for i in range(0, len(representations), self.batch_size):
            representation_batch = representations.get_sentence_representations(range(i, i+self.batch_size))
            scores = torch.cat((scores, self.classifier(representation_batch).cpu()))
            # scores.append(self.classifier(representation_1).cpu())
        print(scores.shape)
        return scores if scores.nelement() > 0 else torch.Tensor([])
   
    def get_aggregated_score(self, representations, posts):
        scores = self.get_sentiment_score(representations)
        print(scores.device)
        # get the indices of the sentneces that pass the threshold
        passed_threshold = scores >= self.score_threshold
        # get the indices when passed_threshold is True
        indices = torch.nonzero(passed_threshold, as_tuple=True)[0]
        related_post_id = set()
        for i in indices:
            related_post_id.add(posts[i][2])
        return len(related_post_id) / len(posts)
        # passed_threshold = scores[scores >= self.score_threshold]
        # return torch.mean(passed_threshold).item() if passed_threshold.nelement() > 0 else 0.0
    
    def get_ordered_given_pairs_score(self, r1, r2):
        # scores = []
        scores = torch.empty((0,))

        for i in range(0, len(r1), self.batch_size):
            representation_1_batch = r1.get_sentence_representations(range(i, i+self.batch_size))
            others = r2.get_sentence_representations(range(i, i+self.batch_size))
            # scores.append(self.classifier.inference(representation_1_batch, others))
            scores = torch.cat((scores, self.classifier.inference(representation_1_batch, others).cpu()))

        return scores if scores.nelement() > 0 else torch.Tensor([])

    def get_unordered_given_pairs_score(self, r1, r2):
        # scores = []
        scores = torch.empty((0,))

        for i in range(0, len(r1), self.batch_size):
            representation_1_batch = r1.get_sentence_representations(range(i, i+self.batch_size))
            others = r2.get_sentence_representations(range(i, i+self.batch_size))
            # scores.append(self.classifier(representation_1_batch, others))
            scores = torch.cat((scores, self.classifier(representation_1_batch, others).cpu()))

        return scores if scores.nelement() > 0 else torch.Tensor([])
     
    
    def get_odered_pairs_index(self, i, j):
        if j < i:
            return j
        return j + 1 

    def ranking(self, text):
        """
        return [4 3 2 0 1] [[(2, 0.6068866848945618)], [], [(0, 0.6068866848945618)], [(4, 0.8789092898368835)], [(3, 0.8789092898368835)]] [0.60688668 0.         0.60688668 0.87890929 0.87890929]
        """
        relatedness = np.zeros(len(text), dtype=np.float32)
        related = [[] for _ in range(len(text))]

        if self.model_params['task'] == "similarity":
            # scores, indices = self.get_unordred_pairs_score(text)
            scores = self.get_unordred_pairs_score(text)
            for i in range(len(scores)):
                passed_threshold = torch.where(scores[i] >= self.score_threshold)[0]
                # if passed_threshold:
                for j in passed_threshold:
                    score = scores[i][j].item()
                    global_j = self.get_unodered_pairs_index(i, j)
                    relatedness[i] += score
                    relatedness[global_j] += score
                    related[i].append((global_j, score))
                    related[global_j].append((i, score))
                # for j in range(len(scores[i])):
                #     # print(scores[i][j], self.score_threshold, related)
                #     score = scores[i][j].item()
                #     # assert indices[i][j] == self.get_unodered_pairs_index(i, j)
                #     # exit()
                #     if score > self.score_threshold:
                #         global_j = self.get_unodered_pairs_index(i, j)
                #         relatedness[i] += score
                #         relatedness[global_j] += score
                #         related[i].append((global_j, score))
                #         related[global_j].append((i, score))
        elif self.model_params['task'] == 'entailment':
            scores = self.get_ordered_pairs_score(text)
            for i in range(len(scores)):
                passed_threshold = torch.where(scores[i] >= self.score_threshold)[0]
                # if passed_threshold:
                for j in passed_threshold:
                    score = scores[i][j].item()
                    global_j = self.get_odered_pairs_index(i, j)
                    # relatedness[i] += score
                    relatedness[global_j] += score
                    # related[i].append((global_j, score))
                    related[global_j].append((i, score))
        elif self.model_params['task'] == 'sentiment':
            # scores = self.binary_score(text)e
            raise NotImplementedError
        else:
            raise ValueError("Invalid task: {}".format(self.model_params['task']))
        
        
        return np.argsort(relatedness)[::-1], related, relatedness
    
    def get_related(self, text, top_k=5):
        ranked_indices, related, relatedness = self.ranking(text)
        # print(ranked_indices, related, relatedness)
        skipped = set()

        ranked = []

        i = 0
        while len(ranked) < top_k and i < len(ranked_indices):

            current_index = ranked_indices[i]
            if current_index not in skipped:
                # print(current_index)
                for j, _ in related[current_index]:
                    skipped.add(j)
                if relatedness[current_index] == 0:
                    break
                ranked.append([current_index, related[current_index], relatedness[current_index]])
            i += 1

        return ranked

    
def group_opinions(posts, text, ranking, max_similar_opinions):
    opinions_return = []

    related_sentences_ids = set()
    for o_id, related_ids, relatedness in ranking:
        splited_text, likes, post_id = posts[o_id]
        if o_id in related_sentences_ids:
            continue
        related_sentences_ids.add(o_id)

        similar_opinions = []

        total_likes = likes
        unique_posts_ids = set([post_id])

        for r_id, score in related_ids:
            similar_opinions.append((text[r_id], score))
            _, r_likes, r_post_id = posts[r_id]
            related_sentences_ids.add(r_id)
            if r_post_id not in unique_posts_ids:
                total_likes += r_likes
                unique_posts_ids.add(r_post_id)
        # print(total_likes)
        # print("--")
        

        opinions_return.append((splited_text, len(unique_posts_ids), total_likes, relatedness, similar_opinions[:max_similar_opinions]))
    return opinions_return
    
def process(posts, text, max_similar_opinions=10):
    # around 6 GB GPU memory
    logging.info("Start ================================================")
    g = OpinionGrouper('bin/2023-Apr-10-20:13:14/E4.pytorch', batch_size=166, score_threshold=0.6296)
    logging.debug("Memory Usage:", torch.cuda.memory_allocated(), 'bytes')
    pretrained_model_name = g.pretrained_model_name

    representer = OpinionRepresenter(text, pretrained_model_name, batch_size=g.batch_size)
    print(pretrained_model_name)
    # semtiment
    semtiment = g.get_aggregated_score(representer, posts)
    # del g
    logging.info("Sentiment Finished ================================================")
    logging.debug("Memory Usage:", torch.cuda.memory_allocated(), 'bytes')

    # similarity
    g = OpinionGrouper('bin/2023-Apr-08-10:29:46/E24.pytorch', batch_size=166, score_threshold=0.6)
    # print(g.get_unordred_pairs_score(text))
    ranking = g.get_related(representer)
    # del g
    # print(ranking)
    similar_opinions_return = group_opinions(posts, text, ranking, max_similar_opinions)
    logging.info("Similarity Finished ================================================")
    logging.debug("Memory Usage:", torch.cuda.memory_allocated(), 'bytes')
    # import time
    # time.sleep(10)
    # entailment
    g = OpinionGrouper('bin/2023-Apr-09-00:49:15/E4.pytorch', batch_size=166, score_threshold=0.109159)
    ranking = g.get_related(representer)
    # del g
    # print(ranking)
    entailed_opinions_return = group_opinions(posts, text, ranking, max_similar_opinions)
    logging.info("Entailment Finished ================================================")
    logging.debug("Memory Usage:", torch.cuda.memory_allocated(), 'bytes')


    return semtiment, similar_opinions_return, entailed_opinions_return
    
if __name__ == '__main__':


    post = [('I go to school', 2), ('I go to school.', 2), ('I don\'t go to school', 1), ('I go to school by bus.', 0), ('I eat dinner.', 0), ('I just ate dinner.', 0)]

    text = [p[0] for p in post]
    # g = OpinionGrouper('bin/2023-Apr-09-00:49:15/E4.pytorch', batch_size=2, score_threshold=0.6)
    # # print(g.get_unordred_pairs_score(text))
    # print(g.get_related(text))
    # process_posts(post)

    semtiment, similar_opinions_return, entailed_opinions_return = process(post, text)
    print(semtiment)
    print(similar_opinions_return)
    print(entailed_opinions_return)
