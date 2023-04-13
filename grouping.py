
from transformers import AutoModel, AutoTokenizer
import torch
import os, json


from setup import load
load()
from model_classification import SimilarClassifier, SentimentClassifier, EntailmentClassifier
from utils import to_var

import numpy as np

class OpinionRepresenter:
    def __init__(self, text, tokenizer, model, batch_size=128):
        self.representations = []
        for i in range(0, len(text), batch_size):
            batch = text[i:i+batch_size]
            # convert text to representations
            tokens = tokenizer(batch, return_tensors='pt', padding=True)
            print(tokens)
            for k, v in tokens.items():
                tokens[k] = to_var(v)

            with torch.no_grad():
                representations = model(**tokens).pooler_output
                print(representations)
                self.representations.extend(representations)
    
    def get_sentence_representation(self, index):
        return self.representations[index]

    def get_sentence_representations(self, indices):
        sliced = []
        for i in indices:
            if i < len(self.representations):
                sliced.append(self.representations[i])
        return torch.stack(sliced)

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
        pretrained_model_name = self.model_params['pretrained_model']
        self.model = AutoModel.from_pretrained(pretrained_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)

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
            self.model = self.model.cuda()
            self.classifier = self.classifier.cuda()
        
        self.classifier.eval()


    def binary_score(self, text):
        """
        text: list of strings
        """
        
        score = []
        for i in range(0, len(text), self.batch_size):
            batch = text[i:i+self.batch_size]
            # convert text to representations
            tokens = self.tokenizer(batch, return_tensors='pt', padding=True)

            for k, v in tokens.items():
                tokens[k] = to_var(v)

            with torch.no_grad():
                representations = self.model(**tokens).pooler_output
                score.extend(self.classifier(representations).cpu().numpy().tolist())

        return score
        
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

    def get_unordred_pairs_score(self, text):
        """
        text: list of strings
        """

        scores = []
        indices = []

        representations = OpinionRepresenter(text, self.tokenizer, self.model, batch_size=self.batch_size)

        for i in range(0, len(text)):
            representation_1 = representations.get_sentence_representation(i)
            print(representation_1.shape)
            temp_scores = []
            temp_indices = []
            for j in range(i+1, len(text), self.batch_size):

                selected_indices = [k for k in range(j, j+self.batch_size)]
                temp_indices.extend(selected_indices)
                others = representations.get_sentence_representations(selected_indices)

                # duplicate representation_1 to match the length of representations
                representation_1_batch = representation_1.repeat(len(others), 1)

                # process the batch
                temp_scores.append(self.classifier(representation_1_batch, others))
            
            scores.append(torch.cat(temp_scores) if temp_scores else [])  # should be extend?
            indices.append(temp_indices)

        return scores, indices
        
    def get_ordered_pairs_score(self, text):
        """
        text: list of strings
        """

        scores = []

        representations = OpinionRepresenter(text, self.tokenizer, self.model, batch_size=self.batch_size)
        for i in range(len(text)):
            representation_1 = representations.get_sentence_representation(i)

            temp_scores = []
            for j in range(0, len(text), self.batch_size):
                others = representations.get_sentence_representations((k for k in range(j, j+self.batch_size) if k != i))

                # process the batch
                temp_scores.extend(self.classifier(representation_1, others))
            scores.append(temp_scores)

        return scores

    def ranking(self, text):
        """
        return [4 3 2 0 1] [[(2, 0.6068866848945618)], [], [(0, 0.6068866848945618)], [(4, 0.8789092898368835)], [(3, 0.8789092898368835)]] [0.60688668 0.         0.60688668 0.87890929 0.87890929]
        """
        relatedness = np.zeros(len(text))
        related = [[] for _ in range(len(text))]

        if self.model_params['task'] == "similarity":
            scores, indices = self.get_unordred_pairs_score(text)
            for i in range(len(scores)):
                for j in range(len(scores[i])):
                    # print(scores[i][j], self.score_threshold, related)
                    score = scores[i][j].item()
                    if score > self.score_threshold:
                        global_j = indices[i][j]
                        relatedness[i] += score
                        relatedness[global_j] += score
                        related[i].append((global_j, score))
                        related[global_j].append((i, score))
        elif self.model_params['task'] == 'entailment':
            scores = self.get_ordered_pairs_score(text)
        elif self.model_params['task'] == 'sentiment':
            scores = self.binary_score(text)
        
        
        return np.argsort(relatedness)[::-1], related, relatedness
    
    def get_related(self, text, top_k=5):
        ranked_indices, related, relatedness = self.ranking(text)
        print(ranked_indices, related, relatedness)
        skipped = set()

        ranked = []

        i = 0
        while len(ranked) < top_k and i < len(ranked_indices):

            current_index = ranked_indices[i]
            if current_index not in skipped:
                print(current_index)
                for j, _ in related[current_index]:
                    skipped.add(j)
                ranked.append([current_index, related[current_index], relatedness[current_index]])
            i += 1

        return ranked
    

    

    
if __name__ == '__main__':


    post = [('I go to school', 2), ('I don\'t go to school', 1), ('I go to school by bus.', 0), ('I eat dinner.', 0), ('I just ate dinner.', 0)]

    # text = [p[0] for p in post]
    # g = OpinionGrouper('bin/2023-Apr-08-10:29:46/E24.pytorch', batch_size=128, score_threshold=0.6)
    # print(g.get_unordred_pairs_score(text))
    # process_posts(post)
