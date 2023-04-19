import numpy as np

from grouping import OpinionRepresenter, OpinionGrouper
from datasets import load_dataset
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

if __name__ == '__main__':
    split = 'validation'

    # # similarity
    # dataset = load_dataset('glue', 'stsb')[split]

    # # g = OpinionGrouper('bin/lstm_similarity/E49.pytorch', batch_size=128, score_threshold=0.6)  # lstm vae
    # # representer1 = OpinionRepresenter(dataset['sentence1'], 'bert-base-uncased', batch_size=g.batch_size, local_model_type='lstm', local_model_path='bin/lstm/E29.pytorch')
    # # representer2 = OpinionRepresenter(dataset['sentence2'], 'bert-base-uncased', batch_size=g.batch_size, local_model_type='lstm', local_model_path='bin/lstm/E29.pytorch')

    # g = OpinionGrouper('bin/2023-Apr-08-10:29:46/E24.pytorch', batch_size=128, score_threshold=0.6)
    # pretrained_model_name = g.pretrained_model_name

    # representer1 = OpinionRepresenter(dataset['sentence1'], pretrained_model_name, batch_size=g.batch_size)
    # representer2 = OpinionRepresenter(dataset['sentence2'], pretrained_model_name, batch_size=g.batch_size)

    # similarity_score = g.get_unordered_given_pairs_score(representer1, representer2)

    # # get the Pearson-Spearman Corr
    # from scipy.stats import pearsonr, spearmanr
    # corr, _ = pearsonr(similarity_score, dataset['label'])
    # print('Pearsons correlation: %.3f' % corr)
    # corr, _ = spearmanr(similarity_score, dataset['label'])
    # print('Spearmans correlation: %.3f' % corr)

    # print(similarity_score.shape)

    # exit()


    # # sentiment
    # dataset = load_dataset('glue', 'sst2')[split]

    # # g = OpinionGrouper('bin/lstm_sentiment/E29.pytorch', batch_size=128, score_threshold=0.6)  # lstm vae
    # # representer = OpinionRepresenter(dataset['sentence'], 'bert-base-uncased', batch_size=g.batch_size, local_model_type='lstm', local_model_path='bin/lstm/E29.pytorch')


    # g = OpinionGrouper('bin/2023-Apr-10-20:13:14/E4.pytorch', batch_size=128, score_threshold=0.6)
    # pretrained_model_name = g.pretrained_model_name


    # print(dataset['sentence'][:3])
    # print(len(dataset))
    # representer = OpinionRepresenter(dataset['sentence'], pretrained_model_name, batch_size=g.batch_size)

    # sentiment_score = g.get_sentiment_score(representer)

    # print(sentiment_score.shape)

    # score = sentiment_score


    # entailment
    split = 'test'
    dataset = load_dataset('snli')[split].map(lambda x: {'label': int(x['label'] == 0)})

    # g = OpinionGrouper('bin/lstm_entailment/E44.pytorch', batch_size=128, score_threshold=0.6)  # lstm vae

    # premise_representer = OpinionRepresenter(dataset['premise'], 'bert-base-uncased', batch_size=g.batch_size, local_model_type='lstm', local_model_path='bin/lstm/E29.pytorch')
    # hypothesis_representer = OpinionRepresenter(dataset['hypothesis'], 'bert-base-uncased', batch_size=g.batch_size, local_model_type='lstm', local_model_path='bin/lstm/E29.pytorch')

    # simcse
    g = OpinionGrouper('bin/2023-Apr-09-00:49:15/E4.pytorch', batch_size=128, score_threshold=0.6)
    pretrained_model_name = g.pretrained_model_name

    
    premise_representer = OpinionRepresenter(dataset['premise'], pretrained_model_name, batch_size=g.batch_size)
    hypothesis_representer = OpinionRepresenter(dataset['hypothesis'], pretrained_model_name, batch_size=g.batch_size)

    entailment_score = g.get_ordered_given_pairs_score(premise_representer, hypothesis_representer)

    print(entailment_score.shape)

    score = entailment_score



    fpr, tpr, thresholds = roc_curve(dataset['label'], score.cpu().detach().numpy())

    print(fpr.shape, tpr.shape, thresholds.shape)
    roc_auc = auc(fpr, tpr)

    # Compute the Youden's J statistic for each threshold
    j_scores = tpr - fpr
    
    best_threshold_idx = np.argmax(j_scores)
    best_threshold = thresholds[best_threshold_idx]

    print('Best Threshold=%f, J-Score=%.3f, AUC=%.3f, FPR=%.3f, TPR=%.3f' % (best_threshold, j_scores[best_threshold_idx], roc_auc, fpr[best_threshold_idx], tpr[best_threshold_idx]))
    # print("Best threshold:", best_threshold)

    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic (ROC) curve')
    plt.legend(loc="lower right")
    plt.savefig("z1.png")

    # report the f1 score
    y_pred = (score >= best_threshold).cpu().detach().numpy()
    print(f"F1={f1_score(dataset['label'], y_pred)}")

