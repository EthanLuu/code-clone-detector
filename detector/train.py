import pandas as pd
import os
import torch
import numpy as np
import pickle
import dill
from settings import settings
from model import BatchProgramCC
from torch.autograd import Variable
from gensim.models.word2vec import Word2Vec
from sklearn.metrics import precision_recall_fscore_support


categories = 5
HIDDEN_DIM = 100
ENCODE_DIM = 128
LABELS = 1
EPOCHS = 5
BATCH_SIZE = 32
USE_GPU = False


def get_batch(dataset, idx, bs):
    tmp = dataset.iloc[idx: idx+bs]
    x1, x2, labels = [], [], []
    for _, item in tmp.iterrows():
        x1.append(item['ast_x'])
        x2.append(item['ast_y'])
        labels.append([item['label']])
    return x1, x2, torch.FloatTensor(labels)


def train():
    train_data = pd.read_pickle(settings.train_block_path).sample(frac=1)
    word2vec = Word2Vec.load(settings.w2v_model_path).wv
    MAX_TOKENS = word2vec.syn0.shape[0]
    EMBEDDING_DIM = word2vec.syn0.shape[1]
    embeddings = np.zeros((MAX_TOKENS + 1, EMBEDDING_DIM), dtype="float32")
    embeddings[:word2vec.syn0.shape[0]] = word2vec.syn0

    model = BatchProgramCC(EMBEDDING_DIM, HIDDEN_DIM, MAX_TOKENS+1, ENCODE_DIM, LABELS, BATCH_SIZE,
                           USE_GPU, embeddings)

    parameters = model.parameters()
    optimizer = torch.optim.Adamax(parameters)
    loss_function = torch.nn.BCELoss()
    print('Start training...')
    for t in range(1, categories+1):
        model_path = "./models/model_" + str(t) + ".pkl"
        if os.path.exists(model_path):
            continue
        # 筛选出当前 type 的克隆代码对，克隆标记为 1，不克隆为 0
        train_data_t = train_data[train_data['label'].isin([t, 0])]
        train_data_t.loc[train_data_t['label'] > 0, 'label'] = 1
        print(train_data_t)
        # training procedure
        for _ in range(EPOCHS):
            # training epoch
            i = 0
            while i < len(train_data_t):
                try:
                    batch = get_batch(train_data_t, i, BATCH_SIZE)
                    i += BATCH_SIZE
                    train1_inputs, train2_inputs, train_labels = batch
                    if USE_GPU:
                        train1_inputs, train2_inputs, train_labels = train1_inputs, train2_inputs, train_labels.cuda()
                    model.zero_grad()
                    model.batch_size = len(train_labels)
                    model.hidden = model.init_hidden()
                    output = model(train1_inputs, train2_inputs)

                    loss = loss_function(output, Variable(train_labels))
                    loss.backward()
                    optimizer.step()
                    print(str(i) + " good")
                except:
                    print(str(i) + " bad")
                    continue
        # save model
        f = open(model_path, 'wb')
        dill.dump(model, f)
        f.close()
        print(model_path + " generated")


def test():
    precision, recall, f1 = 0, 0, 0
    test_data = pd.read_pickle(settings.test_block_path).sample(frac=1)
    loss_function = torch.nn.BCELoss()
    for t in range(1, categories+1):
        test_data_t = test_data[test_data['label'].isin([t, 0])]
        test_data_t.loc[test_data_t['label'] > 0, 'label'] = 1
        model_path = "./models/model_" + str(t) + ".pkl"
        f = open(model_path, 'rb')
        model = dill.load(f)
        f.close()
        print("Testing-%d..." % t)
        # testing procedure
        predicts = []
        trues = []
        total_loss = 0.0
        total = 0.0
        i = 0
        while i < len(test_data_t):
            batch = get_batch(test_data_t, i, BATCH_SIZE)
            i += BATCH_SIZE
            test1_inputs, test2_inputs, test_labels = batch
            if USE_GPU:
                test_labels = test_labels.cuda()

            model.batch_size = len(test_labels)
            model.hidden = model.init_hidden()
            output = model(test1_inputs, test2_inputs)

            loss = loss_function(output, Variable(test_labels))

            # calc testing acc
            predicted = (output.data > 0.5).cpu().numpy()
            predicts.extend(predicted)
            trues.extend(test_labels.cpu().numpy())
            total += len(test_labels)
            total_loss += loss.item() * len(test_labels)
        weights = [0, 0.005, 0.001, 0.002, 0.010, 0.982]
        p, r, f, _ = precision_recall_fscore_support(
            trues, predicts, average='binary')
        precision += weights[t] * p
        recall += weights[t] * r
        f1 += weights[t] * f
        print("Type-" + str(t) + ": " + str(p) +
              " " + str(r) + " " + str(f))

    print("Total testing results(P,R,F1):%.3f, %.3f, %.3f" %
          (precision, recall, f1))


def main():
    train()
    test()


if __name__ == "__main__":
    main()
