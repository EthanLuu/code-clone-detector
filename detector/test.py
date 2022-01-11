import pandas as pd
import os
import sys
import warnings
import torch
import time
import numpy as np
import warnings
from gensim.models.word2vec import Word2Vec
import pickle
from model import BatchProgramCC
from torch.autograd import Variable
from sklearn.metrics import precision_recall_fscore_support
warnings.filterwarnings('ignore')

class Pipeline:
    # 比例，目录，语言
    def __init__(self,  ratio, root, language):
        self.ratio = ratio
        self.root = root
        self.language = language
        self.sources = None
        self.blocks = None
        self.pairs = None
        self.train_file_path = None
        self.dev_file_path = None
        self.test_file_path = None
        self.size = None

    # 预处理源代码文件
    def parse_source(self, output_file, option):
        path = self.root+self.language+'/'+output_file
        if os.path.exists(path) and option == 'existing':
            source = pd.read_pickle(path)
        else:
            if self.language is 'c':
                from pycparser import c_parser
                parser = c_parser.CParser()
                source = pd.read_pickle(self.root+self.language+'/programs.pkl')
                source.columns = ['id', 'code', 'label']
                source['code'] = source['code'].apply(parser.parse)
                source.to_pickle(path)
            else:
                import javalang
                def parse_program(func):
                    tokens = javalang.tokenizer.tokenize(func)
                    parser = javalang.parser.Parser(tokens)
                    tree = parser.parse_member_declaration()
                    return tree
                source = pd.read_csv(self.root+self.language+'/bcb_funcs_all.tsv', sep='\t', header=None, encoding='utf-8')
                source.columns = ['id', 'code']
                source['code'] = source['code'].apply(parse_program)
                source.to_pickle(path)
        self.sources = source
        return source

    # create clone pairs
    def read_pairs(self, filename):
        pairs = pd.read_pickle(self.root+self.language+'/'+filename)
        self.pairs = pairs

    # split data for training, developing and testing
    def split_data(self):
        data_path = self.root+self.language+'/'
        data = self.pairs
        data_num = len(data)
        ratios = [int(r) for r in self.ratio.split(':')]
        train_split = int(ratios[0]/sum(ratios)*data_num)
        val_split = train_split + int(ratios[1]/sum(ratios)*data_num)

        data = data.sample(frac=1, random_state=666)
        train = data.iloc[:train_split]
        dev = data.iloc[train_split:val_split]
        test = data.iloc[val_split:]

        def check_or_create(path):
            if not os.path.exists(path):
                os.mkdir(path)
        train_path = data_path+'train/'
        check_or_create(train_path)
        self.train_file_path = train_path+'train_.pkl'
        train.to_pickle(self.train_file_path)

        dev_path = data_path+'dev/'
        check_or_create(dev_path)
        self.dev_file_path = dev_path+'dev_.pkl'
        dev.to_pickle(self.dev_file_path)

        test_path = data_path+'test/'
        check_or_create(test_path)
        self.test_file_path = test_path+'test_.pkl'
        test.to_pickle(self.test_file_path)

    # construct dictionary and train word embedding
    def dictionary_and_embedding(self, input_file, size):
        self.size = size
        data_path = self.root+self.language+'/'
        if not input_file:
            input_file = self.train_file_path
        pairs = pd.read_pickle(input_file)
        
        train_ids = pairs['id1'].append(pairs['id2']).unique()

        trees = self.sources.set_index('id',drop=False).loc[train_ids]
        if not os.path.exists(data_path+'train/embedding'):
            os.mkdir(data_path+'train/embedding')
        if self.language is 'c':
            sys.path.append('../')
            from prepare_data import get_sequences as func
        else:
            from utils import get_sequence as func

        def trans_to_sequences(ast):
            sequence = []
            func(ast, sequence)
            return sequence
        corpus = trees['code'].apply(trans_to_sequences)
        str_corpus = [' '.join(c) for c in corpus]
        trees['code'] = pd.Series(str_corpus)
        # trees.to_csv(data_path+'train/programs_ns.tsv')
        from gensim.models.word2vec import Word2Vec
        w2v = Word2Vec(corpus, size=size, workers=16, sg=1, max_final_vocab=3000)
        w2v.save(data_path+'train/embedding/node_w2v_' + str(size))

    # generate block sequences with index representations
    def generate_block_seqs(self):
        if self.language is 'c':
            from prepare_data import get_blocks as func
        else:
            from utils import get_blocks_v1 as func
        from gensim.models.word2vec import Word2Vec

        word2vec = Word2Vec.load(self.root+self.language+'/train/embedding/node_w2v_128').wv
        vocab = word2vec.vocab
        max_token = word2vec.syn0.shape[0]
        
        def tree_to_index(node):
            token = node.token
            result = [vocab[token].index if token in vocab else max_token]
            children = node.children
            for child in children:
                result.append(tree_to_index(child))
            return result

        def trans2seq(r):
            blocks = []
            func(r, blocks)
            tree = []
            for b in blocks:
                btree = tree_to_index(b)
                tree.append(btree)
            return tree
        trees = pd.DataFrame(self.sources, copy=True)
        trees['code'] = trees['code'].apply(trans2seq)
        if 'label' in trees.columns:
            trees.drop('label', axis=1, inplace=True)
        self.blocks = trees

    # merge pairs
    def merge(self,data_path,part):
        pairs = pd.read_pickle(data_path)
        pairs['id1'] = pairs['id1'].astype(int)
        pairs['id2'] = pairs['id2'].astype(int)
        df = pd.merge(pairs, self.blocks, how='left', left_on='id1', right_on='id')
        df = pd.merge(df, self.blocks, how='left', left_on='id2', right_on='id')
        df.drop(['id_x', 'id_y'], axis=1,inplace=True)
        df.dropna(inplace=True)

        df.to_pickle(self.root+self.language+'/'+part+'/blocks.pkl')
        print(df)

    # run for processing data to train
    def run(self):
        print('parse source code...')
        self.parse_source(output_file='ast.pkl',option='existing')
        print('read id pairs...')
        if self.language is 'c':
            self.read_pairs('oj_clone_ids.pkl')
        else:
            self.read_pairs('bcb_pair_ids.pkl')
        print('split data...')
        self.split_data()
        print('train word embedding...')
        # self.dictionary_and_embedding(None,128)
        print('generate block sequences...')
        self.generate_block_seqs()
        print('merge pairs and blocks...')
        self.merge(self.train_file_path, 'train')
        self.merge(self.dev_file_path, 'dev')
        self.merge(self.test_file_path, 'test')


def get_batch(dataset, idx, bs):
    tmp = dataset.iloc[idx: idx+bs]
    x1, x2, labels = [], [], []
    for _, item in tmp.iterrows():
        x1.append(item['code_x'])
        x2.append(item['code_y'])
        labels.append([item['label']])
    return x1, x2, torch.FloatTensor(labels)

def processData():
    ppl = Pipeline('3:1:1', 'data/', "java")
    ppl.run()

def main():
    categories = 5
    lang = "java"
    root = 'data/'
    
    train_data = pd.read_pickle(root+lang+'/train/blocks.pkl').sample(frac=1)
    test_data = pd.read_pickle(root+lang+'/test/blocks.pkl').sample(frac=1)

    word2vec = Word2Vec.load(root+lang+"/train/embedding/node_w2v_128").wv
    MAX_TOKENS = word2vec.syn0.shape[0]
    EMBEDDING_DIM = word2vec.syn0.shape[1]
    embeddings = np.zeros((MAX_TOKENS + 1, EMBEDDING_DIM), dtype="float32")
    embeddings[:word2vec.syn0.shape[0]] = word2vec.syn0

    HIDDEN_DIM = 100
    ENCODE_DIM = 128
    LABELS = 1
    EPOCHS = 5
    BATCH_SIZE = 32
    USE_GPU = False
    
    model = BatchProgramCC(EMBEDDING_DIM,HIDDEN_DIM,MAX_TOKENS+1,ENCODE_DIM,LABELS,BATCH_SIZE,
                                   USE_GPU, embeddings)
    
    parameters = model.parameters()
    optimizer = torch.optim.Adamax(parameters)
    loss_function = torch.nn.BCELoss()
    print(train_data)
    precision, recall, f1 = 0, 0, 0
    print('Start training...')
    for t in range(1, categories+1):
        model_path = "./models/model_" + str(t) + ".pkl"
        if not os.path.exists(model_path):
            train_data_t = train_data[train_data['label'].isin([t, 0])]
            train_data_t.loc[train_data_t['label'] > 0, 'label'] = 1

            test_data_t = test_data[test_data['label'].isin([t, 0])]
            test_data_t.loc[test_data_t['label'] > 0, 'label'] = 1
            # training procedure
            for epoch in range(EPOCHS):
                # training epoch
                total_acc = 0.0
                total_loss = 0.0
                total = 0.0
                i = 0
                while i < len(train_data_t):
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
            # save model
            f = open(model_path,'wb')
            pickle.dump(model, f)
            f.close()
        else:
            f = open(model_path,'rb')
            model = pickle.load(f)

        print("Testing-%d..."%t)
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
        if lang == 'java':
            weights = [0, 0.005, 0.001, 0.002, 0.010, 0.982]
            p, r, f, _ = precision_recall_fscore_support(trues, predicts, average='binary')
            precision += weights[t] * p
            recall += weights[t] * r
            f1 += weights[t] * f
            print("Type-" + str(t) + ": " + str(p) + " " + str(r) + " " + str(f))

    print("Total testing results(P,R,F1):%.3f, %.3f, %.3f" % (precision, recall, f1))

if __name__ == "__main__":
    main()


