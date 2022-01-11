import pandas as pd
import os
from gensim.models.word2vec import Word2Vec
from utils import get_sequence, get_blocks

train_data_path = "./data/train"
embedding_path = train_data_path + "/embedding"
input_file_path = train_data_path + "/java_pairs.pkl"


def embed_all(source, size=128):

    if not os.path.exists(train_data_path+'/embedding'):
        os.mkdir(train_data_path+'/embedding')

    pairs = pd.read_pickle(input_file_path)

    train_ids = pairs['id1'].append(pairs['id2']).unique()

    trees = pd.DataFrame(columns=source.columns)
    for idx, row in source.iterrows():
        if row['id'] in train_ids:
            trees = trees.append(row)

    corpus = trees['ast'].apply(get_sequence)
    w2v = Word2Vec(corpus, vector_size=size, workers=16,
                   sg=1, max_final_vocab=3000)
    w2v.save(embedding_path + "/w2v_" + str(size))


def tree_to_index(node, vocab, default):
    token = node.token
    result = [vocab[token].index if token in vocab else default]
    children = node.children
    for child in children:
        result.append(tree_to_index(child))
    return result


def trans2seq(r):
    blocks = []
    get_blocks(r, blocks)
    tree = []
    for b in blocks:
        btree = tree_to_index(b)
        tree.append(btree)
    return tree


def generate_block_seqs(source, file_path):
    word2vec = Word2Vec.load(embedding_path+"/w2v_128").wv
    vocab = word2vec.vocab
    max_token = word2vec.syn0.shape[0]

    blocks = pd.DataFrame(source, copy=True)
    blocks['ast'] = blocks['ast'].apply(
        lambda x: trans2seq(x, vocab, max_token))

    blocks.to_pickle(file_path)
    return blocks


def main():
    source_all = pd.read_pickle("./data/java_ast.pkl")
    embed_all(source_all)
    generate_block_seqs(source_all, "./data/java_blocks.pkl")


if __name__ == "__main__":
    main()
