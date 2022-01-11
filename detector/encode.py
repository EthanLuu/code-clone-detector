import pandas as pd
from settings import settings
from gensim.models.word2vec import Word2Vec
from utils import get_sequence, get_blocks, check_path


def embed_all(source, size=settings.vec_size):

    check_path(settings.train_embedding_path)
    pairs = pd.read_pickle(settings.train_pairs_path)
    train_ids = pairs['id1'].append(pairs['id2']).unique()

    trees = pd.DataFrame(columns=source.columns)
    for _, row in source.iterrows():
        if row['id'] in train_ids:
            trees = trees.append(row)

    corpus = trees['ast'].apply(get_sequence)
    w2v = Word2Vec(corpus, size=size, workers=16,
                   sg=1, max_final_vocab=3000)
    w2v.save(settings.w2v_model_path)


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
    word2vec = Word2Vec.load(settings.w2v_model_path).wv
    vocab = word2vec.vocab
    max_token = word2vec.syn0.shape[0]

    blocks = pd.DataFrame(source, copy=True)
    blocks['ast'] = blocks['ast'].apply(
        lambda x: trans2seq(x, vocab, max_token))

    blocks.to_pickle(file_path)
    return blocks


def main():
    source_all = pd.read_pickle(settings.java_ast_path)
    embed_all(source_all)
    generate_block_seqs(source_all, settings.java_block_path)


if __name__ == "__main__":
    main()
