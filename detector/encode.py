import pandas as pd
from settings import settings
from gensim.models.word2vec import Word2Vec
from utils import get_sequence, get_blocks, check_path


def generate_embeddings(ast_path, pairs_path, size=settings.vec_size):

    source = pd.read_pickle(ast_path)
    pairs = pd.read_pickle(pairs_path)
    train_ids = pairs['id1'].append(pairs['id2']).unique()

    trees = pd.DataFrame(columns=source.columns)
    for _, row in source.iterrows():
        if row['id'] in train_ids:
            trees = trees.append(row)
    print("-----read trees------")
    print(trees)
    corpus = trees['ast'].apply(get_sequence)
    str_corpus = [' '.join(c) for c in corpus]
    trees['code'] = pd.Series(str_corpus)
    print("-----read corpus------")
    print(corpus)
    w2v = Word2Vec(corpus, size=size, workers=16,
                   sg=1, max_final_vocab=3000)
    w2v.save(settings.w2v_model_path)


def tree_to_index(node, vocab, default):
    token = node.token
    result = [vocab[token].index if token in vocab else default]
    children = node.children
    for child in children:
        result.append(tree_to_index(child, vocab, default))
    return result


def ast_to_blocks(r, vocab, default):
    blocks = []
    get_blocks(r, blocks)
    tree = []
    for b in blocks:
        btree = tree_to_index(b, vocab, default)
        tree.append(btree)
    return tree

def generate_block_seqs(ast_path, block_path):
    source = pd.read_pickle(ast_path)
    word2vec = Word2Vec.load(settings.w2v_model_path).wv
    vocab = word2vec.vocab
    max_token = word2vec.syn0.shape[0]

    blocks = pd.DataFrame(source, copy=True)
    blocks['ast'] = blocks['ast'].apply(
        lambda x: ast_to_blocks(x, vocab, max_token))

    blocks.to_pickle(block_path)
    return blocks

def merge_block_pairs(pairs_path, block_path):
    blocks = pd.read_pickle(settings.java_block_path)
    pairs = pd.read_pickle(pairs_path)
    pairs['id1'] = pairs['id1'].astype(int)
    pairs['id2'] = pairs['id2'].astype(int)
    df = pd.merge(pairs, blocks, how='left',
                    left_on='id1', right_on='id')
    df = pd.merge(df, blocks, how='left',
                    left_on='id2', right_on='id')
    df.drop(['id_x', 'id_y'], axis=1, inplace=True)
    df.dropna(inplace=True)

    df.to_pickle(block_path)

def main():
    # generate_embeddings(settings.java_ast_path, settings.java_pairs_path)
    generate_block_seqs(settings.java_ast_path, settings.java_block_path)
    merge_block_pairs(settings.train_pairs_path, settings.train_block_path)
    merge_block_pairs(settings.test_pairs_path, settings.test_block_path)

if __name__ == "__main__":
    main()
