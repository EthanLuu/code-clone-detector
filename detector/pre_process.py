import pandas as pd
import os
import javalang
from utils import check_path
from settings import settings


def code_to_ast(func):
    tokens = javalang.tokenizer.tokenize(func)
    parser = javalang.parser.Parser(tokens)
    tree = parser.parse_member_declaration()
    return tree


def parse_source(file_path, pickle_path):
    # 将 [id, code] 的 csv 文件转换成 [id, code, ast] 的 pickle 文件
    check_path(settings.data_path)
    if os.path.exists(pickle_path):
        return pd.read_pickle(pickle_path)

    old_source = pd.read_csv(file_path, sep='\t',
                             header=None, encoding='utf-8')
    old_source.columns = ['id', 'code']
    new_source = pd.DataFrame(columns=['id', 'code', 'ast'])
    for _, row in old_source.iterrows():
        try:
            code = row['code']
            ast = code_to_ast(code)
            new_source = new_source.append(
                {'id': row['id'], 'code': code, 'ast': ast}, ignore_index=True)
        except:
            continue

    new_source.to_pickle(pickle_path)
    return new_source


def read_pairs(file_path):
    return pd.read_pickle(file_path)


def split_data(pairs):
    # 以 3:1 比例分割 train 和 test 的数据集
    pairs_cnt = len(pairs)
    train_cnt = pairs_cnt // 4 * 3

    data = pairs.sample(frac=1, random_state=666)
    train = data.iloc[:train_cnt]
    test = data.iloc[train_cnt:]

    check_path(settings.train_path)
    train_file_path = settings.train_path+'/java_pairs.pkl'
    train.to_pickle(train_file_path)

    check_path(settings.test_path)
    test_file_path = settings.test_path+'/java_pairs.pkl'
    test.to_pickle(test_file_path)


def main():
    parse_source(settings.java_source_path,
                 settings.java_ast_path)
    pairs = read_pairs(settings.java_pairs_path)
    split_data(pairs)


if __name__ == "__main__":
    main()
    # print(pd.read_pickle(settings.java_ast_path))
