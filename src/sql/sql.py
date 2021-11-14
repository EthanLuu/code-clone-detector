import sqlite3


class DB:
    def __init__(self, db_path) -> None:
        # 在连接数据库前请保证数据库文件已被正确解压缩
        self.con = sqlite3.connect(db_path)
        # 创建游标对象
        self.cur = self.con.cursor()

    def get_all_samples(self):
        '''获取数据库中的所有代码样本'''
        self.cur.execute("select * from submissions")
        data = self.cur.fetchall()
        return data

    def get_one_random_sample(self):
        '''随机获取一个代码样本'''
        self.cur.execute("select * from submissions order by random() limit 1")
        data = self.cur.fetchone()
        return data

    def get_code(self, data):
        '''获取该样本的代码，默认索引是倒数第二个'''
        return data[-2]

    def get_code_file(self, data):
        with open("./assets/tmp.py", "w") as f:
            f.write(self.get_code(data))
            return f


def main():
    db = DB("./assets/python-code-samples.db")
    data = db.get_one_random_sample()
    print(db.get_code(data))
    print(db.get_code_file(data))


if __name__ == "__main__":
    main()
