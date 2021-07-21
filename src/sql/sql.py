import sqlite3

# 链接数据库，若数据库不存在则创建
con = sqlite3.connect("../../database/java-python-clones.db")

# 创建游标对象
cur = con.cursor()

cur.execute("select * from submissions_python")
data = cur.fetchall()

columns = {
    "source": 12
}