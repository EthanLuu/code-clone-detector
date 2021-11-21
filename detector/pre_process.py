# 将我们需要的关键信息从 sqlite 中提取出来并且存储到 txt 中

import os
import pandas as pd
import pickle
from db import Lite


# db = Lite("./assets/python-code-samples.db")

# samples = db.get_all_samples()

# for sample in samples:
#     code = db.get_code(sample)

train_df = pd.read_pickle("../input/training")
train_df.read()