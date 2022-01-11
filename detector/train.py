from db import Lite
from astExtractor import AstExtractor
import pandas as pd

def main():
    train_df = pd.read_pickle("./input/training/train.pkl")
    ast = AstExtractor()


if __name__ == "__main__":
    main()
