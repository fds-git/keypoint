import pandas as pd

from config import (
    train_datasets, 
    image_height, 
    image_width, 
    train_target_mapper, 
    train_df_path
    )
from lib.tools import get_result_dataframe

def main():
    result_dataframe = get_result_dataframe(
        target_datasets=train_datasets, 
        image_height=image_height, 
        image_width=image_width, 
        target_mapper=train_target_mapper
        )
    pd.to_pickle(result_dataframe, train_df_path)


if __name__ == "__main__":
    main()
