import pandas as pd

from config import (
    test_datasets, 
    image_height, 
    image_width, 
    test_target_mapper, 
    test_df_path)

from lib.tools import get_result_dataframe

def main():
    result_dataframe = get_result_dataframe(
        target_datasets=test_datasets, 
        image_height=image_height, 
        image_width=image_width, 
        target_mapper=test_target_mapper
        )
    pd.to_pickle(result_dataframe, test_df_path)


if __name__ == "__main__":
    main()
