import json

import pandas as pd

from config import (image_height, image_width, mapper_path, train_dataset,
                    train_df_path)
from lib.tools import get_result_dataframe


def main():
    result_dataframe, mapper = get_result_dataframe(
        target_dataset=train_dataset, 
        image_height=image_height, 
        image_width=image_width,
        target_mapper=None
        )
    pd.to_pickle(result_dataframe, train_df_path)
    print(f"Generated mapper: {mapper}")
    
    with open(mapper_path, 'w') as outfile:
        json.dump(mapper, outfile)


if __name__ == "__main__":
    main()
