import json

import pandas as pd

from config import (image_height, image_width, mapper_path, test_dataset,
                    test_df_path)
from lib.tools import get_result_dataframe
from lib.exceptions import InvalidDatasetStructure


def main():

    with open(mapper_path, 'r') as outfile:
        target_mapper = json.load(outfile)

    try:
        result_dataframe, _ = get_result_dataframe(
            target_dataset=test_dataset, 
            image_height=image_height, 
            image_width=image_width,
            target_mapper=target_mapper
            )
    except InvalidDatasetStructure:
        print('Структура тестового датасета не соответствует тренировочному')
        return
    
    pd.to_pickle(result_dataframe, test_df_path)


if __name__ == "__main__":
    main()
