# Для масштабирования тренировочных данных под конкретный размер
raw_dataset = "./raw_data"
resized_dataset = "./data"
image_height = 256
image_width = 256

# Данные для обучения
train_datasets = [
    "./data/squirrels_head",
    './data/squirrels_tail',
    "./data/the_center_of_the_gemstone",
    "./data/the_center_of_the_koalas_nose",
    "./data/the_center_of_the_owls_head",
    "./data/the_center_of_the_seahorses_head",
    "./data/the_center_of_the_teddy_bear_nose",
]

# Отображение названия датасета в индекс датасета
train_target_mapper = {train_datasets[i]: i for i in range(len(train_datasets))}

# Данные для тестирования
test_datasets = [
    "./test_data/squirrels_head",
    './test_data/squirrels_tail',
    "./test_data/the_center_of_the_gemstone",
    "./test_data/the_center_of_the_koalas_nose",
    "./test_data/the_center_of_the_owls_head",
    "./test_data/the_center_of_the_seahorses_head",
    "./test_data/the_center_of_the_teddy_bear_nose",
]

# Отображение названия датасета в индекс датасета
test_target_mapper = {test_datasets[i]: i for i in range(len(test_datasets))}

# Для проведения экспериментов
learning_rate = 0.002
num_epochs = 50
gamma = 0.9
rotate_limit = 20
early_stopping = 10
verbose = 20
batch_size = 16
num_workers = 4
treashold = 0.1
full_data = False

train_df_path = './train_dataframe.pkl'
test_df_path = './test_dataframe.pkl'

test_weights = './weights/ep_16_train_loss_334.7822'