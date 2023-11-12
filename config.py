# Для формирования датафреймов
raw_dataset = "./raw_data"
resized_dataset = "./data"

target_datasets = [
    "./data/squirrels_head",
    './data/squirrels_tail',
    "./data/the_center_of_the_gemstone",
    "./data/the_center_of_the_koalas_nose",
    "./data/the_center_of_the_owls_head",
    "./data/the_center_of_the_seahorses_head",
    "./data/the_center_of_the_teddy_bear_nose",
]

target_mapper = {
    "./data/squirrels_head": 0,
    './data/squirrels_tail': 1,
    "./data/the_center_of_the_gemstone": 2,
    "./data/the_center_of_the_koalas_nose": 3,
    "./data/the_center_of_the_owls_head": 4,
    "./data/the_center_of_the_seahorses_head": 5,
    "./data/the_center_of_the_teddy_bear_nose": 6
}

treashold = 0.1
image_height = 256
image_width = 256

# Для обучения и валидации
learning_rate = 0.001
num_epochs = 50
gamma = 0.95
rotate_limit = 10
early_stopping = 10
verbose = 20
batch_size = 16
num_workers = 4

full_data = False

test_weights = "./best_model.pth"

# test_df_path = ["./data/test_df.pkl"]
