# Местонахождение датасетов
raw_train_dataset = "./raw_data"
train_dataset = "./train_data"
test_dataset = "./test_data"

# Размер изображения, с которым будет работать сеть
image_height = 256
image_width = 256

# Названия классов в индексы классов
mapper_path = './generated/mapper.json'

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

# Датафреймы для обучения и тестирования
train_df_path = './generated/train_dataframe.pkl'
test_df_path = './generated/test_dataframe.pkl'

# Веса для тестирования
test_weights = './experiments/exp_27_20-11-2023_18:21:43/weights/ep_3_valid_loss_466.1808'