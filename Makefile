#!make
include .env

build:
	docker build --build-arg USER_ID=$(USER_ID) --build-arg GROUP_ID=$(GROUP_ID) -f Dockerfile -t test_task .

resize_train:
	docker run --name test_task --rm -v ./:/app/ test_task /bin/bash -c "python3.11 resize_dataset.py"

generate_train_df:
	docker run --name test_task --rm -v ./:/app/ test_task /bin/bash -c "python3.11 generate_train_df.py"

train:
	docker run --name test_task --rm --ipc=host --gpus all -v ./:/app/ test_task /bin/bash -c "python3.11 train.py"

generate_test_df:
	docker run --name test_task --rm -v ./:/app/ test_task /bin/bash -c "python3.11 generate_test_df.py"

test:
	docker run --name test_task --rm  --ipc=host --gpus all -v ./:/app/ test_task /bin/bash -c "python3.11 test.py"

exec:
	docker run --name test_task --rm --ipc=host --gpus all -it -v ./:/app/ test_task /bin/bash -c "bash"	