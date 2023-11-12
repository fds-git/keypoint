# keypoint


### Обучение

1. Создать папку raw_data и исходные 7 датасетов положить в нее
2. Изменить размер исходных изображений на целевой (в соответствии с config.py)

        python3.11 resize_dataset.py

3. Сгенерировать датафрейм для обучения и валидации
   
        python3.11 generate_dataframe.py

4. Запустить обучение обучение с валидацией (неправильные целевые значения будут пропущены, full_data должно быть равно False)

        python3.11 train.py

5. Подобрать гиперпараметры и запустить обучение на полных данных (предварительно изменить full_data на значение True)
   
        python3.11 train.py

6. Сохранить модель с наилучшими характеристиками

        python3.11 save.py

### Тестирование

1. Создать папку test_data и исходные 7 датасетов положить в нее

2. Сгенерировать датафрейм для тестирования
   
        python3.11 generate_test_dataframe.py

3. Вычислить необходимые метрики

        python3.11 test.py
