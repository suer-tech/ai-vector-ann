import numpy as np
import json
import os
import requests
from colorama import init, Fore, Style
from annoy import AnnoyIndex

from instructions import full_instructions
from jina import interact_stream

init()


# Функция поиска по вопросу
def find_instruction(question):
    # Путь к файлам для сохранения
    index_file = 'vector_index.ann'
    instructions_file = 'faiss_files/instructions.json'
    embeddings_file = 'faiss_files/embeddings.npy'

    # Проверка на существование файлов
    if os.path.exists(index_file) and os.path.exists(instructions_file) and os.path.exists(embeddings_file):
        # Загрузка инструкций
        with open(instructions_file, 'r', encoding='utf-8') as f:
            instructions_dict = json.load(f)

        # Загрузка эмбеддингов
        embeddings_array = np.load(embeddings_file).astype('float32')
        print(f'Вопрос: {question}\n')

        # Проверка размерности эмбеддингов инструкций
        print("Размерность эмбеддингов инструкций:", embeddings_array.shape)

        # Генерация эмбеддинга для вопроса
        question_embedding = generate_embedding(question)
        question_embedding = np.array(question_embedding).astype('float32').reshape(1, -1)

        # Проверка размерности эмбеддинга вопроса
        print("Размерность эмбеддинга вопроса:", question_embedding.shape)

        # Убедитесь, что размеры совпадают
        if embeddings_array.shape[1] != question_embedding.shape[1]:
            raise ValueError("Размерности эмбеддингов инструкций и вопроса не совпадают!")

        # Создание индекса Annoy (предполагаем, что размерность 512)
        index = AnnoyIndex(embeddings_array.shape[1], 'angular')  # 'angular' для косинусного сходства

        # Загрузка индекса из файла
        index.load(index_file)

        # Поиск ближайших соседей
        k = 1  # количество ближайших соседей
        indices = index.get_nns_by_vector(question_embedding.flatten(), k)

        print("Индексы похожих инструкций:", indices)

        instructions_array = []

        # Извлечение текстов инструкций по найденным индексам
        for idx in indices:
            str_idx = str(idx)
            if str_idx in instructions_dict:
                instruct = instructions_dict[str_idx]
                instructions_array.append(instruct['content'])
                print(f"Инструкция: {instruct['title']}\nСодержание: {instruct['content']}\n")

        return instructions_array

    else:
        print("Не найдены необходимые файлы.")


def interact_stream_voiceflow(text):
    data_launch = {
        "action": {
            "type": "launch"
        }
    }
    data_text = {
        "action": {
            "type": "text",
            'payload': f'{text}'
        }
    }
    user_id = "your_user_id_here"
    voiceflow_api_key = "VF.DM.67499102b357d4928b4ac58b.ZQrIJbkjPUwhCr2I"
    project_id = "672caaaf565396bb1012956f"
    url = f"https://general-runtime.voiceflow.com/v2/project/{project_id}/user/{user_id}/interact/stream"

    headers = {
        'Accept': 'text/event-stream',
        'Authorization': voiceflow_api_key,
        'Content-Type': 'application/json'
    }

    # Запуск взаимодействия
    requests.post(url, headers=headers, json=data_launch)
    response = requests.post(url, headers=headers, json=data_text)

    if response.status_code == 200:
        for line in response.iter_lines():
            if line:
                decoded_line = line.decode('utf-8')
                if decoded_line.startswith("data:"):
                    try:
                        json_data = decoded_line[5:]
                        parsed_data = json.loads(json_data)
                        if 'paths' in parsed_data and len(parsed_data['paths']) > 0:
                            for path in parsed_data['paths']:
                                output = path.get('event', {}).get('payload', {}).get('output')
                                if output:
                                    print(f"{Fore.GREEN}Сгенерированный ответ: {output}{Style.RESET_ALL}")
                                    return output
                    except json.JSONDecodeError:
                        print(f"Не удалось разобрать: {decoded_line}")


def generate_embedding(text):
    # Здесь должен быть код для отправки текста на API модели генерации эмбеддингов
    response = interact_stream(text)
    return response


def generate_data_for_final_response(user_question):
    instructions_data = find_instruction(user_question)
    text = f"{Fore.RED}Вопрос: {user_question}{Style.RESET_ALL}\n{Fore.BLUE}Найденные инструкции: {instructions_data}{Style.RESET_ALL}"
    print(text)
    print('')
    data_for_final_response = interact_stream_voiceflow(text)
    return data_for_final_response



import os
import json
import numpy as np
from annoy import AnnoyIndex

# Путь к файлам для сохранения
index_file = 'vector_index.ann'
instructions_file = 'instructions.json'
embeddings_file = 'embeddings.npy'

# Проверка на существование файлов
if os.path.exists(index_file) and os.path.exists(instructions_file) and os.path.exists(embeddings_file):
    # Загрузка инструкций
    with open(instructions_file, 'r', encoding='utf-8') as f:
        instructions_dict = json.load(f)

    # Загрузка эмбеддингов
    embeddings_array = np.load(embeddings_file).astype('float32')
else:
    instructions = full_instructions  # Предполагается, что full_instructions определены ранее

    # Генерация эмбеддингов для инструкций
    embeddings = []
    for instruction in instructions:
        full_text = f"{instruction['title']} {instruction['content']}"
        embedding = generate_embedding(full_text)  # Вызов функции для получения эмбеддинга через API
        if embedding is not None:
            embeddings.append(embedding)

    # Преобразование списка эмбеддингов в numpy массив
    embeddings_array = np.array(embeddings).astype('float32')

    # Создание индекса Annoy (предполагаем, что размерность 512)
    index = AnnoyIndex(embeddings_array.shape[1], 'angular')  # 'angular' для косинусного расстояния

    # Добавление эмбеддингов в индекс
    for i in range(len(embeddings_array)):
        index.add_item(i, embeddings_array[i])

    # Построение индекса перед его сохранением
    index.build(10)  # Настройте количество деревьев по необходимости

    # Сохранение индекса на диск только после добавления всех элементов
    index.save(index_file)

    # Сохранение инструкций для дальнейшего использования
    instructions_dict = {i: instruction for i, instruction in enumerate(instructions)}

    # Сохранение инструкций в файл (например, JSON)
    with open(instructions_file, 'w', encoding='utf-8') as f:
        json.dump(instructions_dict, f, ensure_ascii=False, indent=4)

    # Сохранение эмбеддингов в файл (например, NumPy)
    np.save(embeddings_file, embeddings_array)

print("Эмбеддинги и инструкции успешно загружены или созданы.")
