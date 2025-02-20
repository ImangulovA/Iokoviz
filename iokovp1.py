import pandas as pd
import numpy as np


def transform_tournament_data(input_file, output_file):
    """
    Преобразует данные турнира из формата с отдельными турами
    в формат с одной строкой на команду.

    Parameters:
    input_file (str): Путь к входному Excel-файлу
    output_file (str): Путь к выходному CSV-файлу
    """
    # Читаем Excel-файл
    df = pd.read_excel(input_file)

    # Создаем словарь для хранения данных команд
    teams = {}

    # Проходим по каждой строке данных
    for _, row in df.iterrows():
        team_id = row[' Номер команды']
        team_name = row['Название']
        city = row['Город']
        tour = row['Тур']

        # Если команда встречается впервые, инициализируем её данные
        if team_id not in teams:
            teams[team_id] = {
                'id': team_id,
                'name': team_name,
                'city': city,
                'answers': [None] * 108  # 9 туров * 12 вопросов
            }

        # Заполняем ответы для текущего тура
        start_idx = (tour - 1) * 12
        answers = row.iloc[4:16].values  # Берем 12 ответов, начиная с 5-го столбца

        for i, answer in enumerate(answers):
            teams[team_id]['answers'][start_idx + i] = answer

    # Создаем список для новых данных
    new_data = []

    # Преобразуем данные в список строк
    for team in sorted(teams.values(), key=lambda x: x['id']):
        row = [team['id'], team['name'], team['city']] + team['answers']
        new_data.append(row)

    # Создаем заголовки
    headers = ['Номер команды', 'Название', 'Город'] + [f'Q{i + 1}' for i in range(108)]

    # Создаем новый DataFrame
    new_df = pd.DataFrame(new_data, columns=headers)

    # Сохраняем в CSV
    new_df.to_csv(output_file, index=False, encoding='utf-8')

    print(f"Преобразование завершено. Результат сохранен в {output_file}")
    print(f"Обработано команд: {len(teams)}")


# Пример использования
if __name__ == "__main__":
    input_file = "C:/Users/Amal Imangulov/Downloads/tournament-tours-11497-20-Feb-2025.xlsx"
    output_file = "tournament_transformed.csv"
    transform_tournament_data(input_file, output_file)