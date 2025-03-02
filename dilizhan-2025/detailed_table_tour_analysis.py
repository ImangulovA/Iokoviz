
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict


def load_and_clean_data(file_path):
    """
    Загружает и обрабатывает CSV-файл с результатами турнира.

    Parameters:
    file_path (str): Путь к CSV-файлу с результатами

    Returns:
    tuple: (df_teams, df_stats) - фреймы с данными команд и статистикой
    """
    # Загрузка данных
    df = pd.read_csv(file_path, sep=',', encoding='utf-8')

    # Отделяем данные команд от статистики
    df_teams = df[df['Команда'].notna() & (df['Команда'] != 'Взявших:')]
    df_stats = df[df['Рейтинг'] == 'Взявших:']

    return df_teams, df_stats

def analyze_tour_difficulty(df_stats):
    """
    Анализирует сложность туров в целом и по частям (начало, середина, конец).

    Parameters:
    df_stats (DataFrame): Фрейм данных со статистикой вопросов

    Returns:
    dict: Результаты анализа сложности туров
    """
    # Определение вопросов по турам
    tour_questions = {
        1: list(range(1, 16)),
        2: list(range(16, 31)),
        3: list(range(31, 46)),
        4: list(range(46, 61)),
        5: list(range(61, 76)),
        6: list(range(76, 91))
    }

    # Результаты анализа
    results = {
        'tour_difficulty': {},  # Средняя сложность туров
        'tour_parts': {}  # Сложность частей туров (начало, середина, конец)
    }

    # Анализ сложности туров и их частей
    for tour, questions in tour_questions.items():
        # Получаем число взявших для каждого вопроса тура
        taken_counts = [df_stats[str(q)].values[0] for q in questions]

        # Средняя сложность тура
        avg_taken = np.mean(taken_counts)
        results['tour_difficulty'][tour] = avg_taken

        # Разделяем тур на части
        q_start = questions[:5]  # Первые 5 вопросов
        q_middle = questions[5:10]  # Средние 5 вопросов
        q_end = questions[10:]  # Последние 5 вопросов

        # Считаем среднюю сложность для каждой части
        results['tour_parts'][tour] = {
            'start': np.mean([df_stats[str(q)].values[0] for q in q_start]),
            'middle': np.mean([df_stats[str(q)].values[0] for q in q_middle]),
            'end': np.mean([df_stats[str(q)].values[0] for q in q_end])
        }

    return results


def find_extreme_questions(df_stats):
    """
    Находит самые легкие и самые сложные вопросы по всему турниру и по турам.

    Parameters:
    df_stats (DataFrame): Фрейм данных со статистикой вопросов

    Returns:
    dict: Информация о самых легких и сложных вопросах
    """
    # Результаты анализа
    results = {
        'easiest_overall': [],  # Самые легкие вопросы в целом
        'hardest_overall': [],  # Самые сложные вопросы в целом
        'by_tour': {}  # Легкие и сложные вопросы по турам
    }

    # Преобразуем столбцы с номерами вопросов в числовой формат
    question_cols = [str(i) for i in range(1, 91)]
    questions_data = []

    for q in question_cols:
        if q in df_stats.columns:
            questions_data.append({
                'question': int(q),
                'tour': (int(q) - 1) // 15 + 1,
                'taken_count': df_stats[q].values[0]
            })

    # Сортируем вопросы по количеству взявших
    sorted_questions = sorted(questions_data, key=lambda x: x['taken_count'])

    # Самые легкие вопросы (топ-10)
    results['easiest_overall'] = sorted_questions[-10:][::-1]

    # Самые сложные вопросы (топ-10)
    results['hardest_overall'] = sorted_questions[:10]

    # Анализ по турам
    for tour in range(1, 7):
        tour_questions = [q for q in questions_data if q['tour'] == tour]
        sorted_tour_questions = sorted(tour_questions, key=lambda x: x['taken_count'])

        results['by_tour'][tour] = {
            'easiest': sorted_tour_questions[-3:][::-1],  # Топ-3 легких
            'hardest': sorted_tour_questions[:3]  # Топ-3 сложных
        }

    return results


def analyze_top_teams_questions(df_teams, df_stats):
    """
    Анализирует, какие вопросы повлияли на топ-5 команд.

    Parameters:
    df_teams (DataFrame): Фрейм данных с результатами команд
    df_stats (DataFrame): Фрейм данных со статистикой вопросов

    Returns:
    dict: Информация о влиянии вопросов на топ-5 команд
    """
    # Определяем топ-5 команд
    top5_teams = df_teams.sort_values('Сумма', ascending=False).head(5)

    # Результаты анализа
    results = {
        'top5_teams': top5_teams[['Команда', 'Сумма']].to_dict('records'),
        'common_taken': [],  # Вопросы, которые взяли все топ-5 команд
        'common_missed': [],  # Вопросы, которые не взяла ни одна из топ-5 команд
        'dividing_questions': []  # Вопросы, которые разделили топ-5 команд
    }

    # Проходим по всем вопросам
    for q in range(1, 91):
        q_str = str(q)

        # Считаем, сколько команд из топ-5 взяли вопрос
        taken_count = 0
        for _, team in top5_teams.iterrows():
            if pd.notna(team[q_str]):
                taken_count += 1

        # Определяем категорию вопроса
        if taken_count == 5:  # Взяли все
            results['common_taken'].append({
                'question': q,
                'tour': (q - 1) // 15 + 1
            })
        elif taken_count == 0:  # Не взял никто
            results['common_missed'].append({
                'question': q,
                'tour': (q - 1) // 15 + 1
            })
        else:  # Вопросы, разделившие команды (взяли 1-4 из топ-5)
            results['dividing_questions'].append({
                'question': q,
                'tour': (q - 1) // 15 + 1,
                'taken_count': taken_count,
                'total_taken': df_stats[q_str].values[0] if q_str in df_stats.columns else 0
            })

    # Сортируем вопросы, разделившие команды, по важности
    results['dividing_questions'] = sorted(
        results['dividing_questions'],
        key=lambda x: (x['taken_count'], -x['total_taken']),
        reverse=True
    )

    return results

def analyze_team_streaks(df_teams):
    """
    Анализирует серии взятых и невзятых вопросов для команд.

    Parameters:
    df_teams (DataFrame): Фрейм данных с результатами команд

    Returns:
    dict: Информация о сериях для команд
    """
    results = {
        'team_streaks': {},  # Серии для каждой команды
        'max_taken_streak': {'team': '', 'tour': 0, 'length': 0, 'start': 0, 'end': 0},
        'max_missed_streak': {'team': '', 'tour': 0, 'length': 0, 'start': 0, 'end': 0}
    }

    # Для каждой команды анализируем серии
    for _, team in df_teams.iterrows():
        team_name = team['Команда']
        results['team_streaks'][team_name] = {}

        # Анализируем по турам
        for tour in range(1, 7):
            start_q = (tour - 1) * 15 + 1
            end_q = tour * 15

            # Получаем результаты по вопросам тура
            tour_questions = []
            for q in range(start_q, end_q + 1):
                taken = pd.notna(team[str(q)])
                tour_questions.append(taken)

            # Ищем серии взятых вопросов
            max_taken_streak = 0
            max_taken_start = -1
            current_taken_streak = 0

            # Ищем серии невзятых вопросов
            max_missed_streak = 0
            max_missed_start = -1
            current_missed_streak = 0

            for i, taken in enumerate(tour_questions):
                if taken:  # Вопрос взят
                    current_taken_streak += 1
                    current_missed_streak = 0

                    if current_taken_streak > max_taken_streak:
                        max_taken_streak = current_taken_streak
                        max_taken_start = i - current_taken_streak + 1
                else:  # Вопрос не взят
                    current_missed_streak += 1
                    current_taken_streak = 0

                    if current_missed_streak > max_missed_streak:
                        max_missed_streak = current_missed_streak
                        max_missed_start = i - current_missed_streak + 1

            # Сохраняем результаты для тура
            results['team_streaks'][team_name][tour] = {
                'taken_streak': {
                    'length': max_taken_streak,
                    'start': start_q + max_taken_start if max_taken_start >= 0 else -1,
                    'end': start_q + max_taken_start + max_taken_streak - 1 if max_taken_start >= 0 else -1
                },
                'missed_streak': {
                    'length': max_missed_streak,
                    'start': start_q + max_missed_start if max_missed_start >= 0 else -1,
                    'end': start_q + max_missed_start + max_missed_streak - 1 if max_missed_start >= 0 else -1
                },
                'performance': tour_questions
            }

            # Обновляем глобальные максимумы
            if max_taken_streak > results['max_taken_streak']['length']:
                results['max_taken_streak'] = {
                    'team': team_name,
                    'tour': tour,
                    'length': max_taken_streak,
                    'start': start_q + max_taken_start,
                    'end': start_q + max_taken_start + max_taken_streak - 1
                }

            if max_missed_streak > results['max_missed_streak']['length']:
                results['max_missed_streak'] = {
                    'team': team_name,
                    'tour': tour,
                    'length': max_missed_streak,
                    'start': start_q + max_missed_start,
                    'end': start_q + max_missed_start + max_missed_streak - 1
                }

    return results


def analyze_question_flow(df_teams, df_stats):
    """
    Анализирует "поток" вопросов - насколько часто команды брали вопросы последовательно.

    Parameters:
    df_teams (DataFrame): Фрейм данных с результатами команд
    df_stats (DataFrame): Фрейм данных со статистикой вопросов

    Returns:
    dict: Информация о последовательности взятия вопросов
    """
    results = {
        'tour_flow': {},  # Общий "поток" тура
        'question_pairs': {}  # Пары вопросов и их "потоковость"
    }

    # Анализируем по турам
    for tour in range(1, 7):
        start_q = (tour - 1) * 15 + 1
        end_q = tour * 15

        # Счетчики для вопросов
        taken_counts = [0] * 15  # Сколько команд взяли каждый вопрос
        pairs_counts = [0] * 14  # Сколько команд взяли пары последовательных вопросов

        # Считаем, сколько команд взяли каждый вопрос и пары вопросов
        for _, team in df_teams.iterrows():
            last_taken = False

            for i in range(15):
                q_num = start_q + i
                taken = pd.notna(team[str(q_num)])

                if taken:
                    taken_counts[i] += 1

                    # Проверяем последовательность
                    if i > 0 and last_taken:
                        pairs_counts[i - 1] += 1

                last_taken = taken

        # Рассчитываем "поток" тура
        total_taken = sum(taken_counts) - taken_counts[0]  # Исключаем первый вопрос
        total_pairs = sum(pairs_counts)
        flow_score = total_pairs / total_taken if total_taken > 0 else 0

        results['tour_flow'][tour] = flow_score

        # Анализируем конкретные пары вопросов
        flow_pairs = []
        for i in range(14):
            q1 = start_q + i
            q2 = start_q + i + 1
            q1_taken = taken_counts[i]
            pair_taken = pairs_counts[i]

            if q1_taken > 0:
                flow_pairs.append({
                    'questions': f"{q1}-{q2}",
                    'flow_rate': pair_taken / q1_taken,
                    'taken_count': pair_taken,
                    'first_question_taken': q1_taken
                })

        # Сортируем пары по "потоковости"
        results['question_pairs'][tour] = {
            'best': sorted(flow_pairs, key=lambda x: (-x['flow_rate'], -x['taken_count']))[:3],
            'worst': sorted(flow_pairs, key=lambda x: (x['flow_rate'], -x['taken_count']))[:3]
        }

    return results


def visualize_results(results, output_dir='.'):
    """
    Создает визуализации на основе результатов анализа.

    Parameters:
    results (dict): Результаты анализа
    output_dir (str): Директория для сохранения графиков
    """
    # Настройка внешнего вида графиков
    plt.style.use('ggplot')

    # 1. Сложность туров
    plt.figure(figsize=(10, 6))
    tours = list(range(1, 7))
    difficulties = [results['tour_difficulty']['tour_difficulty'][t] for t in tours]

    plt.bar(tours, difficulties, color='skyblue')
    plt.title('Сложность туров (среднее количество взявших команд)', fontsize=14)
    plt.xlabel('Номер тура', fontsize=12)
    plt.ylabel('Среднее количество взявших команд', fontsize=12)
    plt.xticks(tours)

    # Добавляем значения над столбцами
    for i, v in enumerate(difficulties):
        plt.text(i + 1, v + 0.5, f'{v:.2f}', ha='center')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/tour_difficulty.png')
    plt.close()

    # 2. Сложность частей туров
    plt.figure(figsize=(12, 8))

    # Данные для графика
    parts = ['start', 'middle', 'end']
    x = np.arange(len(tours))
    width = 0.25

    # Создаем группированную диаграмму
    for i, part in enumerate(parts):
        values = [results['tour_difficulty']['tour_parts'][t][part] for t in tours]
        plt.bar(x + i * width, values, width, label=f'{part.capitalize()}')

    plt.title('Сложность частей туров', fontsize=14)
    plt.xlabel('Номер тура', fontsize=12)
    plt.ylabel('Среднее количество взявших команд', fontsize=12)
    plt.xticks(x + width, tours)
    plt.legend()

    plt.tight_layout()
    plt.savefig(f'{output_dir}/tour_parts_difficulty.png')
    plt.close()

    # 3. Самые легкие и сложные вопросы
    plt.figure(figsize=(12, 8))

    # Топ-10 самых легких вопросов
    easiest = results['extreme_questions']['easiest_overall'][:10]
    q_numbers_easy = [q['question'] for q in easiest]
    taken_counts_easy = [q['taken_count'] for q in easiest]

    plt.subplot(2, 1, 1)
    plt.bar(q_numbers_easy, taken_counts_easy, color='green')
    plt.title('Топ-10 самых легких вопросов', fontsize=14)
    plt.xlabel('Номер вопроса', fontsize=12)
    plt.ylabel('Количество взявших команд', fontsize=12)

    # Топ-10 самых сложных вопросов
    hardest = results['extreme_questions']['hardest_overall'][:10]
    q_numbers_hard = [q['question'] for q in hardest]
    taken_counts_hard = [q['taken_count'] for q in hardest]

    plt.subplot(2, 1, 2)
    plt.bar(q_numbers_hard, taken_counts_hard, color='red')
    plt.title('Топ-10 самых сложных вопросов', fontsize=14)
    plt.xlabel('Номер вопроса', fontsize=12)
    plt.ylabel('Количество взявших команд', fontsize=12)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/extreme_questions.png')
    plt.close()

def run_analysis(file_path, output_dir='.'):
    """
    Запускает полный анализ данных турнира и сохраняет результаты.

    Parameters:
    file_path (str): Путь к CSV-файлу с результатами
    output_dir (str): Директория для сохранения результатов

    Returns:
    dict: Результаты анализа
    """
    import os
    from datetime import datetime

    # Создаем директорию для результатов, если она не существует
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Создаем поддиректорию с датой и временем для текущего анализа
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = os.path.join(output_dir, f"analysis_{timestamp}")
    os.makedirs(result_dir)

    # Директория для графиков
    plots_dir = os.path.join(result_dir, "plots")
    os.makedirs(plots_dir)

    print(f"Начинаем анализ файла: {file_path}")
    print(f"Результаты будут сохранены в: {result_dir}")

    # Загружаем данные
    df_teams, df_stats = load_and_clean_data(file_path)
    print(f"Загружено {len(df_teams)} команд")

    # Проводим анализ по разным направлениям
    results = {}

    # 1. Сложность туров
    print("Анализ сложности туров...")
    results['tour_difficulty'] = analyze_tour_difficulty(df_stats)

    # 2. Самые легкие и сложные вопросы
    print("Анализ самых легких и сложных вопросов...")
    results['extreme_questions'] = find_extreme_questions(df_stats)

    # 3. Влияние вопросов на топ-5 команд
    print("Анализ влияния вопросов на топ-5 команд...")
    results['top_teams'] = analyze_top_teams_questions(df_teams, df_stats)

    # 4. Серии взятых/невзятых вопросов
    print("Анализ серий взятых/невзятых вопросов...")
    results['team_streaks'] = analyze_team_streaks(df_teams)

    # 5. "Поток" вопросов
    print("Анализ последовательности взятия вопросов...")
    results['question_flow'] = analyze_question_flow(df_teams, df_stats)

    # Сохраняем текстовый отчет
    report_file = os.path.join(result_dir, "analysis_report.txt")
    with open(report_file, 'w', encoding='utf-8') as f:
        # Перенаправляем вывод в файл
        import sys
        original_stdout = sys.stdout
        sys.stdout = f

        print(f"АНАЛИЗ РЕЗУЛЬТАТОВ ТУРНИРА")
        print(f"Дата и время анализа: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Файл данных: {os.path.basename(file_path)}")
        print("\n")


        # Возвращаем stdout обратно
        sys.stdout = original_stdout

    print(f"Текстовый отчет сохранен в {report_file}")

    # Создаем и сохраняем визуализации
    print("Создание визуализаций...")
    visualize_results(results, plots_dir)

    # Создаем дополнительные визуализации

    # 1. Тепловая карта взятия вопросов топ-командами
    print("Создание тепловой карты взятия вопросов...")
    create_top_teams_heatmap(df_teams, plots_dir)

    # 2. Визуализация прогресса команд по турам
    print("Создание графика прогресса команд...")
    visualize_team_progress(df_teams, plots_dir)

    # 3. Корреляция между результатами туров
    print("Создание матрицы корреляций между турами...")
    visualize_tour_correlations(df_teams, plots_dir)

    print(f"Все визуализации сохранены в {plots_dir}")

    # Сохраняем результаты в Excel-файл для дальнейшего анализа
    try:
        create_excel_report(results, df_teams, df_stats, os.path.join(result_dir, "detailed_analysis.xlsx"))
        print(f"Файл Excel с детальным анализом сохранен")
    except Exception as e:
        print(f"Не удалось создать Excel-отчет: {e}")

    print("\nАнализ завершен!")
    return results


def create_top_teams_heatmap(df_teams, output_dir):
    """
    Создает тепловую карту взятия вопросов топ-командами.

    Parameters:
    df_teams (DataFrame): Фрейм данных с результатами команд
    output_dir (str): Директория для сохранения графика
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    import pandas as pd

    # Выбираем топ-10 команд
    top_teams = df_teams.sort_values('Сумма', ascending=False).head(10)

    # Создаем матрицу взятия вопросов для каждого тура
    for tour in range(1, 7):
        start_q = (tour - 1) * 15 + 1
        end_q = tour * 15

        # Создаем матрицу взятия вопросов
        question_matrix = np.zeros((len(top_teams), 15))

        for i, (_, team) in enumerate(top_teams.iterrows()):
            for j, q in enumerate(range(start_q, end_q + 1)):
                question_matrix[i, j] = 1 if pd.notna(team[str(q)]) else 0

        # Создаем тепловую карту
        plt.figure(figsize=(15, 8))
        ax = sns.heatmap(
            question_matrix,
            cmap=['lightgray', 'green'],
            cbar=False,
            linewidths=0.5,
            linecolor='white',
            xticklabels=[str(q) for q in range(start_q, end_q + 1)],
            yticklabels=top_teams['Команда'].tolist()
        )

        # Добавляем заголовок и подписи осей
        plt.title(f'Взятие вопросов в туре {tour} топ-10 командами', fontsize=16)
        plt.xlabel('Номер вопроса', fontsize=12)
        plt.ylabel('Команда', fontsize=12)

        # Поворачиваем метки на оси Y для лучшей читаемости
        plt.yticks(rotation=0)

        # Добавляем аннотации (процент взятия)
        for i in range(len(top_teams)):
            for j in range(15):
                plt.text(
                    j + 0.5,
                    i + 0.5,
                    '✓' if question_matrix[i, j] else '✗',
                    ha='center',
                    va='center',
                    color='black' if question_matrix[i, j] else 'red'
                )

        plt.tight_layout()
        plt.savefig(f'{output_dir}/top_teams_heatmap_tour_{tour}.png', dpi=300, bbox_inches='tight')
        plt.close()


def visualize_team_progress(df_teams, output_dir):
    """
    Визуализирует прогресс команд по турам.

    Parameters:
    df_teams (DataFrame): Фрейм данных с результатами команд
    output_dir (str): Директория для сохранения графика
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np

    # Выбираем топ-10 команд
    top_teams = df_teams.sort_values('Сумма', ascending=False).head(10)

    # Создаем фрейм данных с прогрессом команд
    tour_columns = ['Тур 1', 'Тур 2', 'Тур 3', 'Тур 4', 'Тур 5', 'Тур 6']

    # Собираем данные о взятых вопросах в каждом туре
    progress_data = []
    for _, team in top_teams.iterrows():
        team_data = {'Команда': team['Команда']}

        for tour in range(1, 7):
            start_q = (tour - 1) * 15 + 1
            end_q = tour * 15

            # Считаем количество взятых вопросов в туре
            taken_count = 0
            for q in range(start_q, end_q + 1):
                if pd.notna(team[str(q)]):
                    taken_count += 1

            team_data[f'Тур {tour}'] = taken_count

        progress_data.append(team_data)

    progress_df = pd.DataFrame(progress_data)

    # Создаем линейный график
    plt.figure(figsize=(12, 8))

    colors = plt.cm.tab10(np.linspace(0, 1, len(progress_df)))

    for i, (_, row) in enumerate(progress_df.iterrows()):
        plt.plot(
            range(1, 7),
            [row[f'Тур {t}'] for t in range(1, 7)],
            marker='o',
            linewidth=2,
            color=colors[i],
            label=row['Команда']
        )

    plt.title('Прогресс топ-10 команд по турам', fontsize=16)
    plt.xlabel('Номер тура', fontsize=12)
    plt.ylabel('Количество взятых вопросов', fontsize=12)
    plt.xticks(range(1, 7))
    plt.yticks(range(0, 16))
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/team_progress.png', dpi=300, bbox_inches='tight')
    plt.close()


def visualize_tour_correlations(df_teams, output_dir):
    """
    Создает матрицу корреляций между результатами в разных турах.

    Parameters:
    df_teams (DataFrame): Фрейм данных с результатами команд
    output_dir (str): Директория для сохранения графика
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    import numpy as np

    # Собираем данные о количестве взятых вопросов в каждом туре
    tour_results = []

    for _, team in df_teams.iterrows():
        tour_data = {'Команда': team['Команда']}

        for tour in range(1, 7):
            start_q = (tour - 1) * 15 + 1
            end_q = tour * 15

            # Считаем количество взятых вопросов в туре
            taken_count = 0
            for q in range(start_q, end_q + 1):
                if pd.notna(team[str(q)]):
                    taken_count += 1

            tour_data[f'Тур {tour}'] = taken_count

        tour_results.append(tour_data)

    # Создаем DataFrame с результатами по турам
    tour_df = pd.DataFrame(tour_results)

    # Вычисляем корреляции между турами
    tour_cols = [f'Тур {t}' for t in range(1, 7)]
    corr_matrix = tour_df[tour_cols].corr()

    # Создаем тепловую карту корреляций
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        corr_matrix,
        annot=True,
        cmap='coolwarm',
        vmin=-1,
        vmax=1,
        linewidths=0.5,
        fmt='.2f'
    )

    plt.title('Корреляция между результатами в разных турах', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/tour_correlations.png', dpi=300, bbox_inches='tight')
    plt.close()


def create_excel_report(results, df_teams, df_stats, output_file):
    """
    Создает детальный отчет в формате Excel.

    Parameters:
    results (dict): Результаты анализа
    df_teams (DataFrame): Фрейм данных с результатами команд
    df_stats (DataFrame): Фрейм данных со статистикой вопросов
    output_file (str): Путь к выходному файлу Excel
    """
    import pandas as pd

    # Создаем объект Excel Writer
    with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
        # 1. Общая информация о командах
        df_teams.to_excel(writer, sheet_name='Команды', index=False)

        # 2. Статистика вопросов
        if not df_stats.empty:
            df_stats.to_excel(writer, sheet_name='Статистика вопросов', index=False)

        # 3. Сложность туров
        tour_difficulty = pd.DataFrame({
            'Тур': list(results['tour_difficulty']['tour_difficulty'].keys()),
            'Среднее число взявших': [results['tour_difficulty']['tour_difficulty'][t] for t in range(1, 7)]
        })
        tour_difficulty.to_excel(writer, sheet_name='Сложность туров', index=False)

        # 4. Сложность частей туров
        tour_parts = []
        for tour in range(1, 7):
            parts = results['tour_difficulty']['tour_parts'][tour]
            tour_parts.append({
                'Тур': tour,
                'Начало (1-5)': parts['start'],
                'Середина (6-10)': parts['middle'],
                'Конец (11-15)': parts['end']
            })

        pd.DataFrame(tour_parts).to_excel(writer, sheet_name='Части туров', index=False)

        # 5. Экстремальные вопросы
        # Самые легкие вопросы
        easiest = [(q['question'], q['tour'], q['taken_count']) for q in
                   results['extreme_questions']['easiest_overall']]
        easiest_df = pd.DataFrame(easiest, columns=['Вопрос', 'Тур', 'Взявших'])
        easiest_df.to_excel(writer, sheet_name='Легкие вопросы', index=False)

        # Самые сложные вопросы
        hardest = [(q['question'], q['tour'], q['taken_count']) for q in
                   results['extreme_questions']['hardest_overall']]
        hardest_df = pd.DataFrame(hardest, columns=['Вопрос', 'Тур', 'Взявших'])
        hardest_df.to_excel(writer, sheet_name='Сложные вопросы', index=False)

        # 6. Информация о влиянии вопросов на топ-5
        # Вопросы, которые взяли все топ-5 команд
        common_taken = [(q['question'], q['tour']) for q in results['top_teams']['common_taken']]
        common_taken_df = pd.DataFrame(common_taken, columns=['Вопрос', 'Тур'])
        common_taken_df.to_excel(writer, sheet_name='Взяты всеми топ-5', index=False)

        # Вопросы, которые не взяла ни одна из топ-5 команд
        common_missed = [(q['question'], q['tour']) for q in results['top_teams']['common_missed']]
        common_missed_df = pd.DataFrame(common_missed, columns=['Вопрос', 'Тур'])
        common_missed_df.to_excel(writer, sheet_name='Не взяты топ-5', index=False)

        # Вопросы, разделившие топ-5 команд
        dividing = [(q['question'], q['tour'], q['taken_count'], q['total_taken'])
                    for q in results['top_teams']['dividing_questions']]
        dividing_df = pd.DataFrame(dividing, columns=['Вопрос', 'Тур', 'Взяли из топ-5', 'Всего взявших'])
        dividing_df.to_excel(writer, sheet_name='Разделяющие вопросы', index=False)

        # 7. "Поток" туров
        flow_data = [(tour, score * 100) for tour, score in results['question_flow']['tour_flow'].items()]
        flow_df = pd.DataFrame(flow_data, columns=['Тур', 'Поток (%)'])
        flow_df.to_excel(writer, sheet_name='Поток туров', index=False)

        # Форматирование с использованием xlsxwriter
        workbook = writer.book

        # Добавляем форматирование для листов
        for sheet_name in writer.sheets:
            worksheet = writer.sheets[sheet_name]

            # Автоподбор ширины столбцов для xlsxwriter
            # Получаем DataFrame для текущего листа
            if sheet_name == 'Команды':
                df = df_teams
            elif sheet_name == 'Статистика вопросов':
                df = df_stats
            elif sheet_name == 'Сложность туров':
                df = tour_difficulty
            elif sheet_name == 'Части туров':
                df = pd.DataFrame(tour_parts)
            elif sheet_name == 'Легкие вопросы':
                df = easiest_df
            elif sheet_name == 'Сложные вопросы':
                df = hardest_df
            elif sheet_name == 'Взяты всеми топ-5':
                df = common_taken_df
            elif sheet_name == 'Не взяты топ-5':
                df = common_missed_df
            elif sheet_name == 'Разделяющие вопросы':
                df = dividing_df
            elif sheet_name == 'Поток туров':
                df = flow_df
            else:
                continue  # Пропускаем, если нет соответствующего DataFrame

            # Устанавливаем ширину колонок на основе данных
            for i, col in enumerate(df.columns):
                # Находим максимальную длину в колонке
                column_len = max(
                    # Длина заголовка
                    len(str(col)),
                    # Длина значений (с ограничением в первых 100 строках для скорости)
                    df.iloc[:100, i].astype(str).map(len).max() if len(df) > 0 else 0
                )
                # Устанавливаем ширину с небольшим запасом
                worksheet.set_column(i, i, column_len + 2)

if __name__ == "__main__":
    # Путь к файлу с данными
    file_path = "dilizhan_detailed.csv"

    # Директория для результатов
    output_dir = "tournament_analysis_results"

    # Запускаем анализ
    run_analysis(file_path, output_dir)