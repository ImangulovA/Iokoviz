import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns


def analyze_tournament_data(file_path):
    """
    Анализирует данные турнира из CSV-файла и выводит статистику по каждому туру.

    Parameters:
    file_path (str): Путь к CSV-файлу с результатами турнира
    """
    # Загрузка данных
    df = pd.read_csv(file_path, sep='\t')

    # Определение туров
    tour_columns = ['I', 'II', 'III', 'IV', 'V', 'VI']

    # Создание словаря для хранения статистики по каждому туру
    tour_stats = {}

    # Анализ каждого тура
    for tour in tour_columns:
        # Получение данных тура
        tour_data = df[tour]

        # Расчет основных статистических показателей
        stats_data = {
            'Среднее': tour_data.mean(),
            'Медиана': tour_data.median(),
            'Мода': tour_data.mode().tolist(),
            'Минимум': tour_data.min(),
            'Максимум': tour_data.max(),
            'Станд. отклонение': tour_data.std(),
            'Количество команд': len(tour_data)
        }

        # Добавляем коэффициент вариации (показывает относительную меру разброса)
        stats_data['Коэф. вариации (%)'] = (stats_data['Станд. отклонение'] / stats_data['Среднее']) * 100

        # Добавляем межквартильный размах
        q1 = tour_data.quantile(0.25)
        q3 = tour_data.quantile(0.75)
        stats_data['Q1 (25%)'] = q1
        stats_data['Q3 (75%)'] = q3
        stats_data['Межквартильный размах'] = q3 - q1

        # Сохраняем статистику для текущего тура
        tour_stats[tour] = stats_data

    return tour_stats, df, tour_columns


def print_tour_stats(tour_stats):
    """
    Выводит статистику по каждому туру в удобочитаемом формате
    """
    print("=" * 80)
    print("СТАТИСТИКА ПО ТУРАМ")
    print("=" * 80)

    for tour, stats in tour_stats.items():
        print(f"\nТУР {tour}:")
        print("-" * 40)
        for key, value in stats.items():
            # Форматируем вывод числовых значений
            if isinstance(value, (int, float)):
                print(f"{key}: {value:.2f}")
            else:
                print(f"{key}: {value}")

    print("\n")


def plot_tour_distributions(df, tour_columns):
    """
    Создает графики распределения результатов для каждого тура
    """
    plt.figure(figsize=(15, 10))

    for i, tour in enumerate(tour_columns):
        # Создаем подграфик для текущего тура
        plt.subplot(2, 3, i + 1)

        # Строим гистограмму распределения
        sns.histplot(df[tour], kde=True, bins=range(0, 16))

        # Добавляем среднее значение
        plt.axvline(df[tour].mean(), color='red', linestyle='--',
                    label=f'Среднее = {df[tour].mean():.2f}')

        # Добавляем медиану
        plt.axvline(df[tour].median(), color='green', linestyle='-',
                    label=f'Медиана = {df[tour].median():.2f}')

        plt.title(f'Распределение результатов в туре {tour}')
        plt.xlabel('Количество взятых вопросов')
        plt.ylabel('Частота')
        plt.legend()

    plt.tight_layout()
    plt.savefig('tour_distributions.png')
    plt.close()


def plot_correlation_heatmap(df, tour_columns):
    """
    Создает тепловую карту корреляций между турами
    """
    # Вычисляем корреляционную матрицу
    corr_matrix = df[tour_columns].corr()

    # Создаем тепловую карту
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1,
                linewidths=0.5, fmt='.2f')
    plt.title('Корреляция между турами')
    plt.tight_layout()
    plt.savefig('tour_correlations.png')
    plt.close()


def plot_tour_difficulty(tour_stats, tour_columns):
    """
    Создает график, показывающий сложность каждого тура на основе среднего балла
    """
    # Извлекаем средние значения для каждого тура
    means = [tour_stats[tour]['Среднее'] for tour in tour_columns]

    # Сортируем туры по среднему баллу (от высшего к низшему)
    sorted_indices = np.argsort(means)[::-1]
    sorted_tours = [tour_columns[i] for i in sorted_indices]
    sorted_means = [means[i] for i in sorted_indices]

    # Создаем график
    plt.figure(figsize=(12, 6))
    bars = plt.bar(sorted_tours, sorted_means, color='skyblue')

    # Добавляем значения над столбцами
    for bar, value in zip(bars, sorted_means):
        plt.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.1,
                 f'{value:.2f}',
                 ha='center', va='bottom')

    plt.title('Сравнение сложности туров (по среднему баллу)')
    plt.xlabel('Туры')
    plt.ylabel('Средний балл')
    plt.ylim(0, max(means) + 1)
    plt.savefig('tour_difficulty.png')
    plt.close()


def plot_boxplot_comparison(df, tour_columns):
    """
    Создает диаграмму "ящик с усами" для сравнения распределений результатов по турам
    """
    plt.figure(figsize=(12, 8))
    sns.boxplot(data=df[tour_columns])
    plt.title('Сравнение распределений результатов по турам')
    plt.xlabel('Тур')
    plt.ylabel('Количество взятых вопросов')
    plt.savefig('tour_boxplots.png')
    plt.close()


def analyze_team_performance(df, tour_columns):
    """
    Анализирует стабильность выступления команд по турам
    """
    # Вычисляем стандартное отклонение результатов каждой команды
    df['stability'] = df[tour_columns].std(axis=1)

    # Сортируем команды по стабильности (от наиболее стабильных к наименее)
    stability_df = df[['Команда', 'Сумма', 'stability']].sort_values('stability')

    # Выводим топ-5 самых стабильных команд
    print("=" * 80)
    print("Команды по стабильности")
    print("=" * 80)
    print(stability_df[['Команда', 'Сумма', 'stability']].to_string(index=False))

    # Выводим топ-5 самых нестабильных команд
    print("\n")
    print("=" * 80)
    print("ТОП-5 САМЫХ НЕСТАБИЛЬНЫХ КОМАНД")
    print("=" * 80)
    print(stability_df.tail(5)[['Команда', 'Сумма', 'stability']].to_string(index=False))


def run_analysis(file_path):
    """
    Запускает полный анализ данных турнира
    """
    # Получаем статистику и данные
    tour_stats, df, tour_columns = analyze_tournament_data(file_path)

    # Выводим статистику по турам
    print_tour_stats(tour_stats)

    # Создаем визуализации
    # Отключите визуализации, если не требуется генерировать графики
    plot_tour_distributions(df, tour_columns)
    plot_correlation_heatmap(df, tour_columns)
    plot_tour_difficulty(tour_stats, tour_columns)
    plot_boxplot_comparison(df, tour_columns)


    # Анализируем стабильность выступления команд
    analyze_team_performance(df, tour_columns)

    # Выводим рейтинг туров по сложности
    means = {tour: stats['Среднее'] for tour, stats in tour_stats.items()}
    sorted_tours = sorted(means.items(), key=lambda x: x[1], reverse=True)

    print("\n")
    print("=" * 80)
    print("РЕЙТИНГ ТУРОВ ПО СЛОЖНОСТИ (от самого простого к самому сложному)")
    print("=" * 80)

    for i, (tour, mean) in enumerate(sorted_tours, 1):
        print(f"{i}. Тур {tour}: {mean:.2f} вопросов")


if __name__ == "__main__":
    # Путь к файлу с данными
    file_path = "overall_results.csv"

    # Запускаем анализ
    run_analysis(file_path)