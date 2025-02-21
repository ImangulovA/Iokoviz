import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
import pandas as pd
import numpy as np

# Tournament configuration
QUESTIONS_PER_TOUR = 12  # Can be changed to any number
NUMBER_OF_TOURS = 9  # Can be changed to any number
TOTAL_QUESTIONS = QUESTIONS_PER_TOUR * NUMBER_OF_TOURS
INPUT_FILENAME = 'tournament_transformed.csv'  # Change this to your input file name
OUTPUT_FILENAME = INPUT_FILENAME.replace('.txt', '').replace('.csv', '') + '_animated.gif'

# Read data from CSV
df = pd.read_csv(INPUT_FILENAME)
question_cols = [f'Q{i}' for i in range(1, TOTAL_QUESTIONS + 1)]

team_data = []
team_names = []
team_short_names = []

for idx, row in df.iterrows():
    scores = []
    current_score = 0
    for q in question_cols:
        current_score += row[q]
        scores.append(current_score)
    team_data.append(scores)
    team_names.append(f"{row['Название']} ({row['Город']})")

    name = row['Название']
    if len(name) > 10:
        # Get first letters of each word
        short_name = ''.join(word[0] for word in name.split() if word)
    else:
        short_name = name
    team_short_names.append(short_name)

# Setup figure
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['grid.alpha'] = 0.3
fig, ax = plt.subplots(figsize=(20, 12))


def get_y_range(scores_list, frame):
    current_scores = [scores[frame] for scores in scores_list]
    top_20_scores = sorted(current_scores, reverse=True)[:20]
    min_score = min(top_20_scores)
    max_score = max(top_20_scores)
    score_range = max(max_score - min_score, QUESTIONS_PER_TOUR)  # Ensure minimum range of one tour
    padding = score_range * 0.1

    y_min = max(0, min_score - padding)
    y_max = max_score + padding

    # Ensure minimum range of one tour
    if (y_max - y_min) < QUESTIONS_PER_TOUR:
        y_max = y_min + QUESTIONS_PER_TOUR

    return y_min, y_max


def get_x_range(frame):
    current_tour = frame // QUESTIONS_PER_TOUR
    start = max(0, (current_tour * QUESTIONS_PER_TOUR) - QUESTIONS_PER_TOUR)
    end = (current_tour + 1) * QUESTIONS_PER_TOUR
    return start, end


def get_tour_score(scores, frame):
    current_tour = frame // QUESTIONS_PER_TOUR
    tour_start = current_tour * QUESTIONS_PER_TOUR
    current_question = frame % QUESTIONS_PER_TOUR

    if tour_start >= len(scores):
        return 0

    if tour_start == 0:
        tour_score = scores[frame]
    else:
        previous_tour_score = scores[tour_start - 1]
        tour_score = scores[frame] - previous_tour_score

    return tour_score


def init():
    ax.clear()
    ax.set_xlabel('Номер вопроса', fontsize=14)
    ax.set_ylabel('Количество правильных ответов', fontsize=14)
    ax.set_title('Динамика результатов команд', fontsize=16, pad=20)
    ax.grid(True, linestyle='--', alpha=0.3)
    return []


def update(frame):
    ax.clear()

    x_start, x_end = get_x_range(frame)
    y_min, y_max = get_y_range(team_data, frame)

    ax.set_xlim(x_start, x_end)
    ax.set_ylim(y_min, y_max)

    current_tour = (frame // QUESTIONS_PER_TOUR) + 1
    current_question = (frame % QUESTIONS_PER_TOUR) + 1

    ax.set_xlabel('Номер вопроса', fontsize=14)
    ax.set_ylabel('Количество правильных ответов', fontsize=14)
    ax.set_title(f'Тур {current_tour}, Вопрос {current_question} (всего вопросов: {frame + 1})',
                 fontsize=16, pad=20)

    ax.grid(True, linestyle='--', alpha=0.3)

    # Draw all teams in background
    for i, (team_scores, team_name) in enumerate(zip(team_data, team_names)):
        visible_range = range(max(0, x_start), frame + 1)
        visible_scores = team_scores[max(0, x_start):frame + 1]

        line = ax.plot(visible_range, visible_scores,
                       alpha=0.15, linewidth=1, color='gray')[0]
        if frame < len(team_scores):
            ax.plot(frame, team_scores[frame], 'o',
                    color='gray', alpha=0.15)

    # Highlight top-20 teams
    current_scores = [(team_scores[frame], i, team_names[i], team_short_names[i])
                      for i, team_scores in enumerate(team_data)]
    top_20 = sorted(current_scores, reverse=True)[:20]
    colors = plt.cm.tab20(np.linspace(0, 1, 20))

    for (score, idx, name, short_name), color in zip(top_20, colors):
        visible_range = range(max(0, x_start), frame + 1)
        visible_scores = team_data[idx][max(0, x_start):frame + 1]

        tour_score = get_tour_score(team_data[idx], frame)

        line = ax.plot(visible_range, visible_scores,
                       linewidth=3, color=color,
                       label=f"{name} ({tour_score}/{int(score)})")

        ax.plot(frame, score, 'o', color=color, markersize=8)
        ax.annotate(short_name,
                    (frame, score),
                    xytext=(5, 5),
                    textcoords='offset points',
                    color=color,
                    fontsize=10,
                    fontweight='bold')

    # Add tour separators
    for tour in range(QUESTIONS_PER_TOUR, TOTAL_QUESTIONS, QUESTIONS_PER_TOUR):
        if tour <= frame and tour >= x_start and tour <= x_end:
            ax.axvline(x=tour, color='gray', linestyle='--', alpha=0.3)

    # Configure legend
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5),
              fontsize=12,
              bbox_transform=ax.transAxes,
              frameon=True, facecolor='white', edgecolor='none',
              markerscale=2)

    plt.tight_layout()
    return []


# Create frames with pauses at the end of each tour
frames = []
for i in range(TOTAL_QUESTIONS):
    frames.append(i)
    if (i + 1) % QUESTIONS_PER_TOUR == 0 and i < TOTAL_QUESTIONS - 1:
        frames.extend([i] * 10)  # 1-second pause

# Create animation
anim = FuncAnimation(fig, update, frames=frames,
                     init_func=init, blit=True,
                     interval=100)

# Save animation
anim.save(OUTPUT_FILENAME, writer='pillow', dpi=150)