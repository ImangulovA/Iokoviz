import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
import pandas as pd
import numpy as np
from matplotlib.font_manager import FontProperties

# Tournament configuration
QUESTIONS_PER_TOUR = 12
NUMBER_OF_TOURS = 9
TOTAL_QUESTIONS = QUESTIONS_PER_TOUR * NUMBER_OF_TOURS
INPUT_FILENAME = 'tournament_transformed.csv'
OUTPUT_FILENAME = INPUT_FILENAME.replace('.csv', '') + '_table_animated.gif'

# Read data from CSV
df = pd.read_csv(INPUT_FILENAME)
question_cols = [f'Q{i}' for i in range(1, TOTAL_QUESTIONS + 1)]

# Dictionary to store previous positions
previous_positions = {}

# Prepare team data
teams = []
for idx, row in df.iterrows():
    name = row['Название']
    short_name = ''.join(word[0] for word in name.split()) if len(name) > 15 else name
    team = {
        'id': row['Номер команды'],
        'name': f"{name} ({row['Город']})",
        'short_name': short_name,
        'scores': [],
        'tour_scores': [0] * NUMBER_OF_TOURS
    }

    current_score = 0
    for q in question_cols:
        current_score += row[q]
        team['scores'].append(current_score)

        tour = (len(team['scores']) - 1) // QUESTIONS_PER_TOUR
        if tour < NUMBER_OF_TOURS:
            team['tour_scores'][tour] = sum(row[f'Q{i + 1}'] for i in range(tour * QUESTIONS_PER_TOUR,
                                                                            len(team['scores'])))

    teams.append(team)

# Setup figure with custom style
plt.rcParams.update({
    'figure.facecolor': 'white',
    'font.size': 12,
    'font.family': 'sans-serif',
    'grid.alpha': 0.3,
    'grid.color': '#cccccc',
})

fig, ax = plt.subplots(figsize=(20, 14))

# Create color palette for team names
team_colors = plt.cm.viridis(np.linspace(0, 0.8, 20))  # Using viridis colormap


def get_position_change(team_id, current_pos, frame):
    if frame == 0:
        previous_positions[team_id] = current_pos
        return "", ""

    prev_pos = previous_positions.get(team_id, current_pos)
    change = prev_pos - current_pos
    previous_positions[team_id] = current_pos

    if change > 0:
        return f"↑{change}", "green"
    elif change < 0:
        return f"↓{-change}", "red"
    return "", ""


def create_table_data(frame):
    current_scores = [(team['scores'][frame],
                       team['name'],
                       team['tour_scores'][frame // QUESTIONS_PER_TOUR],
                       team['id'],
                       i) for i, team in enumerate(teams)]

    current_scores.sort(reverse=True)
    top_20 = current_scores[:20]

    table_data = []
    cell_colors = []

    for rank, (score, name, tour_score, team_id, idx) in enumerate(top_20, 1):
        change_text, change_color = get_position_change(team_id, rank, frame)
        position_text = f"{rank} {change_text}" if change_text else str(rank)

        row_data = [
            position_text,
            name,
            f"{tour_score}",
            f"{score}"
        ]

        # Create color array for each cell in the row
        row_colors = [
            change_color if change_text else 'black',  # Position column
            team_colors[rank - 1],  # Team name column
            'black',  # Tour score
            'black'  # Total score
        ]

        table_data.append(row_data)
        cell_colors.append(row_colors)

    return table_data, cell_colors


def update(frame):
    ax.clear()

    current_tour = (frame // QUESTIONS_PER_TOUR) + 1
    current_question = (frame % QUESTIONS_PER_TOUR) + 1

    ax.set_title(f'Результаты после {frame + 1} вопроса\n(Тур {current_tour}, вопрос {current_question})',
                 pad=20, fontsize=20, color='#2F2F2F')

    ax.axis('off')

    table_data, cell_colors = create_table_data(frame)
    columns = ['Место', 'Команда', 'В туре', 'Всего']
    col_widths = [0.15, 0.45, 0.2, 0.2]

    table = ax.table(cellText=table_data,
                     colLabels=columns,
                     loc='center',
                     cellLoc='left',
                     colWidths=col_widths)

    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(11)

    # Create font properties for Georgia
    georgia_font = FontProperties(family='Georgia', size=11)

    # Style header and cells
    header_color = '#2F5597'
    alt_row_color = '#F8F9FA'

    for (row, col), cell in table.get_celld().items():
        if row == 0:  # Header row
            cell.set_text_props(weight='bold', color='white', fontsize=12)
            cell.set_facecolor(header_color)
        else:
            # Set cell colors and styles
            if col in [2, 3]:  # Numeric columns (tour score and total score)
                cell.set_text_props(ha='right', family='Georgia')
            elif col == 0:  # Position column
                cell.set_text_props(ha='center', color=cell_colors[row - 1][col])
            else:  # Team name
                cell.set_text_props(ha='center', color=cell_colors[row - 1][col])

            # Alternate row backgrounds
            if row % 2:
                cell.set_facecolor(alt_row_color)

            cell.PAD = 0.5

    table.scale(1, 2)  # Make rows taller

    return []


# Create frames with longer pauses
frames = []
for i in range(TOTAL_QUESTIONS):
    frames.append(i)
    if (i + 1) % QUESTIONS_PER_TOUR == 0 and i < TOTAL_QUESTIONS - 1:
        frames.extend([i] * 5)  # 4-second pause between tours
    else:
        frames.extend([i] * 5)  # Slow down regular transitions

# Create animation
anim = FuncAnimation(fig, update, frames=frames,
                     interval=10, blit=True)

# Save animation
anim.save(OUTPUT_FILENAME, writer='pillow', dpi=200)  # Increased DPI for better quality
plt.close()