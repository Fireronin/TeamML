#%%
import matplotlib.pyplot as plt

data = [
    {"objective_type": "monster_types", "sub_type": "monsterType_BARON_NASHOR", "team_Blue_count": 46153, "team_Red_count": 49726, "team_Blue_win_ratio": 0.803458063397829, "team_Red_win_ratio": 0.8299280054699755, "average_win_ratio": 0.8166930344339023},
    {"objective_type": "monster_types", "sub_type": "monsterType_DRAGON", "team_Blue_count": 51500, "team_Red_count": 68372, "team_Blue_win_ratio": 0.6176310679611651, "team_Red_win_ratio": 0.6264698999590476, "average_win_ratio": 0.6220504839601063},
    {"objective_type": "monster_types", "sub_type": "monsterType_RIFTHERALD", "team_Blue_count": 66089, "team_Red_count": 53869, "team_Blue_win_ratio": 0.5525881765498041, "team_Red_win_ratio": 0.6125415359483191, "average_win_ratio": 0.5825648562490615},
    {"objective_type": "dragon_types", "sub_type": "monsterSubType_AIR_DRAGON", "team_Blue_count": 24868, "team_Red_count": 30721, "team_Blue_win_ratio": 0.6799501367218916, "team_Red_win_ratio": 0.686175580221998, "average_win_ratio": 0.6830628584719448},
    {"objective_type": "dragon_types", "sub_type": "monsterSubType_CHEMTECH_DRAGON", "team_Blue_count": 25180, "team_Red_count": 30570, "team_Blue_win_ratio": 0.6577839555202541, "team_Red_win_ratio": 0.6664703958128885, "average_win_ratio": 0.6621271756665713},
    {"objective_type": "dragon_types", "sub_type": "monsterSubType_EARTH_DRAGON", "team_Blue_count": 25239, "team_Red_count": 30257, "team_Blue_win_ratio": 0.6841000039621221, "team_Red_win_ratio": 0.6904848464818059, "average_win_ratio": 0.687292425221964},
    {"objective_type": "dragon_types", "sub_type": "monsterSubType_ELDER_DRAGON", "team_Blue_count": 973, "team_Red_count": 1117, "team_Blue_win_ratio": 0.7348406988694759, "team_Red_win_ratio": 0.8352730528200537, "average_win_ratio": 0.7850568758447648},
    {"objective_type": "dragon_types", "sub_type": "monsterSubType_FIRE_DRAGON", "team_Blue_count": 25224, "team_Red_count": 30434, "team_Blue_win_ratio": 0.6833967649857279, "team_Red_win_ratio": 0.6946507195899323, "average_win_ratio": 0.68902374228783},
    {"objective_type": "dragon_types", "sub_type": "monsterSubType_HEXTECH_DRAGON", "team_Blue_count": 24842, "team_Red_count": 30329, "team_Blue_win_ratio": 0.682674502858063, "team_Red_win_ratio": 0.6922417488212602, "average_win_ratio": 0.6874581258396616},
    {"objective_type": "dragon_types", "sub_type": "monsterSubType_WATER_DRAGON", "team_Blue_count": 25040, "team_Red_count": 30690, "team_Blue_win_ratio": 0.6813498402555911, "team_Red_win_ratio": 0.6937764744216357, "average_win_ratio": 0.6875631573386134},
    {"objective_type": "objective_kills", "sub_type": "type_BUILDING_KILL", "team_Blue_count": 73586, "team_Red_count": 46386, "team_Blue_win_ratio": 0.5882097138042562, "team_Red_win_ratio": 0.6957702755141637, "average_win_ratio": 0.64198999465921},
    {"objective_type": "objective_kills", "sub_type": "type_CHAMPION_KILL", "team_Blue_count": 60319, "team_Red_count": 59674, "team_Blue_win_ratio": 0.5400620036804323, "team_Red_win_ratio": 0.5839729195294433, "average_win_ratio": 0.5620174616049378},
    {"objective_type": "tower_types", "sub_type": "towerType_BASE_TURRET", "team_Blue_count": 61704, "team_Red_count": 42027, "team_Blue_win_ratio": 0.7113315182159989, "team_Red_win_ratio": 0.8544269160301711, "average_win_ratio": 0.782879217123085},
    {"objective_type": "tower_types", "sub_type": "towerType_INNER_TURRET", "team_Blue_count": 69851, "team_Red_count": 46466, "team_Blue_win_ratio": 0.6532905756538918, "team_Red_win_ratio": 0.7827658933413679, "average_win_ratio": 0.7180282344976299},
    {"objective_type": "tower_types", "sub_type": "towerType_NEXUS_TURRET", "team_Blue_count": 55325, "team_Red_count": 34471, "team_Blue_win_ratio": 0.7348757342973339, "team_Red_win_ratio": 0.9214412114531055, "average_win_ratio": 0.8281584728752197},
    {"objective_type": "tower_types", "sub_type": "towerType_OUTER_TURRET", "team_Blue_count": 73586, "team_Red_count": 46386, "team_Blue_win_ratio": 0.5882097138042562, "team_Red_win_ratio": 0.6957702755141637, "average_win_ratio": 0.64198999465921},
    {"objective_type": "building_types", "sub_type": "buildingType_INHIBITOR_BUILDING", "team_Blue_count": 59804, "team_Red_count": 39465, "team_Blue_win_ratio": 0.7073774329476289, "team_Red_win_ratio": 0.8602812618776131, "average_win_ratio": 0.783829347412621},
    {"objective_type": "building_types", "sub_type": "buildingType_TOWER_BUILDING", "team_Blue_count": 73586, "team_Red_count": 46386, "team_Blue_win_ratio": 0.5882097138042562, "team_Red_win_ratio": 0.6957702755141637, "average_win_ratio": 0.64198999465921},
    {"objective_type": "ward_objectives", "sub_type": "type_WARD_KILL", "team_Blue_count": 59106, "team_Red_count": 60871, "team_Blue_win_ratio": 0.4849084695293202, "team_Red_win_ratio": 0.52788684266728, "average_win_ratio": 0.5063976560983001},
    {"objective_type": "ward_objectives", "sub_type": "type_WARD_PLACED", "team_Blue_count": 0, "team_Red_count": 0, "team_Blue_win_ratio": 0.0, "team_Red_win_ratio": 0.0, "average_win_ratio": 0.0}
]

# Define manual mappings for x-axis labels
subtype_labels = {
    "monsterType_BARON_NASHOR": "Baron Nashor",
    "monsterType_DRAGON": "Dragon",
    "monsterType_RIFTHERALD": "Rift Herald",
    "monsterSubType_AIR_DRAGON": "Air",
    "monsterSubType_CHEMTECH_DRAGON": "Chemtech",
    "monsterSubType_EARTH_DRAGON": "Earth",
    "monsterSubType_ELDER_DRAGON": "Elder",
    "monsterSubType_FIRE_DRAGON": "Fire",
    "monsterSubType_HEXTECH_DRAGON": "Hextech",
    "monsterSubType_WATER_DRAGON": "Water",
    "buildingType_INHIBITOR_BUILDING": "Inhibitor",
    "buildingType_TOWER_BUILDING": "Tower",
}

# Filter and group data
monster_data = [item for item in data if item["objective_type"] == "monster_types"]
building_data = [item for item in data if item["objective_type"] == "building_types"]
dragon_data = [item for item in data if item["objective_type"] == "dragon_types"]

def create_win_probability_graph(data, title, xlabel):
    """Creates a grouped bar graph for win probabilities with custom x-axis labels."""
    plt.figure(figsize=(12, 6))
    bar_width = 0.25
    x = range(len(data))

    # Use manual mappings for x-axis labels
    plt.bar([i - bar_width for i in x], [item["team_Blue_win_ratio"] for item in data], width=bar_width, label="Team Blue First Kill")
    plt.bar(x, [item["team_Red_win_ratio"] for item in data], width=bar_width, label="Team Red First Kill")
    plt.bar([i + bar_width for i in x], [item["average_win_ratio"] for item in data], width=bar_width, label="Average Win Ratio")

    plt.xlabel(xlabel, labelpad=10)  # Add label padding for x-axis
    plt.ylabel("Win Probability")
    plt.title(title)
    plt.xticks(x, [subtype_labels.get(item["sub_type"], item["sub_type"]) for item in data], rotation=45, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{title}.png")
    plt.show()
    # save to file
    

# Create graphs with custom x-axis labels
create_win_probability_graph(monster_data, "Win Probability Given First Monster Kill", "Monster Type")
create_win_probability_graph(building_data, "Win Probability Given First Building Kill", "Building Type")
create_win_probability_graph(dragon_data, "Win Probability Given First Dragon Kill", "Dragon Type")
# %%
