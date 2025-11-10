import re
import pandas as pd

retro_path = "datasets/retro/data.txt"
with open(retro_path, "r", encoding="utf-8") as file:
    retro_raw = file.readlines()

retro_data = []
current_initiative = ""
current_epic = ""
current_type = ""
offset = 3
for line in retro_raw:
    match = re.match(r"^(\d+(?:\.\d+)*)", line)
    if match:
        numbering = match.group(1)
        depth = numbering.count('.')
        if depth == 0:
            current_initiative = line.strip()[offset:]
        elif depth == 1:
            current_epic = line.strip()[offset+depth:].strip()
        elif depth == 2:
            current_type = line.strip()[offset+depth+1:].strip()
        elif depth == 3:
            user_story = line.strip()[offset+depth+2:].strip()
            retro_data.append({
                "initiative": current_initiative,
                "epic": current_epic,
                "type": current_type,
                "user_story": user_story
            })
    elif line.strip() != "":
        if line[0] == "•":
            retro_data[-1]["user_story"] += " " + line.strip()[1:].strip()
        else:
            user_story = line.strip()
            retro_data.append({
                "initiative": current_initiative,
                "epic": current_epic,
                "type": current_type,
                "user_story": user_story
            }) 
retro_df = pd.DataFrame(retro_data)
retro_df.to_csv("datasets/retro/retro_backlog.csv", index=False)