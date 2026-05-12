import pandas as pd
import os
from pypdf import PdfWriter, PdfReader

main_path = "datasets/trident"
entities_path = os.path.join(main_path, "trident_systementities.pdf")
user_stories_path = os.path.join(main_path, "trident_userstories.pdf")
users_path = os.path.join(main_path, "trident_users.pdf")

# Specification Preprocessing
merger = PdfWriter()
for path in [entities_path, users_path]:
    merger.append(path)
merger.write(os.path.join(main_path, "trident_specs.pdf"))
merger.close()

# Backlog Preprocessing
reader = PdfReader(user_stories_path)
trident_raw = [page.extract_text() for page in reader.pages]
trident_raw = "\n".join(trident_raw)
trident_raw = trident_raw.replace("•", "")
trident_raw = trident_raw.split("\n")

trident_data = []
current_epic = ""
for line in trident_raw[1:]:
    if line.strip().startswith("As"):
        if current_epic:
            user_story = line.strip()
            trident_data.append(
                {"epic": current_epic, "user_story": user_story}
            )
    elif line.strip():
        if line.strip()[0].isupper() \
            and not line.strip().startswith("I ") \
            and not line.strip().startswith("Schema"):
            current_epic = line.strip()
        else:
            trident_data[-1]["user_story"] += " " + line.strip()
    else:
        continue
trident_df = pd.DataFrame(trident_data)
trident_df.to_csv(os.path.join(main_path, "trident_backlog.csv"), index=False)