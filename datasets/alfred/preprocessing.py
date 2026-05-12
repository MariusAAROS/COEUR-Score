import pandas as pd
import os
from pypdf import PdfReader, PdfWriter
import re

folder_path = "datasets/alfred/"
raw_data_path = os.path.join(folder_path, "001-D23UserStoriesReportv15.pdf")

reader = PdfReader(raw_data_path)
alfred_raw = [page.extract_text() for page in reader.pages]
def remove_header(pages):
    cleaned = [page.strip().replace('\n', ' ')[260:].strip() \
                  if page.startswith("ALFRED WP2 Public") \
                     else page.strip().replace('\n', ' ')[210:] \
                        if page.startswith("ALFRED WP2 Focus Group") \
                        else page.strip().replace('\n', ' ') \
                        for page in pages]
    return cleaned
alfred_raw_X = '\n'.join(remove_header(alfred_raw[:42]))
alfred_raw_Y = '\n'.join(alfred_raw[42:])
alfred_raw_X = alfred_raw_X.replace('_', '').replace('.', '').replace('☐', '')
alfred_raw_Y = alfred_raw_Y.replace('_', '').replace('.', '').replace('☐', '')

raw = alfred_raw_Y
raw = alfred_raw_Y
raw = raw.replace('\r', ' ').replace('\n', ' ')  # remove line breaks
raw = re.sub(r'\s+', ' ', raw).strip()

# Split whenever a new story begins: lookahead keeps the delimiter
chunks = re.split(r'(?=\bAs\s+(?:an|a)\b)', raw)

# Remove any leading non-story chunk
if chunks and not chunks[0].startswith('As '):
    chunks = chunks[1:]
chunks = chunks[:-4]

records = []
for c in chunks:
    c = c.strip()
    # Find US id
    id_match = re.search(r'\bUS(\d{3})\b', c)
    if not id_match:
        continue
    us_id = 'US' + id_match.group(1)

    # Summary is text from start until US id
    summary = c[:id_match.start()].strip()

    # After ID: title, priority, user group, tasks, use cases
    tail = c[id_match.end():].strip()

    # Extract priority (first single digit 1-5)
    pr_match = re.search(r'\b([1-5])\b', tail)
    if not pr_match:
        continue
    priority = int(pr_match.group(1))

    # Title is from start of tail to priority
    title = tail[:pr_match.start()].strip()

    # After priority parse user group (take up to tasks T..)
    rest = tail[pr_match.end():].strip()
    ug_match = re.search(r'^(Older Person|Medical Caregiver|Informal Caregiver|Formal Caregiver|Caregiver|Older Adult|Stakeholder|Developer)\b', rest)
    if not ug_match:
        continue
    user_group = ug_match.group(1)
    rest2 = rest[ug_match.end():].strip()

    # Tasks list
    tasks_match = re.search(r'(T\d+(?:,\s*T\d+)*)', rest2)
    if tasks_match:
        tasks = [t.strip() for t in tasks_match.group(1).split(',')]
        after_tasks = rest2[tasks_match.end():].strip()
    else:
        tasks = []
        after_tasks = rest2

    # Use cases
    uc_numbers = re.findall(r'\bUC\s*\d+\b', after_tasks)
    use_cases = [re.sub(r'\s+', '', uc) for uc in uc_numbers]

    records.append({
        'id': us_id,
        'user_story': summary,
        'title': title,
        'priority': priority,
        'user_group': user_group,
        'tasks': tasks,
        'epic': use_cases[0] if use_cases else None,
    })

df_user_stories_simple = pd.DataFrame(records).dropna().reset_index(drop=True)
df_user_stories_simple.to_csv(os.path.join(folder_path, "alfred_backlog.csv"), index=False)

#save first 42 pages as a new pdf
pdf_writer = PdfWriter()
for page in reader.pages[:42]:
    pdf_writer.add_page(page)
with open("datasets/alfred/alfred_specs.pdf", "wb") as f:
    pdf_writer.write(f)