import sqlite3, json, os

DB  = "/scratch-ssd/ms25yt/factscore_data/enwiki-20230401.db"
ENT = os.path.expanduser("~/SimpleQA/gen_longform_data/sampled_100_entities.jsonl")

topics = []
with open(ENT) as f:
    for line in f:
        line = line.strip()
        if line:
            topics.append(json.loads(line)["entity"])

cur = sqlite3.connect(DB).cursor()
hit, miss = [], []
for t in topics:
    if cur.execute("SELECT 1 FROM documents WHERE title=?", (t,)).fetchone():
        hit.append(t)
    else:
        miss.append(t)

print(f"命中 {len(hit)}/{len(topics)}")
print("缺失:")
for m in miss:
    print("   ", repr(m))
