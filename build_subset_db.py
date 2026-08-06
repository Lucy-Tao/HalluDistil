import sqlite3, json, os

SRC = "/scratch-ssd/ms25yt/factscore_data/enwiki-20230401.db"
DST = os.path.expanduser("~/SimpleQA/factscore_data/enwiki-20230401-subset.db")
ENT = os.path.expanduser("~/SimpleQA/gen_longform_data/sampled_100_entities.jsonl")

topics = []
with open(ENT) as f:
    for line in f:
        line = line.strip()
        if line:
            topics.append(json.loads(line)["entity"])

if os.path.exists(DST):
    raise SystemExit(f"{DST} 已存在，先手动删除或改名")

src = sqlite3.connect(SRC).cursor()
dst = sqlite3.connect(DST)
dst.execute("CREATE TABLE documents (title PRIMARY KEY, text);")

hit, miss = 0, []
for t in topics:
    row = src.execute("SELECT text FROM documents WHERE title=?", (t,)).fetchone()
    if row:
        dst.execute("INSERT INTO documents VALUES (?, ?)", (t, row[0]))
        hit += 1
    else:
        miss.append(t)
dst.commit()
dst.close()

size = os.path.getsize(DST) / 1e6
print(f"命中 {hit}/{len(topics)}，缺失 {miss}")
print(f"子集大小 {size:.1f} MB，写到 {DST}")
