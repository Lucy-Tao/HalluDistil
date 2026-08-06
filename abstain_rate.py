import sys, json, collections, os

for f in sys.argv[1:]:
    c = collections.Counter(json.loads(l)["grade"] for l in open(f))
    n = sum(c.values())
    print(f"{c['NOT_ATTEMPTED']/n:6.1%}  {c['NOT_ATTEMPTED']:3d}/{n}  {os.path.basename(f)}")
