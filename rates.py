import sys, json, collections, os

print(f"{'abstain':>8} {'acc_all':>8} {'acc_ans':>8}   file")
for f in sys.argv[1:]:
    c = collections.Counter(json.loads(l)["grade"] for l in open(f))
    n = sum(c.values())
    na, cor = c["NOT_ATTEMPTED"], c["CORRECT"]
    ans = n - na
    print(f"{na/n:8.1%} {cor/n:8.1%} {cor/ans if ans else 0:8.1%}   {os.path.basename(f)}")
