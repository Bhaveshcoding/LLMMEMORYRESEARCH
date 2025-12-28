import json
s = 0
t = 0
with open("memory.json", 'r')  as f:
    memory = json.load(f)
    for i in memory:
        t += 1
        s += float(i['meaningfulness_score'])

print(f'{s/t:.2f}/1.0')