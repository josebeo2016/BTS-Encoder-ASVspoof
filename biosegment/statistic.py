with open("data/segments", "r") as f:
    line = f.readlines()


time = 0
for i in line[201:]:
    _, _,start, end = i.strip().split()
    time+=float(end) - float(start)
print(time)