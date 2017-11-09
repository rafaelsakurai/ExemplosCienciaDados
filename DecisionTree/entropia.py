import math

def entropia(probabilidades):
	return sum(- p * math.log(p, 2) for p in probabilidades if p)

p1 = 50.0/150.0
p2 = 50.0/150.0
p3 = 50.0/150.0
print entropia([p1, p2, p3])

p1 = 50.0/50.0
p2 = 0.0/50.0
p3 = 0.0/50.0
print entropia([p1, p2, p3])

p1 = 0.0/100.0
p2 = 50.0/100.0
p3 = 50.0/100.0
print entropia([p1, p2, p3])

p1 = 0.0/54.0
p2 = 49.0/54.0
p3 = 5.0/54.0
print entropia([p1, p2, p3])