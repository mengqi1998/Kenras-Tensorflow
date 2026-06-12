import csv
import random


def calculateBMI(height, weight):
    bmi = weight / ((height / 100) ** 2)
    if bmi < 18.5:
        return 'thin'
    elif bmi < 25:
        return 'normal'
    else:
        return 'fat'


with open('data/bmi.csv', 'w', encoding='utf-8') as csvfile:
    csvfile.write('height,weight,label\n')
    category = {'thin': 0, 'normal': 0, 'fat': 0}
    for i in range(30000):
        h = random.randint(110,220)
        w = random.randint(55,65)
        label = calculateBMI(h, w)
        category[label] += 1
        csvfile.write("%d,%d,%s\n"%(h, w, label))
print(category)
print("generate OK, check directory")