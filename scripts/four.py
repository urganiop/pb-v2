import re

text = ''' Из колодца глубиной 5 м подняли ведро массой 8кг.
Совершенная работа при этом равна.
1) 1.6 Дж
2) 16 Дж
Под действием силы тяги 1000 Н автомобиль движется с постоянной
коростью 72 км/ч. Мощность двигателя равна
1) 10 кВт
2)20 кВт
. Выберите, какие приспособления относятся к простым механизмам.
А. Ворот
Б. Наклонная плоскость
ПА
2) Б
. Рычаг находится в равновесии под действием двух сил. Первая сила 4 Н
имеет плечо 15 см. Определите, чему равна вторая сила, если ее плечо 10
см.
ПАН
2) 0,16 Н
Птичка колибри массой 2 г при полете достигает скорости 180 км/ч.
Определите энергию движения этой птички
1) 0,25 Дж
2) 32.4 Дж '''

b = []
for line in text.split('\n'):
    count = 0
    enumeration = False
    for word in line.split():
        if re.match(r'[0-9]{1,2}[\)|\.]', word):
            enumeration = True
        if word.isdigit():
            count += 1
    b.append([len(line.split()), count, enumeration])

for line in b:
    print(line)