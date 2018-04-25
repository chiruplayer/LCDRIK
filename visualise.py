import csv

from sklearn.metrics import classification_report , accuracy_score , auc ,average_precision_score

#import matplotlib as plt
import matplotlib.pyplot as plt

healthy = []
disease = []
picname = []
cutoff = 0.5
y_pred = []
y_true = []

line = 0

with open("sub.csv") as f:
	reader = csv.reader(f)
	for row in reader:
		if line == 0:
			line +=1
		else:
			healthy.append(row[0])
			disease.append(row[1])
			picname.append(row[2])

			y_pred.append(int(float(row[0]) < cutoff))
			y_true.append(int(len(row[2]) > 8 ))

print(classification_report(y_true , y_pred))
print(accuracy_score(y_true , y_pred))
#print(auc(y_true , y_pred))
print(average_precision_score(y_true , y_pred))

##################################################################

f, ax1 = plt.subplots( sharex=True, figsize=(12,4))
dist = [x for x in range(len(y_true)) if y_true[x] ==1]
ax1.hist(dist, bins = 100)
ax1.set_title('Input distribution')

plt.xlabel(' Input Number')
plt.ylabel('Healthy / Diseased ')
plt.show()

def makePlot(vals , title ):
	barlist=plt.bar( [i for i in range(len(vals))] , vals)
	barlist[0].set_color('r')
	barlist[1].set_color('g')
	barlist[2].set_color('b')
	barlist[3].set_color('y')
	barlist[4].set_color('k')
	if len(vals) == 6:
		barlist[5].set_color('#aa00ff')
	plt.title(title)
	plt.show()
	#barlist.savefig("graphs/"+title + '.png')

def accuracy(x):
	return (x["tp"] + x["tn"] )/ (x["tp"] + x["fn"] +x["tn"] +x["fp"])

def recall(x):
	return x["tp"] / (x["tp"] + x["fn"])

def precision(x):
	return x["tp"] / (x["tp"] + x["fp"])

def f1(x):
	return 2 * (precision(x) * recall(x))/(precision(x)+recall(x))

lcd = [
{"tp":50 , "tn": 50 , "fp":0 , "fn":0 },	#lcd128
{"tp":50 , "tn":50 , "fp":0 , "fn":0 },		#lcd1024
{"tp":41 , "tn":38 , "fp":12 , "fn":9 },	#lcdcolour
{"tp":20 , "tn":48 , "fp":2 , "fn":30 },	#lcdwoaug
{"tp":50 , "tn":50 , "fp":0 , "fn":0 }		#lcdwaug
]

yolo = [
{"tp":18 , "tn":39 , "fp":11 , "fn":32 }, #yolo128
{"tp":21 , "tn":38 , "fp":12 , "fn":29 }, #yolo1024
{"tp":21 , "tn":25 , "fp":25 , "fn":29 },	#yolocolour
{"tp":18 , "tn":39 , "fp":11 , "fn":32 },	#yolowoaug
{"tp":31 , "tn":42 , "fp":8 , "fn":19 },	#yolowaug
{"tp":41 , "tn":36 , "fp":14 , "fn":9 },	#yoloadam
]

makePlot([accuracy(x) for x in lcd] , "LCD Accuracy")
makePlot([precision(x) for x in lcd] , "LCD precision")
makePlot([recall(x) for x in lcd] , "LCD Recall")
makePlot([f1(x) for x in lcd] , "LCD f1")

makePlot([accuracy(x) for x in yolo] , "yolo Accuracy")
makePlot([precision(x) for x in yolo] , "yolo precision")
makePlot([recall(x) for x in yolo] , "yolo Recall")
makePlot([f1(x) for x in yolo] , "yolo f1")

#fig.savefig('path/to/save/image/to.png')
