from PIL import Image,ImageDraw
import glob

def crop(imgname , savename , savepath):
	img = Image.open(imgname).convert("RGBA")
	w , h = img.size
	draw = ImageDraw.Draw(img)
	draw.rectangle(((0, 0), (350, 175)), fill="black")
	draw.rectangle(((0, 2000), (600, 2100)), fill="black")

	img.save("cleaned/"+savepath+"/"+str(savename)+".jpg", "JPEG")
	
def getImages(folder):
	imgs1 = glob.glob("photo/"+folder+"/*.jpg")
	#print(imgs1)
	return imgs1


f1= "DR6727188828"
f2 = "DR71782"

pics1 = getImages(f1)
pics2 = getImages(f2)


for i in range(len(pics1)):
	print(pics1[i])
	crop(pics1[i],i,f1)

for i in range(len(pics2)):
	print(pics2[i])
	crop(pics2[i],i,f2)
	
