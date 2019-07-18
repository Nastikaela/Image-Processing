from flask import Flask, request, jsonify
from flask_restful import Resource, Api
from flask_cors import CORS
import os
import psycopg2
import cv2
import time
from time import gmtime, strftime
import numpy as np
import glob

app = Flask(__name__)
corsr = CORS(app)

@app.route("/api/v1/print", methods = ['POST'])
def exec() :
	face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
	eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

	connection = psycopg2.connect(
	                    database = "mhs",
	                    user="postgres",
	                    password="12345",
	                    host="localhost",
	                    port="5432")
	cursor = connection.cursor()

	PATH_UJI="/home/admins/anaconda3/nanas/wajah/cowo/"        #'C:/xampp/htdocs/myweb/datalatih/' path folder server
	PATH_TRAIN="/home/admins/anaconda3/lib/python3.7/site-packages/flask_restful/template/"

	TM01 = rgb2gray(PATH_TRAIN+'foto1.jpg')
	TM02 = rgb2gray(PATH_TRAIN+'foto2.jpg')
	TM03 = rgb2gray(PATH_TRAIN+'foto4.jpg')
	TM04 = rgb2gray(PATH_TRAIN+'foto5.jpg')
	TM05 = rgb2gray(PATH_TRAIN+'foto3.jpg')
	TM06 = rgb2gray(PATH_TRAIN+'foto6.jpg')
	TM07 = rgb2gray(PATH_TRAIN+'foto7.jpg')

	w, h = TM01.shape[::-1]
	#print("Image Citra ... "+str(w)+"#"+str(h))

	threshold = 0.6


	loop=0
	Nloop=1		
	while (loop<Nloop): 
	    loop=loop+1
	    cocok=0
	    ada=0
	    cursor.execute("select nim from akademik.ak_mahasiswa where flag=%s order by nim desc", ("0"))
	    for row in cursor.fetchall():
	        foto=row[0]
	        format = ".jpg"
	        print(str(foto)+str(format))
	        

	        ada=1
	        
	        AL=(PATH_UJI+foto+format).strip()
	        img_rgb = cv2.imread(AL)
	        b= 400
	        c= 500
	        img_rgb = myresize(img_rgb,b,c)
	        citraUji= cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
	        hsv = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2HSV)
	        
	        jface=cekHitung(face_cascade,eye_cascade,img_rgb,citraUji)

	        if jface>=1:
	            res = cv2.matchTemplate(citraUji,TM01,cv2.TM_CCOEFF_NORMED)
	            loc = np.where( res >= threshold)
	            for pt in zip(*loc[::-1]):
	                cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
	                cocok=1


	            if cocok==0:
	                res = cv2.matchTemplate(citraUji,TM02,cv2.TM_CCOEFF_NORMED)
	                loc = np.where( res >= threshold)
	                for pt in zip(*loc[::-1]):
	                    cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
	                    cocok=2

	            if cocok==0:
	                res = cv2.matchTemplate(citraUji,TM03,cv2.TM_CCOEFF_NORMED)
	                loc = np.where( res >= threshold)
	                for pt in zip(*loc[::-1]):
	                    cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
	                    cocok=3              

	            if cocok==0:
	                res = cv2.matchTemplate(citraUji,TM04,cv2.TM_CCOEFF_NORMED)
	                loc = np.where( res >= threshold)
	                for pt in zip(*loc[::-1]):
	                    cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
	                    cocok=4      

	            if cocok==0:
	                res = cv2.matchTemplate(citraUji,TM05,cv2.TM_CCOEFF_NORMED)
	                loc = np.where( res >= threshold)
	                for pt in zip(*loc[::-1]):
	                    cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
	                    cocok=5 
	                    
	            if cocok==0:
	                res = cv2.matchTemplate(citraUji,TM06,cv2.TM_CCOEFF_NORMED)
	                loc = np.where( res >= threshold)
	                for pt in zip(*loc[::-1]):
	                    cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
	                    cocok=6
	                    
	            if cocok==0:
	                res = cv2.matchTemplate(citraUji,TM07,cv2.TM_CCOEFF_NORMED)
	                loc = np.where( res >= threshold)
	                for pt in zip(*loc[::-1]):
	                    cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
	                    cocok=7 
	            
	            if cocok==0:
	                
	                updateDB2(cursor,connection,foto)
	                rtn = 2
	                delimg(foto)
	            else:
	                lowerred_range = np.array([161, 155, 84], dtype=np.uint8)  #Merah
	                upperred_range = np.array([179, 255, 255], dtype=np.uint8)
	                mask = cv2.inRange(hsv, lowerred_range, upperred_range)
	                v = np.count_nonzero(mask == 255)
	                print (v)
	                if v>=100:
	                    updateDB1(cursor,connection,foto)
	                    time.sleep(5)
	                    rtn = 1
	                else:
	                    #print("background bukan merah")
	                    updateDB3(cursor,connection,foto)
	                    rtn = 3
	                    delimg(foto)
	        else:
	                
	                updateDB2(cursor,connection,foto)
	                time.sleep(5)  #proses upload lama
	                rtn = 2
	                delimg(foto)
	 				
	data = {
		'status' : rtn
	}
	return jsonify({ 'response': data })
	

def delimg(foto):
    format = ".jpg"
    files = glob.glob((PATH_UJI+foto+format).strip())
    for f in files:
       os.remove(f)
 	

def myresize(img,b,c):
    resized = cv2.resize(img, (b,c), fx=0.5, fy=0.5, interpolation = cv2.INTER_AREA)
    return resized


def rgb2gray(img):
    b=300
    c=400
    I= cv2.imread(img)
    #A=myresize(I,b,c)
    #praproses
    G= cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
    return G

def updateDB1(cursor,connection,foto):
    sql = "UPDATE akademik.ak_mahasiswa set flag=1 where nim='%s'" % (foto)       #ganti nama table jadi akademik.ak_mahasiswa
    print(sql)
    v=0
    try:
        cursor.execute(sql)
        connection.commit()
        v=1
    except:
        connection.rollback()
        return v
def updateDB2(cursor,connection,foto):
    sql = "UPDATE akademik.ak_mahasiswa  set flag=2 where nim='%s'" % (foto)   #ganti nama table jadi akademik.ak_mahasiswa
    print(sql)
    v=0
    try:
        cursor.execute(sql)	
        connection.commit()
        v=1
    except:
        connection.rollback()
        return v
def updateDB3(cursor,connection,foto):
    sql = "UPDATE akademik.ak_mahasiswa  set flag=3 where nim='%s'" % (foto)   #ganti nama table jadi akademik.ak_mahasiswa
    print(sql)
    v=0
    try:
        cursor.execute(sql)
        connection.commit()
        v=1
    except:
        connection.rollback()
        return v

    return v

def cekMerah(hsv):
#lower_range = np.array([100, 100, 100], dtype=np.uint8)  #Merah kulit
#upper_range = np.array([189, 255, 255], dtype=np.uint8)

    lower_range = np.array([161, 155, 84])
    upper_range = np.array([179, 255, 255])
    mask = cv2.inRange(hsv, lower_range, upper_range)
    return mask

# def cekBiru(hsv):
#     lower_range = np.array([100, 100, 100], dtype=np.uint8)  #Biru
#     upper_range = np.array([189, 255, 255], dtype=np.uint8)

#     mask = cv2.inRange(hsv, lower_range, upper_range)
#     return mask


def cekHitung(face_cascade,eye_cascade,ximage,gray):
    faces=face_cascade.detectMultiScale(gray, 1.3, 5)
    n=0
    m=0
    h3=0
    for (x,y,w,h) in faces:
        ximage = cv2.rectangle(ximage,(x,y),(x+w,y+h),(189,183,107),2) #BGR
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = ximage[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        n=n+1
        for (ex,ey,ew,eh) in eyes:
            #cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
            m=m+1
            xmata1=ex
            ymata1=ey
        
            xmata2=ex+ew
            ymata2=ey+eh
            
            h1=ymata1-xmata1
            h2=ymata2-ymata2
            h3=abs(h1-h2)    
    if n>1 and m>2:
        out=-1
    elif n==0 and m==0:
        out=-1
    else:
        h1=ymata1-xmata1
        h2=ymata2-ymata2
        h3=abs(h1-h2)

        print("H3="+str(h3))
        out=1
        
    return out 
if __name__ == '__main__':
	app.run(debug=True, port=8080)