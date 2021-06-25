# coding: utf-8
# random-erasing

# parameters
#   IRE or ORE: Image-aware Random Erasing
#   probability = 0.5
#   ratio of area = 0.02 ~ 0.4
#   ratio of aspect r1 = 1/r2 = 0.3

import glob
import sys
import random
import os
import math
from PIL import Image, ImageDraw

image_list = glob.glob(sys.argv[1]+"*")
# print(image_list)

for iter in range(20):
    for i in image_list:
        p = 0.5
        W = 641
        H = 361
        
        while(1):
            area_rate = random.uniform(0.02, 0.4)
            aspect_h = 1
            aspect_w = random.uniform(0.3, 3)
            # print("aspect_w", aspect_w, "areaa_rate", area_rate)
            
            threshold = random.uniform(0, 1)
            _img = Image.open(i)
            w, h = _img.size
    
            area = w * h * area_rate
            x = math.sqrt(area / aspect_w)
            # モード、サイズ、塗りつぶす色を引数で指定
            BLACK_RATE = random.randint(0, 255)
            W = int(aspect_w * x)
            if int(aspect_w * x) == 640:
                W -= 1
            H = int(x)
            if int(x) == 360:
                H -= 1
            if W <= 640 and H <= 360:
                # print(W, H)
                break
        
        if threshold >= 0.5:
            erase_area = Image.new('RGB', (W, H), (BLACK_RATE, BLACK_RATE, BLACK_RATE))
            _img.paste(erase_area, (random.randint(0, 640 - int(aspect_w * x)), random.randint(0, 360 - int(x))))
        else:
            pass
        
        l = i.split("/")
        file_name = l[-1].replace(".png", "")
        DIR = "generated_images_RE"
        if not os.path.exists(DIR):
            os.makedirs(DIR)
        _img.save('generated_images_RE/' + str(l[-1].replace(".png", ""))+ "_" + str(iter) + '.png', quality=95)

# from ann_txt, reading one by one
for iter in range(20):
    # reading file path and ann
    ann = sys.argv[2]
    with open(ann) as f:
        files = f.readlines()
        
    for i, k in enumerate(files):
        path, x, y = k.split(" ")
        # print(path, x, y)
        x = float(x)
        y = float(y)
    
        # generated_images_RE/0000.png
        l = path.split("/")
        l[-1] = l[-1].replace(".png", "") + "_" + str(iter) + ".png"
        path = str(l[0]) + str(l[-1])
        with open("ann_RE.txt", mode="a") as f:
            f.write(str(path) + " " + str(x) + " " + str(y) + "\n")