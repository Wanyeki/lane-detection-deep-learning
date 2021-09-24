import os
import cv2
import time
import json
import numpy as np

batch_path="batch1"
image_paths=os.listdir(batch_path+"/images")
print(image_paths)
file= open(os.path.join(batch_path,"road_lane_labels.json"),'r')
file_content=file.read()
json_file=json.loads(file_content)

for annotation in json_file.values():
    path=annotation["filename"]
    full_path=os.path.join(batch_path,"images",path)
    frame=cv2.imread(full_path)
    img=np.zeros_like(frame)
    
    

    print(full_path," ===========")
    annotation["regions"].reverse()
    for shape in annotation["regions"]:
        pts=[]
        count=0
        for x in shape["shape_attributes"]["all_points_x"]:
            y=shape["shape_attributes"]["all_points_y"][count]
            pts.append([x,y]) 
            count=count+1
        coord=np.array(pts)

        for type,value in shape['region_attributes'].items():
            if("not" in value):
                continue
            shape_type=type

        color=(0,0,255)
        if(shape_type=='cars'):
            color=(0,255,0)
        elif(shape_type=='lanes'):
            color=(255,0,0)

        cv2.fillPoly(img,[coord],color)
    mask=img
    cv2.imwrite(os.path.join(batch_path,'masks/',path),mask)
    img=np.concatenate((frame,img),axis=1)
    cv2.imshow("image",img)
    time.sleep(0.3)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
