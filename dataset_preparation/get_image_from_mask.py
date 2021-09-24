import os


all_files=os.listdir('./batch1')

for file in all_files:
    if(".png" in file):
        try:
         os.rename("./images/"+file,"./batch1/images/"+file)
        except:
            print("not_found")
        else:
            print("found")