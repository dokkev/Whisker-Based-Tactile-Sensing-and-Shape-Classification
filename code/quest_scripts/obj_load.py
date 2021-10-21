import csv
   
   
obj_tag = 1

wh
with open('obj_param.csv','r')as file:
            filecontent=csv.reader(file)
            row = list(filecontent)
            col = row[obj_tag]
            obj_num = int(col[1])
            obj_name = col[2]
            x = float(col[4])
            y = float(col[5])
            z = float(col[6])
            yaw = float(col[7])
            pitch = float(col[8])
            roll = float(col[9])
            class_name = str(col[10])
            class_num = int(col[11])
            print(class_num)

        obj_tag+=1