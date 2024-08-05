import os
import shutil
import cv2
import sys
from PIL import Image
import numpy as np
import shapely.geometry as shgeo

def dota2yolo(imgpath,txtpath,yolopath,classname):
    """
    :param imgpath: the path of images
    :param txtpath: the path of txt in dota format
    :param yolopath: the path of txt in YOLO format
    :param classname: the category you selected
    :return:
            txt format: id x y w h
  
    """
    if os.path.exists(yolopath):
        shutil.rmtree(yolopath)
    os.makedirs(yolopath)
    filelist=GetFileFromThisRootDir(txtpath)
    for fullname in filelist:
       objects=parse_dota_poly(fullname)
       '''
       objects=
       [{'name':'ship',
       'difficult':'1',
       'poly':[(1,1),(2,2),(3,3),(4,4)]
       'area':1159.5
       }]
       '''
       name=os.path.splitext(os.path.basename(fullname))[0]
       img_fullname=os.path.join(imgpath,name+'.jpg')
       img =Image.open(img_fullname)
       img_w,img_h=img.size
       with open(os.path.join(yolopath,name+'.txt'),'w') as f:
           for obj in objects:
               poly=obj['poly']
               bbox = np.array(dots4ToRecC(poly,img_w,img_h))
               if (sum(bbox <= 0) + sum(bbox >= 1)) >= 1:  # 若bbox中有<=0或>= 1的元素则将该box排除
                    continue
               if (obj['name'] in classname):
                    id = classname.index(obj['name'])  # id=类名的索引 比如'plane'对应id=0
               else:
                    continue
               outline = str(id) + ' ' + ' '.join(list(map(str, bbox)))  # outline='id x y w h'
               f.write(outline + '\n')  # 写入txt文件中并加上换行符号 \n 

def dots4ToRecC(poly, img_w, img_h):
    """
    求poly四点坐标的最小外接水平矩形,并返回yolo格式的矩形框表现形式xywh_center(归一化)
    @param poly: poly – poly[4] [x,y]
    @param img_w: 对应图像的width
    @param img_h: 对应图像的height
    @return: x_center,y_center,w,h(均归一化)
    """
    xmin, ymin, xmax, ymax = dots4ToRec4(poly)
    x = (xmin + xmax)/2
    y = (ymin + ymax)/2
    w = xmax - xmin
    h = ymax - ymin
    return x/img_w, y/img_h, w/img_w, h/img_h

def dots4ToRec4(poly):
    """
    求出poly四点的最小外接水平矩形
    @param poly: poly[4]  [x,y]
    @return: xmin,xmax,ymin,ymax
    """
    xmin, xmax, ymin, ymax = min(poly[0][0], min(poly[1][0], min(poly[2][0], poly[3][0]))), \
                            max(poly[0][0], max(poly[1][0], max(poly[2][0], poly[3][0]))), \
                             min(poly[0][1], min(poly[1][1], min(poly[2][1], poly[3][1]))), \
                             max(poly[0][1], max(poly[1][1], max(poly[2][1], poly[3][1])))
    return xmin, ymin, xmax, ymax

def GetFileFromThisRootDir(dir,ext = None):#获取文件夹下的文件路径 ext指定扩展
  allfiles = []
  needExtFilter = (ext != None)
  for root,dirs,files in os.walk(dir):
    for filespath in files:
      filepath = os.path.join(root, filespath)
      extension = os.path.splitext(filepath)[1][1:]
      if needExtFilter and extension in ext:
        allfiles.append(filepath)
      elif not needExtFilter:
        allfiles.append(filepath)
  return allfiles


def parse_dota_poly(filename):
    """
        parse the dota ground truth in the format:
        [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
    """
    objects = []
    #print('filename:', filename)
    f = []
    if (sys.version_info >= (3, 5)):
        fd = open(filename, 'r')
        f = fd
    elif (sys.version_info >= 2.7):
        fd = codecs.open(filename, 'r')
        f = fd
    # count = 0
    while True:
        line = f.readline()
        # count = count + 1
        # if count < 2:
        #     continue
        if line:
            splitlines = line.strip().split(' ')
            object_struct = {}
            ### clear the wrong name after check all the data
            #if (len(splitlines) >= 9) and (splitlines[8] in classname):
            if (len(splitlines) < 9):
                continue
            if (len(splitlines) >= 9):
                    object_struct['name'] = splitlines[8]
            if (len(splitlines) == 9):
                object_struct['difficult'] = '0'
            elif (len(splitlines) >= 10):
                # if splitlines[9] == '1':
                # if (splitlines[9] == 'tr'):
                #     object_struct['difficult'] = '1'
                # else:
                object_struct['difficult'] = splitlines[9]
                # else:
                #     object_struct['difficult'] = 0
            object_struct['poly'] = [(float(splitlines[0]), float(splitlines[1])),
                                     (float(splitlines[2]), float(splitlines[3])),
                                     (float(splitlines[4]), float(splitlines[5])),
                                     (float(splitlines[6]), float(splitlines[7]))
                                     ]
            gtpoly = shgeo.Polygon(object_struct['poly'])
            object_struct['area'] = gtpoly.area
            # poly = list(map(lambda x:np.array(x), object_struct['poly']))
            # object_struct['long-axis'] = max(distance(poly[0], poly[1]), distance(poly[1], poly[2]))
            # object_struct['short-axis'] = min(distance(poly[0], poly[1]), distance(poly[1], poly[2]))
            # if (object_struct['long-axis'] < 15):
            #     object_struct['difficult'] = '1'
            #     global small_count
            #     small_count = small_count + 1
            objects.append(object_struct)
        else:
            break
    return objects
classname=['car', 'truck', 'bus', 'van', 'feright_car','feright']
dota2yolo('E:\yolov5-master\yolo_obb+thermal\SLBAF-Net-20.04\yolov5-dual\data\images','E:\yolov5-master\yolo_obb+thermal\SLBAF-Net-20.04\yolov5-dual\data\labels','C:/Users/33498/Desktop/Summer/SLBAF-Net-20.04/SLBAF-Net-20.04/modules/yolov5-dual/data/labels2',classname)