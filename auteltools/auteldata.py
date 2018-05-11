import numpy as np
import matplotlib.pyplot as plt
import sys
import xml.etree.ElementTree as ET
import re
import os
from os.path import isfile
import cv2
import pickle
import pandas as pd
from collections import defaultdict
import datetime
from torch.utils.data import DataLoader, Dataset


#TODO 4: Update
#TODO 5: Show stadistics

class Annotation(object):
    def __init__(self, file_name, path_file, width, height, depth, labels):
        """
         Constructor of the XML file for read all the parameters related with the image (Folder, Image associated (.jpg),
         dimensions and classes inside)
         :param folder_name (str)
         :param file_name (str): name of the jpg image
         :param shape_image (int array): shape of the image (width, height, depth)
         :param classes (array of Label): array of classes of the image
         :return:
         """
        self.file_name = file_name
        self.path_file = path_file
        self.width = width
        self.height = height
        self.depth = depth
        self.labels = labels

    def show_annotation(self):
        print(self.file_name)
        print(self.path_file)
        print(self.width)
        print(self.height)
        print(self.depth)
        for i in range(len(self.labels)):
            self.labels[i].show_values()

    def get_labels_name(self):
        names_labels = []
        for i in range(len(self.labels)):
            names_labels.append(self.labels[i].name)

        return names_labels

class Label(object):
    def __init__(self, label_name, xmin, ymin, xmax, ymax):
        """
        Constructor of a class of the image.
        :param label_name (str): name of the label
        :param xmin (int): x min bounding box
        :param ymin (int): y min bounding box
        :param xmax (int): x max bounding box
        :param ymax (int): y max bounding box
        :return:
        """
        self.label_name = label_name
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax

    def get_name(self):
        return self.label_name

    def show_values(self):

        print("Label name: {}".format(self.label_name))
        print("Xmax: {} Xmin: {} Xmax - Xmin: {}".format(self.xmax, self.xmin, self.xmax - self.xmin))
        print("Ymax: {} Ymin: {} Ymax - Ymin: {}".format(self.ymax, self.ymin, self.ymax - self.ymin))

class Autel(Dataset):

    def __init__(self, data_location, read_all_data = False, batch_size = 1):
        """
        Constructor of Autel helper class for reading images and annotations
        :param data_location (str): location of folder data
        :return:
        """
        self.ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
        self.batch_size = batch_size
        self.data_location = data_location
        self.img_dict = dict()
        self.img_wrong = defaultdict(list)


        if read_all_data:

            list_annotation = self.load_data()
            self.create_ann_file(list_annotation)
            self.generate_csv_file()

        else:
            list_annotation = self.load_pkl_file()

        self.annotations = list_annotation

    def __len__(self):
        return (len(self.annotations))



    def load_data(self):
        print("Loading dataset...")

        main_path = os.path.join(os.path.expanduser('~'), self.data_location, 'AutelData')
        list_annotation = []

        for root,dirs,files in os.walk(main_path, topdown=False):
            files.sort()
            for name_file in files:
                if name_file.endswith('.jpg'):
                    self.img_dict[name_file] = os.path.join(root, name_file)

                elif name_file.endswith('.xml'):
                    annotation = self.load_annotation(root, name_file)
                    if annotation != 0:
                        list_annotation.append(annotation)

        print("Dataset loaded")
        return list_annotation

    def generate_csv_file(self):
        if bool(self.img_wrong):
            print("Creating csv with wrong images...")
            df = pd.DataFrame(self.img_wrong,
                              columns=['image_name', 'path', 'shape', 'labels'])
            df.to_csv(os.path.join(os.path.expanduser('~'), 'outAutelData',
                                   'csv_wrong', 'images_wrong_3.csv'))
            print("CSV created")
    def load_pkl_file(self):

        print("Loading pkl file...")

        with open(os.path.join(self.ROOT_DIR,"annotation_file"), "rb") as fp:
            annotations = pickle.load(fp)
        return annotations
        print("pkl loaded")
    def create_ann_file(self,list_annotation):

        print("Creating pkl file...")
        with open("annotation_file", "wb") as fp:  # Pickling
            pickle.dump(list_annotation, fp)

        print("pkl created")
    def load_annotation(self, root, name_xml):

        tree = ET.parse(os.path.join(root, name_xml))
        root = tree.getroot()
        name_jpg = root.find('filename').text


        annotation = 0
        if name_jpg in self.img_dict:
            path_file = self.img_dict[name_jpg]
            width = root.find('size').find('width').text
            height = root.find('size').find('height').text
            depth = root.find('size').find('depth').text

            if self.check_sizes(int(width), int(height), int(depth)):
                labels = self.parse_labels(root)
                annotation = Annotation(name_jpg, path_file, width, height, depth, labels)
            else:
                self.generate_dict_wrong_image(name_jpg, path_file,
                                                int(width), int(height), int(depth), root)
        #else:
            #print("File {} not match with jpg image".format(name_jpg))

        return annotation

    def generate_dict_wrong_image(self, name_jpg, path_file, width, height, depth, root):
        self.img_wrong['image_name'].append(name_jpg)
        self.img_wrong['path'].append(path_file)
        self.img_wrong['shape'].append("({},{},{})".format(int(width), int(height), int(depth)))
        labels = self.parse_labels(root)
        names = []
        for i in range(len(labels)):
            names.append(labels[i].get_name())

        labels = "/".join(map(str, names))
        self.img_wrong['labels'].append(labels)

    def check_sizes(self, width, height, depth):
        """
        Function for verifiy the size of the images
        :param width (int):
        :param height (int):
        :param depth (int):
        :return: boolean
        """
        if width == 1280 and height == 720 and depth == 3:
            return True
        return False

    def parse_labels(self, root):
        """
        Function that parse all the labels of the xml file
        :param root:
        :return:
        """
        objects = root.findall('object')
        array_labels = []
        for i in range(len(objects)):
            name = objects[i].find('name').text
            xmin = int(objects[i].find('bndbox').find('xmin').text)
            ymin = int(objects[i].find('bndbox').find('ymin').text)
            xmax = int(objects[i].find('bndbox').find('xmax').text)
            ymax = int(objects[i].find('bndbox').find('ymax').text)
            label = Label(name,xmin,ymin,xmax,ymax)
            array_labels.append(label)

        return array_labels

    def load_images(self, batch_size):

        image_list = [bch.path_file for i, bch in enumerate(self.annotations) if i < batch_size]

        return image_list




    def convert_labels_to_yolo(self):
        train_file = open(os.path.join(os.path.expanduser('~'), 'outAutelData', 'train_yolo','train.txt'),'w')
        #train_file_darknet = open(os.path.join(os.path.expanduser('~'),'')
        for ann in self.annotations:
            self.convert_annotation(ann)
            train_file.write('{}\n'.format(ann.path_file))

    def convert_annotation(self,ann):

        classes = self.load_classes()
        out_file = open(os.path.join(os.path.expanduser('~'), 'outAutelData', 'labels_yolo','{}.txt'.format(ann.file_name.split('.')[0])),'w')

        for label in ann.labels:
            if label.label_name not in classes:
                continue

            idx_class = classes.index(label.label_name)

            b = (float(label.xmin), float(label.xmax), float(label.ymin), float(label.ymax))
            bb = self.convert_yolo_sizes((float(ann.width),float(ann.height)),b)
            b_draw = (int(label.xmin), int(label.xmax), int(label.ymin), int(label.ymax))
            #self.print_bboxes(b_draw,bb,label.label_name,ann)
            out_file.write(str(idx_class) + " " + " ".join([str(a) for a in bb]) + '\n')



    def convert_yolo_sizes(self,size,box):

        dw = 1. / (size[0])
        dh = 1. / (size[1])
        x = (box[0] + box[1]) / 2.0 - 1
        y = (box[2] + box[3]) / 2.0 - 1
        w = box[1] - box[0]
        h = box[3] - box[2]
        x = x * dw
        w = w * dw
        y = y * dh
        h = h * dh
        return (x, y, w, h)

    def load_classes(self):
        name = os.path.join(self.ROOT_DIR, "autel.names")
        fp = open(name, "r")
        names = fp.read().split("\n")[:-1]
        return names

    def print_bboxes(self,b,bb,name,ann):
        font = cv2.FONT_HERSHEY_TRIPLEX
        font_scale = 0.75
        line_type = 1
        #b[0] xmin b[1] xmax
        #b[2] ymin b[3] ymax
        img = cv2.imread(ann.path_file)
        font_color = self.set_color(name)
        bbox = cv2.rectangle(img,(b[0],b[2]),(b[1],b[3]),font_color,2)
        newimg = cv2.putText(bbox, name, (b[0] - 5, b[2] - 5),font,
                                        font_scale, font_color, line_type)

        k = cv2.waitKey(1)

        if k == 27:  # If escape was pressed exit
            cv2.destroyAllWindows()
            sys.exit()

        cv2.imshow('random', newimg)

    def set_color(self,name):
        font_color = (0, 0, 0)
        if name == 'Car':
            font_color = (255, 0, 0)
        elif name == 'Person':
            font_color = (0, 0, 255)
        elif name == 'Vehicle':
            font_color = (255, 255, 255)
        elif name == 'Rider':
            font_color = (204, 204, 0)
        elif name == 'Animal':
            font_color = (255, 140, 0)
        elif name == 'Boat':
            font_color = (0, 140, 255)

        else:
            print("class not found")

        return font_color
        '''
        for i in (range(len(image.classes))):
            font_color = setColor(image.classes[i].name)
            bbox = cv2.rectangle(image.data, (image.classes[i].xmin, image.classes[i].ymin),
                                 (image.classes[i].xmax, image.classes[i].ymax), font_color, 2)

            newimg = cv2.putText(bbox, image.classes[i].name, (image.classes[i].xmin - 5, image.classes[i].ymin - 5),
                                 font,
                                 fontScale, font_color, lineType)
        
        '''