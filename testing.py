from __future__ import division
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2
from util import *
import argparse
import os
import os.path as osp
from darknet import Darknet
import pickle as pkl
import pandas as pd
import random


def arg_parse():
    """
    Parse arguements to the detect module

    """

    parser = argparse.ArgumentParser(description='YOLO v3 Detection Module')

    parser.add_argument("--images", dest='images', help=
    "Image / Directory containing images to perform detection upon",
                        default="/home/alupotto/Desktop/testing", type=str)
    parser.add_argument("--det", dest='det', help=
    "Image / Directory to store detections to",
                        default="/home/alupotto/Desktop/out_testing", type=str)
    parser.add_argument("--bs", dest="bs", help="Batch size", default=1)
    parser.add_argument("--confidence", dest="confidence", help="Object Confidence to filter predictions", default=0.5)
    parser.add_argument("--nms_thresh", dest="nms_thresh", help="NMS Threshhold", default=0.4)
    parser.add_argument("--cfg", dest='cfgfile', help=
    "Config file",
                        default="cfg/yolov3.cfg", type=str)
    parser.add_argument("--weights", dest='weightsfile', help=
    "weightsfile",
                        default="yolov3.weights", type=str)
    parser.add_argument("--reso", dest='reso', help=
    "Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
                        default="416", type=str)

    return parser.parse_args()


if __name__ == '__main__':


    ###########################################
    #         LOADING ARGS                   ##
    ###########################################

    args = arg_parse()

    #arguments default
    images = args.images
    batch_size = int(args.bs)
    confidence = float(args.confidence)
    nms_thesh = float(args.nms_thresh)
    start = 0
    CUDA = torch.cuda.is_available()

    #coco names
    num_classes = 80 #For COCO
    classes = load_classes("data/coco.names")

    ###########################################
    #         LOADING THE NETWORK            ##
    ###########################################

    #Set up the neural network
    print("Loading network......")
    model = Darknet(args.cfgfile)
    model.load_weights(args.weightsfile)
    print("Network successfully loaded")

    #Setup the heights of the images
    model.net_info["height"] = args.reso
    inp_dim = int(model.net_info["height"])
    assert inp_dim % 32 == 0
    assert inp_dim > 32

    # If there's a GPU availible, put the model on GPU
    if CUDA:
        model.cuda()


    #Set the model in evaluation mode
    model.eval()

    ###########################################
    #      READ THE INPUT IMAGES              #
    ###########################################

    read_dir = time.time()
    # Detection phase
    try:
        imlist = [osp.join(osp.realpath('.'), images, img) for img in os.listdir(images)]
    except NotADirectoryError:
        imlist = []
        imlist.append(osp.join(osp.realpath('.'), images))
    except FileNotFoundError:
        print("No file or directory with the name {}".format(images))
        exit()

    if not os.path.exists(args.det):
        os.makedirs(args.det)

    load_batch = time.time()
    loaded_ims = [cv2.imread(x) for x in imlist]

    # PyTorch Variables for images
    im_batches = list(map(prep_image, loaded_ims, [inp_dim for x in range(len(imlist))]))

    # List containing dimensions of original images
    im_dim_list = [(x.shape[1], x.shape[0]) for x in loaded_ims]
    im_dim_list = torch.FloatTensor(im_dim_list).repeat(1, 2)


    if CUDA:
        im_dim_list = im_dim_list.cuda()

    ###########################################
    #      CREATE THE BATCHES                 #
    ###########################################

    leftover = 0
    if (len(im_dim_list) % batch_size):
        leftover = 1

    if batch_size != 1:
        num_batches = len(imlist) // batch_size + leftover
        im_batches = [torch.cat((im_batches[i * batch_size: min((i + 1) * batch_size,
                                                                len(im_batches))])) for i in range(num_batches)]

    ###########################################
    #      DETECTION LOOP                     #
    ###########################################
    write = 0
    start_det_loop = time.time()
    for i, batch in enumerate(im_batches):
        # load the image
        start = time.time()
        if CUDA:
            batch = batch.cuda()
        with torch.no_grad():
            prediction = model(Variable(batch), CUDA)

        prediction = write_results(prediction, confidence, num_classes, nms_conf=nms_thesh)

        end = time.time()
        print("CHECKPOINT 1")
        if type(prediction) == int:
            print("CHECKPOINT 2")
            for im_num, image in enumerate(imlist[i * batch_size: min((i + 1) * batch_size, len(imlist))]):
                print("CHECKPOINT 3")
                im_id = i * batch_size + im_num
                print("{0:20s} predicted in {1:6.3f} seconds".format(image.split("/")[-1], (end - start) / batch_size))
                print("{0:20s} {1:s}".format("Objects Detected:", ""))
                print("----------------------------------------------------------")
            continue

        prediction[:, 0] += i * batch_size  # transform the atribute from index in batch to index in imlist

        if not write:  # If we have't initialised output
            output = prediction
            write = 1
        else:
            output = torch.cat((output, prediction))

        for im_num, image in enumerate(imlist[i * batch_size: min((i + 1) * batch_size, len(imlist))]):
            im_id = i * batch_size + im_num
            objs = [classes[int(x[-1])] for x in output if int(x[0]) == im_id]
            print("{0:20s} predicted in {1:6.3f} seconds".format(image.split("/")[-1], (end - start) / batch_size))
            print("{0:20s} {1:s}".format("Objects Detected:", " ".join(objs)))
            print("----------------------------------------------------------")

        if CUDA:
            torch.cuda.synchronize()
