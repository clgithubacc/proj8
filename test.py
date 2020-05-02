import argparse, sys
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile
import os
import numpy as np
import json
import collections
from PIL import Image


def process_points_to_np(klist,convert_to_local=True):
    new_list=[]
    for i in range(len(klist)):
        if i%3==0: #x
            if convert_to_local:
                if klist[i]==0:
                    new_list.append(klist[i])
                else:
                    new_list.append(klist[i]-klist[0])
            else:
                new_list.append(klist[i])
        elif i%3 ==1:
            if convert_to_local:
                if klist[i]==0:
                    new_list.append(klist[i])
                else:
                    new_list.append(klist[i]-klist[1])
            else:
                new_list.append(klist[i])
        else:
            continue
    return np.array(new_list)


def process_one_file(fpath,json_dir,temporal_dimension=10):
    fname=os.path.basename(fpath)
    fname=fname[:fname.rfind('.')]
    print(fname,json_dir)
    json_fname = [f for f in listdir(json_dir) if isfile(os.path.join(json_dir, f)) and f.startswith(fname)]
    #print(json_fname)
    if len(json_fname)==0:
        print('Warning: No json file found for '+fpath)
        return None
    d={}
    for o in json_fname:
        bar_pos=o.rfind('_')
        d[o]=int(o[bar_pos-12:bar_pos])
    ordered_json_fname=[k for k, v in sorted(d.items(), key=lambda item: item[1])]
    keypoints=[]
    last_skeleton=None
    for j in ordered_json_fname:
        jpath=os.path.join(json_dir, j)
        with open(jpath) as f:
            data = json.load(f)
        if len(data['people'])==0:
            keypoints.append(process_points_to_np([-999]*75))
            last_skeleton=process_points_to_np([-999]*75,convert_to_local=False)
        elif len(data['people'])==1:
            keypoints.append(process_points_to_np(data['people'][0]['pose_keypoints_2d']))
            last_skeleton=process_points_to_np(data['people'][0]['pose_keypoints_2d'],convert_to_local=False)
        else:
            if len(keypoints)==0:
                #If first frame, pick the largest object
                maxi=-1
                maxv=-1
                for k in range(len(data['people'])):
                    pts=process_points_to_np(data['people'][k]['pose_keypoints_2d'])
                    average_square=0
                    nonzero_count=0
                    for epts in pts:
                        if epts!=0:
                            average_square+=(epts*epts)
                            nonzero_count+=1
                    average_square=average_square/nonzero_count
                    if average_square>maxv:
                        maxv=average_square
                        maxi=k
                keypoints.append(process_points_to_np(data['people'][maxi]['pose_keypoints_2d']))
                last_skeleton=process_points_to_np(data['people'][maxi]['pose_keypoints_2d'],convert_to_local=False)
            else:
                minv=np.inf
                mini=0
                for k in range(len(data['people'])):
                    pts=process_points_to_np(data['people'][k]['pose_keypoints_2d'],convert_to_local=False)
                    square_diff=np.square(pts-last_skeleton).sum()
                    if square_diff<minv:
                        minv=square_diff
                        mini=k
                keypoints.append(process_points_to_np(data['people'][mini]['pose_keypoints_2d']))
                last_skeleton=process_points_to_np(data['people'][mini]['pose_keypoints_2d'],convert_to_local=False)

    point_buffer=collections.deque(maxlen=temporal_dimension)
    out_arr=None
    for i in range(temporal_dimension):
        point_buffer.append(keypoints[0])
    for i in range(len(keypoints)):
        point_buffer.append(keypoints[i])
        if out_arr is None:
            out_arr=np.expand_dims(np.stack(point_buffer,axis=0), axis=0)
        else:
            current_arr=np.expand_dims(np.stack(point_buffer,axis=0), axis=0)
            out_arr=np.concatenate([out_arr,current_arr],axis=0)
    return out_arr


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument('--videopath_name', help='Path to the directory for video and OpenPose jsons (See README)')
    args=parser.parse_args()
    print('Using '+args.videopath_name)

    temporal_dim=20
    test_path=args.videopath_name
    model = tf.keras.models.load_model('proj8')
    test_fname = [f for f in listdir(test_path) if isfile(os.path.join(test_path, f))]
    if not os.path.isdir(os.path.join(test_path, 'results')):
        os.mkdir(os.path.join(test_path, 'results'))
    is_single_file = len(test_fname) == 1
    for f in test_fname:
        Xtest = process_one_file(os.path.join(test_path, f), os.path.join(test_path, 'jsons'), temporal_dim)
        if is_single_file:
            pltname = 'timeLabel.png'
            jsonname = 'timeLabel.json'
        else:
            pltname = os.path.join(test_path, 'results/' + f + '.png')
            jsonname = os.path.join(test_path, 'results/' + f + '.json')
        y_pred = model.predict(Xtest)
        plt.figure(figsize=(10, 5))
        plt.ylim(0.0, 1.1)
        plt.plot(y_pred[:, 1])
        plt.title(f)
        plt.savefig(pltname)
        # json
        json_out = {}
        result_list = []
        for i in range(y_pred.shape[0]):
            result_list.append([(i / 30), y_pred[i, 0]])
        json_out["punch"] = result_list
        with open(jsonname, 'w+') as outfile:
            json.dump(str(json_out), outfile)
        if is_single_file:
            Image.open('timeLabel.png').convert("RGB").save('timeLabel.jpg', 'JPEG')
    print('Finished: for single video file, result will be saved in current directory. For multiple videos, result will'
          'be saved in the input video path folder, under results folder')