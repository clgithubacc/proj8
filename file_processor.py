from os import listdir
from os.path import isfile
import numpy as np
import json
import collections
import os
def process_points_to_np(klist):
    new_list=[]
    for i in range(len(klist)):
        if i%3==0: #x
            if klist[i]==0:
                new_list.append(klist[i])
            else:
                new_list.append(klist[i]-klist[0])
        elif i%3 ==1:
            if klist[i]==0:
                new_list.append(klist[i])
            else:
                new_list.append(klist[i]-klist[1])
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
    d={}
    for o in json_fname:
        bar_pos=o.rfind('_')
        d[o]=int(o[bar_pos-12:bar_pos])
    ordered_json_fname=[k for k, v in sorted(d.items(), key=lambda item: item[1])]
    keypoints=[]
    for j in ordered_json_fname:
        jpath=os.path.join(json_dir, j)
        with open(jpath) as f:
            data = json.load(f)
        if len(data['people'])==0:
            keypoints.append(process_points_to_np([-999]*75))
        elif len(data['people'])==1:
            keypoints.append(process_points_to_np(data['people'][0]['pose_keypoints_2d']))
        else:
            if len(keypoints)==0:
                keypoints.append(process_points_to_np(data['people'][0]['pose_keypoints_2d']))
            else:
                minv=np.inf
                mini=0
                for k in range(len(data['people'])):
                    pts=process_points_to_np(data['people'][k]['pose_keypoints_2d'])
                    square_diff=np.square(pts-keypoints[-1]).sum()
                    if square_diff<minv:
                        minv=square_diff
                        mini=k
                keypoints.append(process_points_to_np(data['people'][mini]['pose_keypoints_2d']))

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