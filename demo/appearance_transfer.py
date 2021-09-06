# Copyright (c) 2020-2021 impersonator.org authors (Wen Liu and Zhixin Piao). All rights reserved.

import os
import os.path as osp
import platform
import argparse
import subprocess
import sys
import time
import gc

from iPERCore.services.options.options_setup import setup
from iPERCore.services.run_imitator import run_imitator
from iPERCore.services.run_swapper import run_swapper

###############################################################################################
##                   Setting
###############################################################################################
parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", type=str, default="Zalando_256_192", help="[Deepfashion_256_192 | Zalando_256_192 | Zalora_256_192")
parser.add_argument("--dataset_range", type=str, default="0:582", help="[start_id:end:id)")
parser.add_argument("--gpu_ids", type=str, default="0", help="the gpu ids.")
parser.add_argument("--image_size", type=int, default=256, help="the image size.")
parser.add_argument("--num_source", type=int, default=2, help="the number of sources.")
parser.add_argument("--output_dir", type=str, default="/mnt/date/zhaofuwei/Projects-warehouse/iPERCore/results_tryon_808", help="the output directory.")
parser.add_argument("--assets_dir", type=str, default="./assets",
                    help="the assets directory. This is very important, and there are the configurations and "
                         "the all pre-trained checkpoints")

src_path_format = """the source input information. 

    All source paths and it supports multiple paths, uses "|" as the separator between all paths. 
    The format is "src_path_1|src_path_2|src_path_3".
    
    Each src_input is "path?=path1,name?=name1,bg_path?=bg_path1,parts?=part1-part2". 
    
    It must contain 'path'. If 'name' and 'bg_path' are empty, they will be ignored.
   
    The 'path' could be an image path, a path of a directory contains source images, and a video path.
   
    The 'name' is the rename of this source input, if it is empty, we will ignore it, and use the filename of the path.
   
    The 'bg_path' is the actual background path if provided, otherwise we will ignore it.
    
    The `parts?` is the selected parts of this source input. Here, we use `-` as the separator among different parts.
    The valid part names are {head, torso, left_leg, right_leg, left_arm, right_arm, left_root, right_root, 
    left_hand, right_hand, facial, upper, lower, body, all},

    {
        "head": [0],
        "torso": [1],
        "left_leg": [2],
        "right_leg": [3],
        "left_arm": [4],
        "right_arm": [5],
        "left_foot": [6],
        "right_foot": [7],
        "left_hand": [8],
        "right_hand": [9],
        "facial": [10],
        "upper": [1, 2, 3, 8, 9],
        "lower": [4, 5, 6, 7],
        "body": [1, 2, 3, 4, 5, 6, 7, 8, 9],
        "all": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    }
    
    A whole body = {head, torso, left_leg, right_leg, left_arm, right_arm, left_root, right_root, left_hand, right_hand}.
    So, there are no over-lap regions among {head, torso, left_leg, right_leg, left_arm, right_arm, left_root, right_root, left_hand, right_hand},
    and the left parts might have over-lap regions with each other. 
    upper = {torso, left_arm, right_arm, left_hand, right_hand};
    lower = {left_leg, right_leg, left_foot, right_foot};
    body = upper + lower.

    Here we show some examples, "src_path_1|src_path_2", and "src_path_1" will be set as the primary source inputs.
    
    The synthesized background is come from the primary source inputs ("src_path_1").
    
    If all selected parts are not enough being a whole body = {head, torso, left_leg, right_leg, left_arm, right_arm, left_root, right_root, left_hand, right_hand}.
    The left parts will come from the primary source inputs ("src_path_1").

    1. "path?=path1,name?=name1,bg_path?=bg_path1,parts?=head|path?=path2,name?=name2,bg_path?=bg_path1,parts?=body"
    It means we take the head part from the "path1" and take the left body parts from "path2".

    2. "path?=path1,name?=name1,bg_path?=bg_path1,parts?=head-torso|path?=path2,name?=name2,bg_path?=bg_path1,parts?=left_arm-right_arm-left_hand-right_hand-left_leg-right_leg-left_foot-right_foot"
    It means we take the head and torso part from the "path1" and take the left parts from "path2".

    3. "path?=path1,name?=name1,bg_path?=bg_path1,parts?=head-torso|path?=path2,name?=name2,bg_path?=bg_path1,parts?=left_leg-right_leg-left_foot-right_foot"
    We take {head, torso} from "path1" and {left_leg, right_leg, left_foot, right_foot} from "path2",
    and the selected parts are {head, torso, left_leg, right_leg, left_foot, right_foot}, 
    and the left parts are {left_arm, right_arm, left_hand, right_hand} will be selected from the primary source inputs ("path1").
    Therefore the actual selected parts of "path1" are {head, torso, left_arm, right_arm, left_hand, right_hand}.

    4. "path?=path1,name?=name1,bg_path?=bg_path1,parts?=head-torso|path?=path2,name?=name2,bg_path?=bg_path1,parts?=upper|path?=path3,name?=name3,parts?=lower"
    There are 3 source inputs. 

"""

parser.add_argument("--model_id", type=str, default="person_a+person_b", help="the renamed model.")
parser.add_argument("--src_path", type=str, default = "path?=./tryon_data/a/fashionWOMENBlouses_Shirtsid0000092204_4full.jpg,name?=person_a,parts?=head-lower|path?=./tryon_data/b/fashionMENTees_Tanksid0000662101_4full.jpg,name?=person_b,parts?=upper", help=src_path_format)

ref_path_format = """the reference input information. All reference paths. It supports multiple paths,
    and uses "|" as the separator between all paths.
    The format is "ref_path_1|ref_path_2|ref_path_3".
    Each ref_path is "path?=path1,name?=name1,audio?=audio_path1,fps?=30,pose_fc?=300,cam_fc?=150".
    It must contain "path", and others could be empty, and they will be ignored.

    The "path" could be an image path, a path of a directory contains source images, and a video path.

    The "name" is the rename of this source input, if it is empty, we will ignore it,
    and use the filename of the path.

    The "audio" is the audio path, if it is empty, we will ignore it. If the "path" is a video,
    you can ignore this, and we will firstly extract the audio information of this video (if it has audio channel).

    The "fps" is fps of the final outputs, if it is empty, we will set it as the default fps 25.

    The "pose_fc" is the smooth factor of the temporal poses. The smaller of this value, the smoother of the
    temporal poses. If it is empty, we will set it as the default 300. In the most cases, using the default
    300 is enough, and if you find the poses of the outputs are not stable, you can decrease this value.
    Otherwise, if you find the poses of the outputs are over stable, you can increase this value.

    The "cam_fc" is the smooth factor of the temporal cameras (locations in the image space). The smaller of
    this value, the smoother of the locations in sequences. If it is empty, we will set it as the default 150.
    In the most cases, the default 150 is enough.

    1. "path?=path1,name?=name1,audio?=audio_path1,fps?=30,pose_fc?=300,cam_fc?=150|
        path?=path2,name?=name2,audio?=audio_path2,fps?=25,pose_fc?=450,cam_fc?=200",
        this input will be parsed as
        [{path: path1, name: name1, audio: audio_path1, fps: 30, pose_fc: 300, cam_fc: 150},
         {path: path2, name: name2, audio: audio_path2, fps: 25, pose_fc: 450, cam_fc: 200}]

    2. "path?=path1,name?=name1, pose_fc?=450|path?=path2,name?=name2", this input will be parsed as
    [{path: path1, name: name1, fps: 25, pose_fc: 450, cam_fc: 150},
     {path: path2, name: name2, fps: 25, pose_fc: 300, cam_fc: 150}].

    3. "path?=path1|path?=path2", this input will be parsed as
    [{path: path1, fps:25, pose_fc: 300, cam_fc: 150}, {path: path2, fps: 25, pose_fc: 300, cam_fc: 150}].

    4. "path1|path2", this input will be parsed as
    [{path: path1, fps:25, pose_fc: 300, cam_fc: 150}, {path: path2, fps: 25, pose_fc: 300, cam_fc: 150}].

    5. "path1", this will be parsed as [{path: path1, fps: 25, pose_fc: 300, cam_fc: 150}].
"""
parser.add_argument("--ref_path", type=str, default="path?=./tryon_data/a/fashionWOMENBlouses_Shirtsid0000092204_4full.jpg,name?=person_ref", help=ref_path_format)

# args = parser.parse_args()
args, extra_args = parser.parse_known_args()

# symlink from the actual assets directory to this current directory
work_asserts_dir = os.path.join("./assets")
if not os.path.exists(work_asserts_dir):
    os.symlink(osp.abspath(args.assets_dir), osp.abspath(work_asserts_dir),
               target_is_directory=(platform.system() == "Windows"))

args.cfg_path = osp.join(work_asserts_dir, "configs", "deploy.toml")

if __name__ == "__main__":
    # run swapper
    dataset_root = '/mnt/date/xiezhy/Datasets'
    # Deepfashion: 934, Zalando: 1746, Zalora: 4369
    # (934 + 1746 + 4369) * 3min / 60 / 24 = 19.67 ~= 20 days
    # Deepfashion划分为[0:467)[gpu0],[467,934)[gpu1]
    # Zalando划分为[0:582)[gpu2],[582,1164)[gpu3],[1164,1746)[gpu4]
    # Zalora划分为[0:728)[gpu5],[728:1456)[gpu0],[1456:2184)[gpu1],[2184:2912)[gpu2],[2912:3640)[gpu3],[3640:4369)[gpu5]
    # 总共划分了11份，其中最多的单份是728，需要用时 728*3min/60/24 ~= 1.5 days，因此预计两日内可以全部完成
    dataset_name = args.dataset_name
    dataset_range = args.dataset_range
    start_id, end_id = int(dataset_range.split(':')[0]), int(dataset_range.split(':')[1])
    test_lines = open(os.path.join(dataset_root, dataset_name,'test_pairs_front_list_shuffle_0508.txt'), 'r')

    output_root = args.output_dir
    lines = test_lines.readlines()[start_id:end_id]
    for i, test_line in enumerate(lines):
        i += start_id
        image_a_path = os.path.join(dataset_root, dataset_name, 'image', test_line.split(' ')[0])
        image_b_path = os.path.join(dataset_root, dataset_name, 'image', test_line.split(' ')[1][:-1])

        # output_dir = os.path.join(args.output_dir, dataset_name, image_a_path.split('/')[-1]+'+'+image_b_path.split('/')[-1])
        args.output_dir = os.path.join(output_root, dataset_name, f'test_pair_{i}')
        exist_check = os.path.join(args.output_dir,'primitives/person_a/synthesis/swappers/person_a+person_b-person_ref.mp4')
        if not os.path.exists(exist_check):
            os.makedirs(args.output_dir, exist_ok=True)
            args.src_path = 'path?=' + image_a_path + ',name?=person_a,parts?=head-lower' + '|' + 'path?=' + image_b_path + ',name?=person_b,parts?=upper'
            args.ref_path = 'path?=' + image_a_path + ',name?=person_ref'

            # cfg = setup(args)
            # all_meta_outputs = run_swapper(cfg)

            cmd = [
                sys.executable, "-m", "iPERCore.services.run_swapper",
                "--cfg_path", args.cfg_path,
                "--gpu_ids", args.gpu_ids,
                "--image_size", str(args.image_size),
                "--num_source", str(args.num_source),
                "--output_dir", args.output_dir,
                "--model_id", args.model_id,
                "--src_path", args.src_path,
                "--ref_path", args.ref_path
            ]

            cmd += extra_args
            subprocess.call(cmd)

        
        print('*-'*50 + '*')
        print(f'*- the {i}th test pair of ' + dataset_name + ' in range ' + '[' + dataset_range + ')' + ' has been processed sucessfully! -*')
        print('*-'*50 + '*')

    # or use the system call wrapper
    # cmd = [
    #     sys.executable, "-m", "iPERCore.services.run_swapper",
    #     "--cfg_path", args.cfg_path,
    #     "--gpu_ids", args.gpu_ids,
    #     "--image_size", str(args.image_size),
    #     "--num_source", str(args.num_source),
    #     "--output_dir", args.output_dir,
    #     "--model_id", args.model_id,
    #     "--src_path", args.src_path,
    #     "--ref_path", args.ref_path
    # ]

    # cmd += extra_args
    # subprocess.call(cmd)
