import os
import subprocess

root_dir = '/disk3/hyeon/FaceForensics_jpg'
target_dir = '/disk3/hyeon/FaceForensics_dist'
original_dir = '/disk3/hyeon/FaceForensics/manipulated_sequences'

video_dict = {}

for root, subdirs, files in os.walk(original_dir):
    if 'original_sequences' in root:
        continue
    for f in files:
        if f.endswith('.mp4'):
            video_dict[f.split('.')[0]] = root.split('/')[-3]

for root, subdirs, files in os.walk(root_dir):
    if len(files) > 0 and 'jpg' in files[0]:
        if 'manipulated' in root:
            target_path = os.path.join(target_dir,
                                       root.split('/')[-2],
                                       video_dict[root.split('/')[-1]])
            os.makedirs(target_path, exist_ok=True)
            cmd = 'cp -r {} {}'.format(root, target_path)
            print(cmd)
            subprocess.call(cmd, shell=True)
