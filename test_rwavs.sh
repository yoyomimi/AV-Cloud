#!/bin/bash
# test avcloud full
for i in {1..13}
do
    CUDA_VISIBLE_DEVICES=0 python -W ignore tools/test_rwavs.py --cfg configs/rwavs.yaml  output_dir avcloud_full_rwavs_${i}_22050  dataset.N_points -1 dataset.video _${i} model.file avcloud model.model_type full model.render_type simple model.resume_path logs/avcloud_full_rwavs_${i}_22050/avcloud_full_rwavs_${i}_22050/100.pth
done
python tools/av_metrics.py --log-dir avcloud_full_rwavs

# # test sh
# for i in {1..13}
# do
#    CUDA_VISIBLE_DEVICES=0 python -W ignore tools/test_rwavs.py --cfg configs/rwavs.yaml  output_dir avcloud_sh_rwavs_${i}_22050  dataset.N_points -1 dataset.video _${i} model.file avcloud model.model_type sh model.render_type simple model.resume_path logs/avcloud_sh_rwavs_${i}_22050/avcloud_sh_rwavs_${i}_22050/100.pth
# done
# python tools/av_metrics.py --log-dir avcloud_sh_rwavs

# # test simple-sh
# for i in {1..13}
# do
#     CUDA_VISIBLE_DEVICES=0 python -W ignore tools/test_rwavs.py --cfg configs/rwavs.yaml  output_dir avcloud_simplesh_rwavs_${i}_22050  dataset.N_points -1 dataset.video _${i} model.file avcloud model.model_type simple-sh model.render_type simple model.resume_path logs/avcloud_simplesh_rwavs_${i}_22050/avcloud_simplesh_rwavs_${i}_22050/100.pth
# done
# python tools/av_metrics.py --log-dir avcloud_simplesh_rwavs