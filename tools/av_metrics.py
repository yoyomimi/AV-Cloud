import argparse
import json
import os
import pickle

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--best', action="store_true")
    parser.add_argument('--log-dir', type=str)

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()

    env_dict = {"office": list(range(1, 6)),
               "house": list(range(6, 9)),
               "apartment": list(range(9, 12)),
               "outdoor": list(range(12, 14))
               }

    print("===================================")
    result = {}
    avg_dict = {}
    for env_name, env_id in env_dict.items():
        avg_dict.setdefault(env_name, {})
        loss_list = []
        for i in env_id:
            log_path = f'av_results/{args.log_dir}_{i}_22050.pkl'
            try:
                metrics = pickle.load(open(log_path, "rb"))
                for k in metrics.keys():
                    if k in ['mag', 'lre', 'dpam', 'rte', 'env']:
                        avg_dict[env_name].setdefault(k, [])
                        avg_dict[env_name][k].append(np.array(metrics[k]).reshape(1)[0])
            except:
                print(f"{log_path} Not exists")
                metrics = {}
            
print(avg_dict)
for env in avg_dict.keys():
    print(env, end=" \n")
    for k in avg_dict[env].keys():
        result.setdefault(k, [])
        avg_dict[env][k] = sum(avg_dict[env][k]) / len(avg_dict[env][k])
        if k in ['mag', 'lre', 'dpam', 'rte', 'env']:
            print(k, round(avg_dict[env][k], 3), end=" \n")
        result[k].append(round(avg_dict[env][k], 3))
    print('\n')

print("overall average", end=" \n")
for k in result.keys():
    result[k] = sum(result[k]) / len(result[k])
    if k in ['mag', 'lre', 'dpam', 'rte', 'env']:
        print(k, round(result[k], 3), end=" \n")
print("===================================")