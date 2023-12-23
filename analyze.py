import ast, glob, json
#import matplotlib.pyplot as plt
import numpy as np

from collections import Counter
from rich import print
from typing import Dict

data_path = "./logs/experiments/"
lan = "python"
env = "ic_" + lan
model = "gpt-3.5-turbo"
n_turn = "_5_turns"
suffix = ".json"
policy = "_multiturn_" + model + n_turn
template = "_function"
template0 = "_pythonCOT0"
templateF = "_pythonCOTF"
templater = "_function_r"
template0r = "_pythonCOT0_r"
templateFr = "_pythonCOTF_r"
def multi_turn_analysis(data):
    reward, turns_taken, err_percent, total = 0, 0, 0, 0
    for key, value in data.items():
#         reward += value['summary']['max_reward']
        reward += 1 if value["summary"]['max_reward'] == 1 else 0
        turns_taken += value['summary']['turns_taken']
        err_percent += value["turn_history"]["valid_action"].count(False)*1./len(value["turn_history"]["valid_action"])
        total += 1
    print(f'Reward: {reward * 1. / total}')
    print(f'Turns Taken: {turns_taken * 1. / total}')

def pands_analysis(data):
    reward, plan_len, err_percent, total = 0, 0, 0, 0
    for key, value in data.items():
        reward += 1 if value["summary"]['max_reward'] == 1 else 0
        plan_len += len(value['turn_history']['actions'])
        err_percent += value["turn_history"]["valid_action"].count(False)*1./len(value["turn_history"]["valid_action"])
        total += 1
    print(f'Reward: {reward * 1. / total}')
    print(f'Plan Length: {plan_len * 1. / total}')
    print(f'Error Rate: {err_percent * 1. / total}')

def react_analysis(data):
    reward, turns_taken, err_percent, total = 0, 0, 0, 0
    for key, value in data.items():
        reward += 1 if value["summary"]['max_reward'] == 1 else 0
        turns_taken += value['summary']['turns_taken']
        err_percent += value["turn_history"]["valid_action"].count(False)*1./len(value["turn_history"]["valid_action"])
        total += 1
    print(f'Reward: {reward * 1. / total}')
    print(f'Turns Taken: {turns_taken * 1. / total}')
    print(f'Error Rate: {err_percent * 1. / total}')

log_path = data_path+ env + policy + template + suffix
log_path0 = data_path+ env + policy + template0 + suffix
log_pathF = data_path+ env + policy + templateF + suffix
log_pathr = data_path+ env + policy + templater + suffix
log_path0r = data_path+ env + policy + template0r + suffix
log_pathFr = data_path+ env + policy + templateFr + suffix
log = json.load(open(log_path, "r"))
log0 = json.load(open(log_path0, "r"))
logF = json.load(open(log_pathF, "r"))
logr = json.load(open(log_pathr, "r"))
log0r = json.load(open(log_path0r, "r"))
logFr = json.load(open(log_pathFr, "r"))
multi_turn_analysis(log)
multi_turn_analysis(log0)
multi_turn_analysis(logF)
multi_turn_analysis(logr)
multi_turn_analysis(log0r)
multi_turn_analysis(logFr)
logbash = data_path+"ic_bash_multiturn_gpt-3.5-turbo_5_turns_v2.json"
logbashr = data_path+"ic_bash_multiturn_gpt-3.5-turbo_5_turns_v2r.json"
logbash = json.load(open(logbash, "r"))
logbashr = json.load(open(logbashr, "r"))
multi_turn_analysis(logbash)
multi_turn_analysis(logbashr)

data_path2 = "./data/results"
log_bashps = data_path2+"/bash/gpt-3.5/ic_bash_plan_solve_fs_1.json"
log_bashre = data_path2+"/bash/gpt-3.5/ic_bash_react_10_turns_fs_1.json"
log_bashps = json.load(open(log_bashps, "r"))
log_bashre = json.load(open(log_bashre, "r"))
pands_analysis(log_bashps)
react_analysis(log_bashre)