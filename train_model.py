import os
import time
import colorama
from colorama import Fore, Back, Style
import random as r
import math
import pandas as pd
from gab_bot import gab_bot
import numpy as np
import pickle

from util import PLAYERS_PER_TEAM, MAP_WIDTH, MAP_HEIGHT, TURNS_UNTIL_DRAW, STATS_LOG, ENT_LOG
from util import Board, generate_teams, save_all, load_all, pad, make_matchups
from util import player_names, player_icons, stat_rev, stats
from util import damage_vals, range_vals, aoe_vals, survival_vals, support_vals, mobility_vals

from genetic_auto_battler import run_round

NUM_ROUNDS = 500
NUM_TEAMS = 200
MUTATION_RATE = 1.0
STAT_CHANGE_RATE = 0.15
NUM_MATCHES = 5
USE_RANDOM_ELIM = True

NAME = "v1"
MODEL_FILE = './model_saves/' + NAME

bot = gab_bot()
if(os.path.exists(MODEL_FILE)):
    bot.load(MODEL_FILE)

if(os.path.exists('./data/res_' + NAME)):
    print("Loading Data/Skipping Simulation")
    with open('./data/ss_' + NAME, 'rb') as ss_file:
        all_states = pickle.load(ss_file)
    with open('./data/res_' + NAME, 'rb') as res_file:
        results = pickle.load(res_file)
    with open('./data/lens_' + NAME, 'rb') as lens_file:
        time_points = pickle.load(lens_file)
else:
    print("Running First Round...")
    teams = generate_teams(NUM_TEAMS)
    save_all(teams, './starting_teams.txt')
    all_states, results, time_points = [], [], []
    for i in range(NUM_ROUNDS):
        teams, wins, r_states, r_results, r_round_nums = run_round(teams, i, print_stats=True, train_bot=True, in_game_states=False)
        all_states.extend(r_states)
        results.extend(r_results)
        time_points.extend(r_round_nums)

    with open('./data/ss_' + NAME, 'wb') as ss_file:
        pickle.dump(all_states, ss_file)
    with open('./data/res_' + NAME, 'wb') as res_file:
        pickle.dump(results, res_file)
    with open('./data/lens_' + NAME, 'wb') as lens_file:
        pickle.dump(time_points, lens_file)

    save_all(teams, './final_teams.txt')

bot.enter_data(all_states, results, time_points)
bot.train_model()
bot.save(MODEL_FILE)
