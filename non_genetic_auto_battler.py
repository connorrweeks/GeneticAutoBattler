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
from util import Board, generate_teams, save_all, load_all, pad, make_matchups, create_stats, ent_str
from util import player_names, player_icons, stat_rev, stats
from util import damage_vals, range_vals, aoe_vals, survival_vals, support_vals, mobility_vals

from AI_team_builder import team_builder

NUM_MATCHES = 5
NUM_TEAMS = 200
NUM_ROUNDS = 500

def main():
    os.system('clear')
    current_teams = generate_teams(20)
    for i in range(20):
        current_teams = run_round(current_teams, i)
        save_all(current_teams, f"./my_teams/non_gen_teams_{i}.txt")

def run_round(teams, num):
    t0 = time.perf_counter()
    matchups = make_matchups(len(teams), NUM_MATCHES)

    new_teams = []
    for i in range(len(teams)):
        print(f"\rUpdating team {i}/{len(teams)}", end="")
        opponent_ids = [x[0] if x[1] == i else x[1] for x in matchups if x[0] == i or x[1] == i]

        tb = team_builder("net", teams[i], opponents=[teams[x] for x in opponent_ids])
        tb.improve_player(r.randrange(PLAYERS_PER_TEAM))
        counter_team = tb.current_team
        new_teams.append(counter_team)
    print(f"\rUpdating team {len(teams)}/{len(teams)}")

    scores = [0] * len(teams)
    for x1, x2 in matchups:
        b = Board(teams[x1], teams[x2])
        res = b.resolve(False)

        scores[x1] += 1 - res
        scores[x2] += res

    os.system('clear')
    print(f"Round #{num}")
    t1 = time.perf_counter()
    print(f"Round Time: {t1-t0:.2f}")

    entropy = create_stats(teams, scores, True)
    return new_teams

if(__name__ == "__main__"):
    main()
