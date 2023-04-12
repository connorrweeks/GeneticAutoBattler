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

VERSION_NUMBER = 'v7.0'
MUTATION_RATE = 0.3
STAT_CHANGE_RATE = 0.2
NUM_MATCHES = 5
USE_RANDOM_ELIM = True
NUM_TEAMS = 100
NUM_ROUNDS = 200

RUN_MODE = 'run_rounds'

MODEL_FILE = './model_saves/team_improver'
#Micro-Fighters

r.seed(7)

def main():
    os.system('clear')

    if(RUN_MODE == 'test_metas'):
        test_random_metas(num=40)
    elif(RUN_MODE == 'battle_mode'):
        my_team = []
        while(len(my_team) == 0):
            team_name = input("Input your team's name:")
            file_name = f"./my_teams/{team_name}.txt"
            if(not os.path.exists(file_name)):
                print("Team Not Found!")
                continue
            else:
                my_team = load_all(file_name)[0]
        teams = generate_teams(1000)
        wins = 0
        for i in range(1000):
            b = Board(my_team, teams[i])
            wins += 1 if b.resolve(True, game_speed=0.5) == 0 else 0
        print(f"Winrate:{wins/1000}")
    elif(RUN_MODE == 'train_model'):
        bot = gab_bot()
        if(os.path.exists(MODEL_FILE)):
            bot.load(MODEL_FILE)

        if(os.path.exists('./data/res')):
            print("Loading Data/Skipping Simulation")
            with open('./data/ss', 'rb') as ss_file:
                all_states = pickle.load(ss_file)
            with open('./data/res', 'rb') as res_file:
                results = pickle.load(res_file)
            with open('./data/lens', 'rb') as lens_file:
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

            with open('./data/ss', 'wb') as ss_file:
                pickle.dump(all_states, ss_file)
            with open('./data/res', 'wb') as res_file:
                pickle.dump(results, res_file)
            with open('./data/lens', 'wb') as lens_file:
                pickle.dump(time_points, lens_file)

            save_all(teams, './final_teams.txt')

        bot.enter_data(all_states, results, time_points)
        bot.train_model()
        bot.save(MODEL_FILE)

    elif(RUN_MODE == 'run_rounds'):
        print("Running First Round...")
        teams = generate_teams(NUM_TEAMS)
        save_all(teams, './starting_teams.txt')
        for i in range(NUM_ROUNDS):
            teams, wins = run_round(teams, i, print_stats=True)
            #print(wins)
        #save_all(teams, f'./teams_{i}.txt')
        save_all(teams, './final_teams.txt')

    elif(RUN_MODE == 'single_battle'):
        teams = load_all('./final_teams.txt')
        for i in range(NUM_ROUNDS):
            match_up = r.sample(teams, 2)
            b = Board(match_up[0], match_up[1])
            b.resolve(True)

    elif(RUN_MODE == 'multi'):
        #teams = load_all('./final_teams.txt')
        teams = generate_teams(100)
        multi_display(teams, height=1, width=3, num=NUM_ROUNDS, speed=1, against_random=False)
    else:
        print("invalid", RUN_MODE)
        exit()

def multi_display(teams, height, width, num, speed, against_random=False):
    random_teams = generate_teams(len(teams))
    boards = []
    for i in range(width * height):
        match_up = r.sample(teams, 2)
        if(against_random): match_up[1] = r.sample(random_teams, 1)[0]
        boards.append(Board(match_up[0], match_up[1]))

    total_turns = [0 for _ in boards]
    finished_games = 0
    delay = 3
    while(finished_games < num):
        states = []
        for i, b in enumerate(boards):
            if(total_turns[i] < delay):
                total_turns[i] += 1
                states.append(b.print(show=False))
                continue
            if(b.state == None):
                states.append(b.print(show=False))
                total_turns[i] += 1
                if(total_turns[i] - delay > TURNS_UNTIL_DRAW): b.state = 2
                result = b.take_turn()
            else:
                finished_games += 1
                states.append(b.print(show=False))
                match_up = r.sample(teams, 2)
                if(against_random): match_up[1] = r.sample(random_teams, 1)[0]
                boards[i] = Board(match_up[0], match_up[1])
                total_turns[i] = 0

        states = [x.strip().split('\n') for x in states]

        all_print = f"Finished Games: {finished_games}\n"
        all_print += "-"*((MAP_WIDTH*width)+width+1) + '\n'
        for row in range(height):
            for y in range(MAP_HEIGHT*2):
                my_print_row = '|'.join([x[y] for x in states[row*width:(row+1)*width]])
                all_print += '|' + my_print_row + "|\n"
            all_print += ("-"*((MAP_WIDTH*width)+width+1)) + "\n"

        all_print += '\n'
        line1, line2 = '', ''
        i = 0
        for k in player_names:
            line1 += Fore.BLUE + player_icons[k] + Fore.RESET + " " + pad(player_names[k], 15) + '  '
            line2 += pad('/'.join([stat_rev[j] for j, val in enumerate(k) if val > 0]), 17) + '  '
            i += 1
            if(i == 5):
                i = 0
                all_print += line1 + '\n' + line2 + '\n\n'
                line1, line2 = '', ''

        os.system('clear')
        print(all_print)
        time.sleep(speed)


def test_random_metas(num=10):
    global ENT_LOG
    global damage_vals
    global range_vals
    global aoe_vals
    global survival_vals
    global support_vals
    global mobility_vals
    global STATS_LOG

    round_num = 100

    t0 = time.perf_counter()
    for j in range(num):
        #damage_vals = [20, 40, 70]
        #range_vals = [1.5, 3.5, 8.5]
        #aoe_vals = [0, 2.5, 6.5]
        #survival_vals = [200, 300, 500]
        #support_vals = [0, 20, 40]
        #mobility_vals = [1.2, 2.5, 4.5]

        damage_vals = r.choice([[20, 40, 70]])
        range_vals = r.choice([[1.5, 4.5, 7.5], [1.5, 2.5, 3.5], [1.5, 3.5, 5.5]])
        aoe_vals = r.choice([[0, 1.5, 3.5], [0, 2.5, 4.5]])
        survival_vals = r.choice([[10, 20, 30], [5, 10, 15], [10, 15, 20], [10, 15, 30]])
        support_vals = r.choice([[0, 4, 8], [0, 2, 4], [0, 3, 8], [0, 5, 10]])
        mobility_vals = r.choice([[1.5, 2.5, 4.5], [1.5, 3.5, 6.5], [1.5, 5.5, 10.5]])

        teams = generate_teams(100)
        for i in range(round_num):
            teams, wins = run_round(teams, i, print_stats=False)
            t1 = time.perf_counter()

            remaining = (round_num * (num - j - 1)) + round_num - i - 1
            eta = (t1 - t0) * remaining / (j * round_num + i + 1)
            print(f"\rRunning sim #{j} round:{i+1}/{100} current:{ent_str(ENT_LOG[-1])} score:{ENT_LOG[-1]:.2f} eta:{eta:.2f}" + ' ' * 5, end="")
        print()

        f = open('./meta_log.txt', 'a+')
        save_all(teams, f'./team_saves/sim_{j}.txt')

        ENT_LOG = ENT_LOG[-50:]
        ent_avg = sum(ENT_LOG) / len(ENT_LOG)
        f.write(f"{j} ent:{ent_str(ent_avg)} ent_val:{ent_avg} dmg:{damage_vals} rng:{range_vals} aoe:{aoe_vals} sur:{survival_vals} sup:{support_vals} mob:{mobility_vals}\n")
        f.close()
        ENT_LOG = []
        STATS_LOG = []



def run_round(teams, num=0, print_stats=False, train_bot=False, in_game_states=False):
    t0 = time.perf_counter()
    wins = [0] * len(teams)

    assert(train_bot == True or in_game_states == False)

    all_states, all_results, round_nums = [], [], []
    for team1_id, team2_id in make_matchups(len(teams), NUM_MATCHES):
        b = Board(teams[team1_id], teams[team2_id])
        if(train_bot): starting_state = b.get_state()
        if(train_bot and in_game_states):
            result, game_states = b.resolve(get_states=True)

            all_states.extend(game_states)
            all_results.extend([result] * len(game_states))
            round_nums.extend(list(range(1, len(game_states) + 1)))
        else:
            result = b.resolve()

        if(train_bot):
            all_states.append(starting_state)
            all_results.append(result)
            round_nums.append(0)

        if(result == -1):
            wins[team1_id] += 0.5
            wins[team2_id] += 0.5
        if(result == 0): wins[team1_id] += 1
        if(result == 1): wins[team2_id] += 1

    os.system('clear')
    if(print_stats): print("Version:", VERSION_NUMBER)
    if(print_stats): print(f"Round #{num}")
    t1 = time.perf_counter()
    if(print_stats): print(f"Round Time: {t1-t0:.2f}")

    entropy = create_stats(teams, wins, print_stats)

    teams = mutate_teams(teams, wins, print_stats)

    if(train_bot):
        return teams, wins, all_states, all_results, round_nums
    else:
        return teams, wins

def mutate_teams(teams, wins, print_stats):
    if(print_stats): print("Ranking Teams...")
    combined = list(zip(teams, wins))
    r.shuffle(combined)
    ind = list(range(len(teams)))
    sorted_ind = sorted(ind, key=lambda i: wins[i], reverse=False)

    if(print_stats): print("Mutating New Teams...")

    teams = teams[:]
    for i in sorted_ind[:int(len(teams)/2)]:
        if(USE_RANDOM_ELIM):
            if(r.random() > i / int(len(teams)/2)): teams[i] = mutate(teams[-(i+1)])
        else:
            teams[i] = mutate(teams[-(i+1)])
    r.shuffle(teams)
    return teams

def mutate(team):
    team = team[:]
    for i, p in enumerate(team):
        stats, x_pos, y_pos = p
        if(r.random() > MUTATION_RATE): continue
        if(r.random() < STAT_CHANGE_RATE):
            #Change stats
            current_stats = []
            for j in range(6):
                if(stats[j] == 1): current_stats.append(j)
                if(stats[j] == 2): current_stats = [j,j]
            keep = r.choice(current_stats)
            other = r.randrange(6)
            new_stats = [0] * 6
            new_stats[keep] += 1
            new_stats[other] += 1
            team[i] = (tuple(new_stats),p[1],p[2],)
        else:
            #Change location
            new_x = min(max(x_pos + r.randrange(-1,2), 0), MAP_WIDTH-1)
            new_y = min(max(y_pos + r.randrange(-1,2), 0), MAP_HEIGHT-1)
            if(not any([x[1] == new_x and x[2] == new_y for x in team])):
                team[i] = (p[0],new_x,new_y,)
    return team

if(__name__ == "__main__"):
    main()
