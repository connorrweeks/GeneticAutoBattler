import time
import colorama
from colorama import Fore, Back, Style
import random as r
import math
import pandas as pd
import numpy as np
import pickle
from pynput import keyboard
import os
import torch

from util import PLAYERS_PER_TEAM, MAP_WIDTH, MAP_HEIGHT, TURNS_UNTIL_DRAW
from util import Board, generate_teams, save_all, load_all, pad
from util import player_names, player_icons, stat_rev, stats
from util import damage_vals, range_vals, aoe_vals, survival_vals, support_vals, mobility_vals

from gab_bot import gab_bot

#tourney
#Final Score [0.8765]
#Final Score [0.833]
#Final Score [0.847]
#Final Score [0.8465]
#Final Score [0.9575]

#net
#Final Score [0.9885833085179329]
#Final Score [0.9856755268573761]
#Final Score [0.94890312987566]
#Final Score [0.9828879225254059]
#Final Score [0.9844957569241524]

def main():
    r.seed(0)

    previous_team = generate_teams(1)[0]
    for i in range(5):
        tb = team_builder("net", opponents=[previous_team])
        counter_team = tb.improve_team(5)

        tb = team_builder("tourney", opponents=[previous_team])
        scores = [tb.get_scores([counter_team], 1)[0] for i_ in range(10)]
        print("\nFinal Score", sum(scores) / len(scores))
        print(" ".join([player_names[x[0]] for x in counter_team]))

        #save_all([my_team], f'./my_teams/random_tourney.txt')
        save_all([counter_team], f'./my_teams/net_{i}.txt')
        previous_team = counter_team
    #print(my_team)



class team_builder():
    def __init__(self, optim, starting_team=None, skip_factor=0, opponents=None):
        self.optim = optim
        self.skip_factor = skip_factor

        if(self.optim == "net"):
            self.gb = gab_bot()
            #self.gb.load("./model_saves/team_builder")
            self.gb.load("./model_saves/v1")

        if(starting_team == None):
            starting_team = generate_teams(1)[0]
        self.current_team = [list(x) for x in starting_team]
        self.opponents = opponents

    def improve_player(self, i):
        p = self.current_team[i]
        tests = {}
        for player_type in player_names:
            if(self.skip_factor > 0 and r.random() < self.skip_factor): continue
            temp_team = [x[:] for x in self.current_team]
            temp_team[i][0] = player_type
            tests[player_type] = temp_team

        if(p[0] not in tests): tests[p[0]] = [x[:] for x in self.current_team]
        best_score, best_type = -1, None
        scores = self.get_scores([tests[x] for x in tests])
        for j, score in enumerate(scores):
            if(score > best_score):
                best_score = score
                best_type = list(tests.keys())[j]
        self.current_team[i][0] = best_type

        tests = {}
        pos_set = set([(x[1], x[2]) for x in self.current_team])
        for x_pos in range(MAP_WIDTH):
            for y_pos in range(MAP_HEIGHT):
                if(self.skip_factor > 0 and r.random() < self.skip_factor): continue
                if((x_pos, y_pos) in pos_set): continue
                temp_team = [x[:] for x in self.current_team]
                temp_team[i][1] = x_pos
                temp_team[i][2] = y_pos
                tests[(x_pos, y_pos)] = temp_team

        if((p[1], p[2]) not in tests): tests[(p[1], p[2])] = [x[:] for x in self.current_team]
        best_score, best_x, best_y = -1, None, None
        scores = self.get_scores([tests[x] for x in tests])
        for j, score in enumerate(scores):
            if(score > best_score):
                best_score = score
                best_x, best_y = list(tests.keys())[j]
        self.current_team[i][1] = best_x
        self.current_team[i][2] = best_y

    def improve_team(self, num_iters):
        for i_ in range(num_iters):
            for i, p in enumerate(self.current_team):
                self.improve_player(i)
        return self.current_team

    def get_scores(self, teams, num=20):
        if(self.opponents == None):
            self.opponents = generate_teams(num)
        if(self.optim == "tourney"):
            scores = []
            for t in teams:
                score = 0
                for i in range(len(self.opponents)):
                    b = Board(t, self.opponents[i])
                    result = b.resolve(False)
                    score += {0:1, 1:0, -1:0.5}[result]
                scores.append(score/len(self.opponents))
            return scores
        elif(self.optim == "net"):
            states = []
            for t in teams:
                for t2 in self.opponents:
                    states.append(Board(t, t2).get_state())
            input_states = torch.Tensor(np.stack(states))
            output_preds = self.gb(input_states)
            scores = torch.nn.functional.softmax(output_preds, dim=1)[:, 0]
            scores = scores.detach().numpy()
            new_scores = []
            i = 0
            for t in teams:
                new_scores.append(0)
                for t2 in self.opponents:
                    new_scores[-1] += scores[i]
                    i += 1
                new_scores[-1] = new_scores[-1] / len(self.opponents)
            return new_scores
        else:
            print("Invalid")
            exit()
        return -1



if(__name__ == "__main__"):
    main()
