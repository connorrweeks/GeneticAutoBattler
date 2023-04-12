import numpy as np
import pickle
import os
import colorama
from colorama import Fore, Back, Style
import random as r
import math
import pandas as pd
import time

STATS_LOG = []
ENT_LOG = []
TURNS_UNTIL_DRAW = 50
PLAYERS_PER_TEAM = 5
MAP_WIDTH = 16
MAP_HEIGHT = 8
NUM_MATCHES = 5
DEBUG = False

class Board():
    def __init__(self, team1, team2):
        self.team1 = team1
        self.team2 = team2
        self.team1 = [list(x) for x in self.team1]
        self.team2 = [[x[0], MAP_WIDTH-1-x[1], (2*MAP_HEIGHT)-1-x[2]]  for x in self.team2]

        self.owner_arr = ([0] * PLAYERS_PER_TEAM) + ([1] * PLAYERS_PER_TEAM)
        self.player_arr = self.team1 + self.team2
        self.n = len(self.player_arr)
        self.max_health_arr = [survival_vals[x[0][stats["Survival"]]] for x in self.player_arr]
        self.health_arr = self.max_health_arr[:]

        positions = set([(x[1], x[2],) for x in self.player_arr])
        if(len(positions) != len(self.player_arr)):
            print(team1)
            print(team2)
        assert(len(positions) == len(self.player_arr))

        self.board_arr = []
        for i in range(MAP_HEIGHT*2):
            self.board_arr.append([-1 for j in range(MAP_WIDTH)])

        for i in range(self.n):
            p = self.player_arr[i]
            #if(self.owner_arr[i] == 0):
            self.board_arr[p[2]][p[1]] = i
            #if(self.owner_arr[i] == 1):
            #    self.board_arr[9 - p[2]][9 - p[1]] = i

        self.move_order = r.sample(list(range(self.n)), self.n)
        self.target_arr = [-1] * self.n

        self.state = None

    def get_state(self):
        state_arr = np.zeros((9, MAP_WIDTH, MAP_HEIGHT*2)) #team1, team2, stats, health

        for x in range(MAP_HEIGHT*2):
            for y in range(MAP_WIDTH):
                if(self.board_arr[y][x] != -1):
                    p = self.board_arr[y][x]
                    if(self.owner_arr[p] == 0): state_arr[0, x, y] = 1
                    if(self.owner_arr[p] == 1): state_arr[1, x, y] = 1
                    for i in range(6):
                        state_arr[i+2, x, y] = self.player_arr[p][0][i]
                    state_arr[8, x, y] = self.health_arr[p] / self.max_health_arr[p]
        return state_arr

    def take_turn(self):
        for p in self.move_order:
            if(self.health_arr[p] <= 0): continue
            current = self.player_arr[p]

            if(DEBUG): print(f"Starting turn for {p}")
            if(DEBUG): print("current", current)
            if(DEBUG): print("current health", self.health_arr[p])

            if(self.retarget(p, current)): break

            healed_target = self.heal_action(p, current)
            if(not healed_target):
                can_move = self.move_action(p, current)
            self.attack_action(p, current)

        #Find Winner
        self.health_arr = [0 if x < 0 else x for x in self.health_arr]
        team1_health = sum(self.health_arr[:PLAYERS_PER_TEAM])
        team2_health = sum(self.health_arr[PLAYERS_PER_TEAM:])

        if(team1_health <= 0):
            self.state = 1
            return 1
        if(team2_health <= 0):
            self.state = 0
            return 0
        return -1

    def retarget(self, p, current):
        target = self.target_arr[p]
        if(target == -1 or self.health_arr[target] <= 0 or current[0][stats["Mobility"]] > 0): #Retarget if needed
        #if(target == -1 or self.health_arr[target] <= 0): #Retarget if needed
            best_t, best_d = -1, 999
            for n_t in range(self.n):
                if(self.owner_arr[n_t] != self.owner_arr[p] and self.health_arr[n_t] > 0):
                    d = self.dist(p, n_t, True)
                    if(d < best_d):
                        best_t = n_t
                        best_d = d
            self.target_arr[p] = best_t
            if(DEBUG): print(f"Targeting {best_t}")
            if(DEBUG): print("target", self.player_arr[self.target_arr[p]])
        if(self.target_arr[p] == -1):
            return True
        return False

    def move_action(self, p, current):
        move_speed = mobility_vals[current[0][stats["Mobility"]]]
        if(DEBUG): print(f"Distance to target {self.dist(p, self.target_arr[p]):.2f}")
        if(self.dist(p, self.target_arr[p]) > range_vals[current[0][stats["Range"]]]): #Move if needed
            best_d, best_x, best_y = 999, -1, -1
            x_search = list(range(max(0, int(current[1] - move_speed - 1)), min(int(current[1] + move_speed + 1), MAP_WIDTH)))
            for n_x in r.sample(x_search, len(x_search)):
                y_search = list(range(max(0, int(current[2] - move_speed - 1)), min(int(current[2] + move_speed + 1), MAP_HEIGHT*2)))
                for n_y in r.sample(y_search, len(y_search)):
                    dist_to = self.dist(p, n_x, y=n_y)
                    if(dist_to > move_speed): continue
                    if(self.board_arr[n_y][n_x] != -1): continue

                    dist_from = 0
                    dist_from = self.dist(self.target_arr[p], n_x, y=n_y)
                    #print(dist_from)
                    if(dist_from < range_vals[current[0][stats["Range"]]]): dist_from = 0
                    #print(dist_from)
                    if(current[0][stats["Mobility"]] > 0):
                        dist_from -= sum([math.sqrt(self.dist(p, j)) for j in range(self.n) if self.health_arr[j] > 0]) / 10
                    #print(dist_from)
                    #print("Adjusting")
                    if(dist_from < best_d):
                        best_d, best_x, best_y = dist_from, n_x, n_y
            #print(best_x, best_y, best_d)
            #exit()
            #print(best_d)
            if(best_x == -1):
                return True
                print("No valid moves???")
                exit()
            if(DEBUG): print(f"Moving from {(current[1],current[2],)} to {(best_x, best_y,)}")
            assert(self.board_arr[current[2]][current[1]] == p)
            assert(self.board_arr[best_y][best_x] == -1)
            self.board_arr[current[2]][current[1]] = -1
            self.board_arr[best_y][best_x] = p
            current[1] = best_x
            current[2] = best_y
        #if(current[0][stats["Mobility"]] > 0):
        #    exit()
        return False

    def attack_action(self, p, current):
        t = self.target_arr[p]
        if(self.dist(p, t) < range_vals[current[0][stats["Range"]]]): #Attack
            if(DEBUG): print("Attacking")
            if(current[0][stats["AOE"]] > 0):
                for n_t in range(self.n):
                    if(self.owner_arr[n_t] != self.owner_arr[p] and self.health_arr[n_t] > 0 and self.dist(t, n_t) <= aoe_vals[current[0][stats["AOE"]]]):
                        self.health_arr[n_t] -= damage_vals[current[0][stats["Damage"]]]
                        if(self.health_arr[n_t] <= 0):
                            self.board_arr[self.player_arr[n_t][2]][self.player_arr[n_t][1]] = -1
            else:
                self.health_arr[t] -= damage_vals[current[0][stats["Damage"]]]
                if(self.health_arr[t] <= 0):
                    self.board_arr[self.player_arr[t][2]][self.player_arr[t][1]] = -1

    def heal_action(self, p, current):
        heal_range = 5.5
        if(current[0][stats["Support"]] > 0): #Heal
            if(DEBUG): print("Healing")
            best_t, best_d = -1, 999
            for h_t in range(self.n): #Find heal target
                d = self.dist(p, h_t)
                if(d >= heal_range): continue
                if(self.owner_arr[h_t] != self.owner_arr[p]): continue
                if(self.health_arr[h_t] <= 0): continue
                if(self.health_arr[h_t] == self.max_health_arr[h_t]): continue
                if(h_t == p): continue
                if(d < best_d):
                    best_d = d
                    best_t = h_t

            if(best_t != -1): #Heal target found
                if(current[0][stats["AOE"]] > 0):
                    for h_t in range(self.n):
                        if(self.owner_arr[h_t] == self.owner_arr[p] and self.health_arr[h_t] > 0 and self.dist(best_t, h_t) <= aoe_vals[current[0][stats["AOE"]]]):
                            self.health_arr[h_t] += support_vals[current[0][stats["Support"]]]
                            if(self.health_arr[h_t] >= self.max_health_arr[h_t]):
                                self.health_arr[h_t] = self.max_health_arr[h_t]
                else:
                    self.health_arr[best_t] += support_vals[current[0][stats["Support"]]]
                return True
        return False

    def dist(self, p1, p2, targeting=False, y=None):
        if(y != None):
            P1 = self.player_arr[p1]
            return math.sqrt(((P1[1] - p2) ** 2) + ((P1[2] - y) ** 2))
        P1 = self.player_arr[p1]
        P2 = self.player_arr[p2]
        travel_dist = math.sqrt(((P1[1] - P2[1]) ** 2) + ((P1[2] - P2[2]) ** 2))
        if(P1[0][stats["Mobility"]] > 0 and targeting): #Mobile units get better targeting
            return travel_dist + self.health_arr[p2] / 20.0
        else:
            return travel_dist

    def resolve(self, animate=False, game_speed=0.2, get_states=False):
        if(animate):
            output = self.print(show=True)
            time.sleep(game_speed)
        if(get_states): states = []
        for turn_num in range(TURNS_UNTIL_DRAW):
            result = self.take_turn()
            if(get_states): states.append(self.get_state())
            if(animate):
                output = self.print(show=True)
                if(result != -1): print(f"Winner is {result}")
                time.sleep(game_speed)
            if(result != -1):
                if(get_states): return result, states
                return result
        if(get_states): return -1, states
        return -1

    def print(self, show=True, numbers=False):
        if(not DEBUG and show): os.system('clear')
        s = ""
        for row in self.board_arr:
            for val in row:
                if(self.state == 0):
                    s += Fore.RED + '.' + Fore.RESET
                elif(self.state == 1):
                    s += Fore.BLUE + '.' + Fore.RESET
                elif(val == -1 or self.state == 2):
                    s += '.'
                else:
                    team_color = (Fore.RED if val < PLAYERS_PER_TEAM else Fore.BLUE)
                    health_style = Style.NORMAL
                    if(self.health_arr[val] <= self.max_health_arr[val] / 2): health_style = Style.DIM
                    if(self.health_arr[val] == self.max_health_arr[val]): health_style = Style.BRIGHT
                    if(numbers):
                        icon = str(val)
                    else:
                        icon = player_icons[self.player_arr[val][0]]
                    s += team_color + health_style + icon + Fore.RESET + Style.RESET_ALL
            s += "\n"
        if(show):
            print(s)
        else:
            return s


def pad(s, n):
    assert(len(s) <= n)
    return s + (" " * (n-len(s)))

def save_all(teams, file_name):
    f = open(file_name, "w+")
    for t in teams:
        for p in t:
            stats = ' '.join([str(x) for x in p[0]])
            f.write(f"{stats} {p[1]} {p[2]}\n")
        f.write('\n')
    f.close()

    df = pd.DataFrame()
    for k in player_names:
        df[player_names[k]] = [x[k] for x in STATS_LOG]
    df["ENT"] = ENT_LOG
    df.to_csv('./stats_log.csv')

def load_all(file_name):
    f = open(file_name)
    teams = f.read().strip().split('\n\n')
    teams = [[[int(x) for x in p.split(' ')] for p in t.split('\n')] for t in teams]
    teams = [[(tuple(p[:6]), p[6], p[7],) for p in t] for t in teams]
    return teams

def make_matchups(num_teams, matches_per):
    op1 = list(range(num_teams)) * matches_per
    op2 = list(range(num_teams)) * matches_per
    r.shuffle(op1)
    r.shuffle(op2)

    real_op2 = []
    while(len(op2) > 0):
        i = 0
        while(i < len(op2) and op2[i] == op1[len(real_op2)]):
            i += 1
        if(i >= len(op2)): #Unlucky
            j = 0
            while(op1[j] == op2[0] or real_op2[j] == op1[len(real_op2)]):
                j += 1
            real_op2.append(real_op2[j])
            real_op2[j] = op2.pop(0)
        else:
            real_op2.append(op2.pop(i))
    return [(op1[i], real_op2[i],) for i, x in enumerate(op1)]

def generate_teams(num):
    teams = []
    for i in range(num):
        team = []
        pos = set()
        for j in range(PLAYERS_PER_TEAM):
            stats = [0] * 6
            stats[r.randrange(6)] += 1
            stats[r.randrange(6)] += 1

            X = r.randrange(MAP_WIDTH)
            Y = r.randrange(MAP_HEIGHT)
            while((X, Y,) in pos):
                X = r.randrange(MAP_WIDTH)
                Y = r.randrange(MAP_HEIGHT)
            pos.add((X,Y,))
            player = (tuple(stats), X, Y,)
            team.append(player)
        teams.append(team)
    return teams

def ent_str(entropy):
    if(entropy > 2.5): return 'Diverse Meta'
    if(entropy > 2 and entropy < 2.5): return 'Varied Meta'
    if(entropy > 1.5 and entropy < 2): return 'Stagnant Meta'
    if(entropy < 1.5): return 'Extreme Meta'
    return '???'

def create_stats(teams, wins, print_stats=True):
    player_wins = {k:0 for k in player_names}
    player_counts = {k:0 for k in player_names}
    stat_counts = [0] * 6
    for i, team in enumerate(teams):
        for p in team:
            player_wins[p[0]] += wins[i]
            player_counts[p[0]] += 1
            for j in range(6):
                stat_counts[j] += p[0][j]
    player_win_rates = {k:(-1 if player_counts[k] == 0 else player_wins[k] / player_counts[k] / (NUM_MATCHES * 2)) for k in player_names}
    player_play_rates = {k:player_counts[k]/len(teams) for k in player_names}
    stat_rates = [x/sum(stat_counts) for x in stat_counts]

    combos = [(player_names[k], player_wins[k], player_win_rates[k], player_play_rates[k]) for k in player_names]
    combos = sorted(combos, key=lambda x: x[3], reverse=True)

    STATS_LOG.append(player_play_rates)

    entropy = [player_play_rates[k] for k in player_play_rates]
    entropy = [x/sum(entropy) for x in entropy if x != 0]
    entropy = -1 * sum([x*math.log(x) for x in entropy])
    ENT_LOG.append(entropy)

    if(not print_stats): return entropy
    print(f"Current entropy: {entropy:.2f}")
    print(ent_str(entropy))

    print(f"class_name\twin_rate\tavg_per_team")
    for n, w, wr, pr  in combos:
        print(f"{n + (' ' * (15 - len(n)))}\t{wr:.2f}\t\t{pr:.2f}")
    print("Stat Rates: ", ' '.join([stat_rev[i] + ':' + str(stat_rates[i])[:5] for i in range(6)]))
    for i in range(6):
        if(stat_rates[i] < 0.05): print(f"Low {stat_rev[i]}!")
        if(stat_rates[i] > 0.3): print(f"High {stat_rev[i]}!")

stat_rev = ["Damage", "Range", "AOE", "Survival", "Support", "Mobility"]
stats = {"Damage":0, "Range":1, "AOE":2, "Survival":3, "Support":4, "Mobility":5}

damage_vals = [25, 45, 75]
range_vals = [1.5, 3.5, 9]
aoe_vals = [0, 2.5, 7.5]
survival_vals = [200, 400, 900]
support_vals = [0, 30, 50]
mobility_vals = [1.2, 2.5, 6.5]

player_names = {(0,0):"Assassin", (0,1):"Gunner", (0,2):"Exploder", (0,3):"Duelist", (0,4):"Battle Medic", (0,5):"Ninja",
                (1,1):"Sniper", (1,2):"Mage", (1,3):"Cannon", (1,4):"Druid", (1,5):"Archer",
                (2,2):"Hazard", (2,3):"Gladiator", (2,4):"Support", (2,5):"Raider",
                (3,3):"Wall", (3,4):"Paladin", (3,5):"Charger",
                (4,4):"Healer", (4,5): "Monk",
                (5,5):"Flyer"}

player_icons = {(0,0):"A", (0,1):"g", (0,2):"X", (0,3):"D", (0,4):"B", (0,5):"n",
                (1,1):"s", (1,2):"m", (1,3):"c", (1,4):"d", (1,5):"a",
                (2,2):"Z", (2,3):"G", (2,4):"S", (2,5):"R",
                (3,3):"W", (3,4):"P", (3,5):"C",
                (4,4):"h", (4,5): "M",
                (5,5):"f"}

def fix_map(d):
    n_d = {}
    for k in d:
        n_k = [0] * 6
        n_k[k[0]] += 1
        n_k[k[1]] += 1
        n_d[tuple(n_k)] = d[k]
    return n_d

player_names = fix_map(player_names)
player_icons = fix_map(player_icons)
