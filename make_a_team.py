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

from util import PLAYERS_PER_TEAM, MAP_WIDTH, MAP_HEIGHT, TURNS_UNTIL_DRAW
from util import Board, generate_teams, save_all, load_all, pad, make_matchups
from util import player_names, player_icons, stat_rev, stats
from util import damage_vals, range_vals, aoe_vals, survival_vals, support_vals, mobility_vals

class team_builder():

    def __init__(self):
        self.MAP_HEIGHT = 8
        self.MAP_WIDTH = 16
        self.cursorx = int(self.MAP_WIDTH / 2)
        self.cursory = int(self.MAP_HEIGHT / 2)
        self.cursor_back = (0,0)
        self.board_arr = []
        for i in range(8):
            self.board_arr.append([-1] * 16)

        self.building_unit = False
        self.stat1 = 0
        self.stat2 = 0
        self.current_stats = None
        self.player_arr = []

    def move_on_board(self, key):
        if(key == "w"):
            self.cursory = max(0, self.cursory - 1)
        elif(key == 'a'):
            self.cursorx = max(0, self.cursorx - 1)
        elif(key == 's'):
            self.cursory = min(self.MAP_HEIGHT-1, self.cursory + 1)
        elif(key == 'd'):
            self.cursorx = min(self.MAP_WIDTH-1, self.cursorx + 1)
        elif(key == 'e'):
            self.building_unit = True
            self.cursor_back = self.cursorx, self.cursory
            self.load_unit()
            self.cursorx, self.cursory = 0, 0
        else:
            print('Key released: {0}'.format(key))

    def load_unit(self):
        if(self.board_arr[self.cursory][self.cursorx] != -1):
            player = self.player_arr[self.board_arr[self.cursory][self.cursorx]]
            current_stats = []
            for j in range(6):
                if(player[j] == 1): current_stats.append(j)
                if(player[j] == 2): current_stats = [j,j]
            self.stat1 = current_stats[0]
            self.stat2 = current_stats[1]
            self.save_stats()
        else:
            self.stat1 = 0
            self.stat2 = 0
            self.save_stats()

    def save_stats(self):
        player = [0] * 6
        player[self.stat1] += 1
        player[self.stat2] += 1
        self.current_stats = tuple(player)

    def save_unit(self):
        at_pos = self.board_arr[self.cursory][self.cursorx]
        if(at_pos == -1):
            self.board_arr[self.cursory][self.cursorx] = len(self.player_arr)
            self.player_arr.append(self.current_stats)
        else:
            self.player_arr[at_pos] = self.current_stats

    def save_team(self):
        real_players = []
        for y_pos, row in enumerate(self.board_arr):
            for x_pos, val in enumerate(row):
                if(val != -1):
                    real_players.append((self.player_arr[val],x_pos,y_pos,))

        os.system('clear')
        _ = input("Hit Enter")
        team_name = input("Enter Team Name:")
        f = open(f"./my_teams/{team_name.strip()}.txt", "w+")
        for p in real_players:
            stats = ' '.join([str(x) for x in p[0]])
            f.write(f"{stats} {p[1]} {p[2]}\n")
        f.close()
        exit()

    def move_on_creator(self, key):
        if(key == "w"):
            self.cursory = max(0, self.cursory - 1)
        elif(key == 'a'):
            self.cursorx = max(0, self.cursorx - 1)
        elif(key == 's'):
            self.cursory = min(2, self.cursory + 1)
            if(self.cursory == 2):
                self.cursorx = min(1, self.cursorx)
        elif(key == 'd'):
            self.cursorx = min(5, self.cursorx + 1)
        elif(key == 'e'):
            if(self.cursory == 0): #First Stat
                self.stat1 = self.cursorx
                self.save_stats()
            elif(self.cursory == 1): #First Stat
                self.stat2 = self.cursorx
                self.save_stats()
            elif(self.cursorx == 0): #Save
                self.building_unit = False
                self.cursorx, self.cursory = self.cursor_back
                self.save_unit()
            else: #Delete
                self.cursorx, self.cursory = self.cursor_back
                self.board_arr[self.cursory][self.cursorx] = -1
                self.building_unit = False
        else:
            print('Key released: {0}'.format(key))

    def on_release(self, key):
        global cursor
        if('char' in key.__dict__): key = str(key.char)
        if(key == 'q'):
            self.save_team()

        if(self.building_unit):
            self.move_on_creator(key)
        else:
            self.move_on_board(key)
        self.print_current_state()

        if key == keyboard.Key.esc:
            # Stop listener
            return False

    def main(self):
        os.system('clear')
        self.print_current_state()

        with keyboard.Listener(
                on_release=self.on_release) as listener:
            listener.join()

    def print_current_state(self):
        os.system('clear')
        s = ""
        for y_pos, row in enumerate(self.board_arr):
            for x_pos, val in enumerate(row):
                if(self.building_unit == False and y_pos == self.cursory and x_pos == self.cursorx):
                    s += Fore.YELLOW + '+' + Fore.RESET
                elif(self.building_unit == True and y_pos == self.cursor_back[1] and x_pos == self.cursor_back[0]):
                    s += Fore.YELLOW + '+' + Fore.RESET
                elif(val == -1):
                    s += '.'
                else:
                    icon = player_icons[self.player_arr[val]]
                    s += Fore.BLUE + icon + Fore.RESET
            s += "\n"
        s += "\/ " * int(self.MAP_WIDTH / 3)
        s += "\n\n"

        if(self.building_unit):
            if(self.board_arr[self.cursor_back[1]][self.cursor_back[0]] == -1):
                s += "Creating New Unit\n"
            else:
                s += f"Editing Unit ID:{self.board_arr[self.cursor_back[1]][self.cursor_back[0]]}\n"

            unit_name = player_names[self.current_stats]
            s += f"Current unit type: {unit_name}\n\n"

            for i in range(6):
                stat_name = stats_rev[i]
                if(self.stat1 == i): stat_name = "<" + stat_name + ">"
                stat_name = pad(stat_name, 10)
                if(self.cursorx == i and self.cursory == 0): stat_name = Fore.YELLOW + stat_name + Fore.RESET
                s += stat_name + '\t'
            s += "\n"
            for i in range(6):
                stat_name = stats_rev[i]
                if(self.stat2 == i): stat_name = "<" + stat_name + ">"
                stat_name = pad(stat_name, 10)
                if(self.cursorx == i and self.cursory == 1): stat_name = Fore.YELLOW + stat_name + Fore.RESET
                s += stat_name + '\t'
            save_str = pad("Save", 10)
            if(self.cursory == 2 and self.cursorx == 0): save_str = Fore.YELLOW + save_str + Fore.RESET
            del_str = pad("Delete", 10)
            if(self.cursory == 2 and self.cursorx == 1): del_str = Fore.YELLOW + del_str + Fore.RESET
            s += f"\n{save_str}{del_str}"

        #s += f"{self.cursorx} {self.cursory}"

        print(s)

if(__name__ == "__main__"):
    tb = team_builder()
    tb.main()
