"""

Game Rules

-infinite deck of cards

-Each card has a value between 1-10 (uniform probability)
-Each card has a colour red (probability 1/3) or black (probability 2/3)
-No aces or special cards
-Both players draw a black card at the beginning
-A player sticks or hits
-Hit = draw another card
-Stick = don't draw any more cards
-Total value = black cards - red cards
-If the sum>21 then sum=1
-Dealer always stacks on >=17
-Win +1, Lose -1, draw 0

"""
import sys
import enum
import time

import numpy as np
import matplotlib.pyplot as plt

from base_env import BaseEnv

class Action(enum.Enum):
    hit = 1
    stick = 0

class Colour(enum.Enum):
    black = 0
    red = 1

class Easy21(BaseEnv):

    def __init__(self):
        self.action_space = [Action.stick, Action.hit]
        self.action_space_shape = (1,)
        self.state_space_shape = (22,12)

    def reset(self):
        self.dealer_state = np.random.choice(range(1, 11))
        self.player_state = np.random.choice(range(1, 11))

        return (self.player_state, self.dealer_state)

    def step(self, action):

        done = False
        reward = 0

        if action is Action.hit or action==Action.hit.value:

            self.player_state  = self.player_state  + self.draw_card()

            if self.player_state  > 21 or self.player_state<1:
                self.player_state  = 0
                reward = -1
                done = True

        elif action is Action.stick or action==Action.stick.value:
            done = True
            while self.dealer_state>=1 and self.dealer_state<17:
                self.dealer_state += self.draw_card()
            if self.dealer_state>21 or self.dealer_state<1:
                self.dealer_state = 0
            if self.dealer_state>self.player_state :
                reward = -1
            elif self.dealer_state<self.player_state :
                reward = 1
            else:
                reward = 0
        else:
            print("You entered an invalid action.\nPossible actions are 0 (Action.hit) and 1 (Action.stick).")

        return (self.player_state ,self.dealer_state), reward, done, None

    def draw_card(self):
        new_card_value = np.random.choice(range(1, 11))
        new_card_colour = np.random.choice([Colour.black, Colour.red], p = [2/3, 1/3])

        if new_card_colour is Colour.red:
            new_card_value = -new_card_value

        return new_card_value

    def play_game(self):
        state, dealer_state = self.reset()

        print("Your score is {}".format(state))
        print("Dealer's score is {}".format(dealer_state))

        done = False

        while not done:
            action = int(input("Enter your action: 1 - Hit or 0 - Stick\n"))
            if action == 1:
                (state, dealer_state), reward, done, info = self.step(Action.hit)
            elif action==0:
                (state, dealer_state), reward, done, info = self.step(Action.stick)
            else:
                print("Please enter a valid action")
                continue

            print("Your score is {}".format(state))
            print("Dealer's score is {}".format(dealer_state))

        if reward == 1:
            print("YOU WON")
        if reward == -1:
            print("YOU LOST")
        if reward == 0:
            print("DRAW")
