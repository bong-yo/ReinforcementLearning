import numpy as np


class EventDisplayer:
    def __init__(self, print_every_n: int = 500, render_every_n: int = 50,
                 show_success: bool = True):
        self.print_every_n = print_every_n
        self.render_every_n = render_every_n
        self.show_success = show_success

    def print_episode_stats(self, episode_num, loss, episodes_max_position):
        if episode_num % self.print_every_n == 0:
            avg_last_n_max_pos = np.mean(episodes_max_position[-self.print_every_n:])
            print(f"Episode num : {episode_num}, loss : {loss}, avg_max_pos : {avg_last_n_max_pos}")

    def print_success(self, episode):
        if self.show_success:
            print("Success at :", episode)

    def display(self, environment, episode_num, starting_from):
        '''Show episode actual animation.'''
        if self.render_every_n is not None \
            and episode_num % self.render_every_n == 0 \
                and episode_num > starting_from:
            environment.render()
