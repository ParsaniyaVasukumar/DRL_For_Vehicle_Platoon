from __future__ import division, print_function
import numpy as np
import tensorflow as tf 
from Environment import *
import matplotlib.pyplot as plt

# # This py file using the random algorithm.

# def main():
#     # up_lanes = [3.5/2,3.5/2 + 3.5,250+3.5/2, 250+3.5+3.5/2, 500+3.5/2, 500+3.5+3.5/2]
#     # down_lanes = [250-3.5-3.5/2,250-3.5/2,500-3.5-3.5/2,500-3.5/2,750-3.5-3.5/2,750-3.5/2]
#     left_lanes = [16-2,20-2,24-2]
#     right_lanes = [2,2+4,2+8]
#     width = 7000
#     height = 24 
#     n = 40
#     Env = Environ(left_lanes,right_lanes, width, height)
#     number_of_game = 50
#     n_step = 100
#     V2I_Rate_List = np.zeros([number_of_game, n_step])
#     Fail_Percent = np.zeros([number_of_game, n_step])
#     for game_idx in range(number_of_game):
#         print (game_idx)
#         Env.new_random_game(n)
#         for i in range(n_step):
#             #print(i)
#             actions = np.random.randint(0,20,[n,3])
#             power_selection = np.zeros(actions.shape, dtype = 'int')
#             actions = np.concatenate((actions[..., np.newaxis],power_selection[...,np.newaxis]), axis = 2)
#             reward, percent = Env.act(actions)
#             V2I_Rate_List[game_idx, i] = np.sum(reward)
#             Fail_Percent[game_idx, i] = percent
#         print(np.sum(reward))
#         print ('percentage here is ', percent)
#     print ('The number of vehicles is ', n)
#     print ('mean of V2I rate is that ', np.mean(V2I_Rate_List))
#     print ('mean of percent is ', np.mean(Fail_Percent[:,-1]))

# main()


import numpy as np
import matplotlib.pyplot as plt
from Environment import Environ
from agent import Agent

# # Simulate the Random method
# def simulate_random(env, n, steps):
#     V2I_rate_list = np.zeros(steps)
#     for step in range(steps):
#         actions = np.random.randint(0, 20, [n, 3])  # Random actions
#         power_selection = np.zeros(actions.shape, dtype="int")
#         actions = np.concatenate((actions[..., np.newaxis], power_selection[..., np.newaxis]), axis=2)
#         reward, _ = env.act(actions)
#         V2I_rate_list[step] = np.sum(reward)  # Sum rate for this step
#     return np.mean(V2I_rate_list)

# # Main function for plotting
# def main():
#     left_lanes = [16-2, 20-2, 24-2]
#     right_lanes = [2, 2+4, 2+8]
#     width = 7000
#     height = 24
#     num_vehicles_list = [20, 40, 60, 80, 100]  # Number of vehicles
#     steps = 100
#     num_games = 50

#     env = Environ(left_lanes, right_lanes, width, height)

#     # Results storage for Random method
#     random_results = []

#     for n in num_vehicles_list:
#         env.new_random_game(n)  # Initialize the environment
#         random_results.append(simulate_random(env, n, steps))

#     # Plotting the results
#     plt.figure(figsize=(8, 6))
#     plt.plot(num_vehicles_list, random_results, "bo--", label="Random")
#     plt.xlabel("Number of Vehicles")
#     plt.ylabel("Sum Rate of V2I links (Mb/Hz)")
#     plt.legend()
#     plt.grid()
#     plt.title("Performance of Random Algorithm for V2I Link Sum Rate")
#     plt.savefig("random_performance.png")

# main()

# Simulate the Random method
def simulate_random(env, n, steps):
    V2I_rate_list = np.zeros(steps)
    for step in range(steps):
        actions = np.random.randint(0, 20, [n, 3])  # Random actions
        power_selection = np.zeros(actions.shape, dtype="int")
        actions = np.concatenate((actions[..., np.newaxis], power_selection[..., np.newaxis]), axis=2)
        reward, _ = env.act(actions)
        V2I_rate_list[step] = np.sum(reward)  # Sum rate for this step
    return np.mean(V2I_rate_list)
def simulate_deep_rl(left_lanes, right_lanes, width, height, n):
    print(f"Running DRL simulation with {n} vehicles...")
    
    # Create a new environment for the specified number of vehicles
    Env = Environ(left_lanes, right_lanes, width, height, n_Veh=n)  # Use the same parameters as before
    Env.new_random_game()  # Initialize a new random game

    # Create a TensorFlow session
    with tf.compat.v1.Session() as sess:  # Use tf.compat.v1 for compatibility with TF 1.x code
        agent = Agent([], Env, sess)  # Initialize the agent with the session
        agent.train()  # Train the agent
        
        # Play the game after training
        agent.play()  # Assuming this method collects V2I rates during gameplay

        # Collect the raw V2I rates
        return agent.raw_v2i_rates_over_time  # Return the collected V2I rates


def main():
    left_lanes = [16-2, 20-2, 24-2]
    right_lanes = [2, 2+4, 2+8]
    width = 7000
    height = 24
    vehicle_counts = [30, 40, 50]  # The vehicle counts we want to test

    # Results storage
    random_results = []
    deep_rl_results = []

    env = Environ(left_lanes, right_lanes, width, height)

    for n in vehicle_counts:
        # Random method simulation
        env.new_random_game(n)  # Initialize the environment for random method
        random_results.append(simulate_random(env, n, 100))  # Random method results

        # Deep reinforcement learning method simulation
        deep_v2i_rates = simulate_deep_rl(left_lanes, right_lanes, width, height, n)  # Pass parameters
        deep_rl_results.append(np.mean(deep_v2i_rates))  # Calculate mean V2I rate for DRL

    # Plotting the results
    plt.figure(figsize=(10, 6))
    plt.plot(vehicle_counts, random_results, "bo--", label="Random Method")
    plt.plot(vehicle_counts, deep_rl_results, "ro--", label="Deep Reinforcement Learning")
    plt.xlabel("Number of Vehicles")
    plt.ylabel("Sum Rate of V2I links (Mb/Hz)")
    plt.legend()
    plt.grid()
    plt.title("Comparison of Random Algorithm and Deep Reinforcement Learning for V2I Link Sum Rate")
    plt.savefig("comparison_performance.png")
    plt.show()

if __name__ == "__main__":
    main()
