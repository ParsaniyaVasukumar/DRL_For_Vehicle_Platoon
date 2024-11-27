from __future__ import division, print_function
import random
from matplotlib import pyplot as plt
import tensorflow.compat.v1 as tf
from agent import Agent
from Environment import *
flags = tf.app.flags

# Model
flags.DEFINE_string('model', 'm1', 'Type of model')
flags.DEFINE_boolean('dueling', False, 'Whether to use dueling deep q-network')
flags.DEFINE_boolean('double_q', False, 'Whether to use double q-learning')

# Environment
flags.DEFINE_string('env_name', 'Breakout-v0', 'The name of gym environment to use')
flags.DEFINE_integer('action_repeat', 4, 'The number of action to be repeated')

# Etc
flags.DEFINE_boolean('use_gpu', True, 'Whether to use gpu or not')
flags.DEFINE_string('gpu_fraction', '1/1', 'idx / # of gpu fraction e.g. 1/3, 2/3, 3/3')
flags.DEFINE_boolean('display', False, 'Whether to do display the game screen or not')
flags.DEFINE_boolean('is_train', True, 'Whether to do training or testing')
flags.DEFINE_integer('random_seed', 123, 'Value of random seed')

FLAGS = flags.FLAGS

# Set random seed
tf.set_random_seed(FLAGS.random_seed)
random.seed(FLAGS.random_seed)

if FLAGS.gpu_fraction == '':
  raise ValueError("--gpu_fraction should be defined")

def calc_gpu_fraction(fraction_string):
  idx, num = fraction_string.split('/')
  idx, num = float(idx), float(num)

  fraction = 1 / (num - idx + 1)
  print(" [*] GPU : %.4f" % fraction)
  return fraction

def main(_):
    left_lanes = [16-2, 20-2, 24-2]
    right_lanes = [2, 2+4, 2+8]
    width = 10000
    height = 24

    vehicle_counts = [30, 40, 50]  # The vehicle counts we want to test
    raw_v2i_rates_all = {n: [] for n in vehicle_counts}  # Dictionary to store rates for each count

    for n_vehicles in vehicle_counts:
        print(f"Running simulation with {n_vehicles} vehicles...")
        Env = Environ(left_lanes, right_lanes, width, height, n_Veh=n_vehicles)  # Create environment with n_Vehicles
        Env.new_random_game()  # Initialize a new random game
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=calc_gpu_fraction(FLAGS.gpu_fraction))
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:
            agent = Agent([], Env, sess)  # Initialize the agent
            agent.train()  # Train the agent
            agent.play()  # Play the game after training

            # Collect the raw V2I rates
            raw_v2i_rates_all[n_vehicles] = agent.raw_v2i_rates_over_time

    # Now we plot the raw V2I rates for each vehicle count
    plt.figure(figsize=(10, 6))

    for n_vehicles, raw_v2i_rates in raw_v2i_rates_all.items():
        # Calculate the mean raw V2I rates over intervals of 250 steps
        mean_raw_v2i_rates = []
        for i in range(0, len(raw_v2i_rates), 250):
            interval_data = raw_v2i_rates[i:i+250]
            if interval_data:  # Check if the interval has data
                mean_raw_v2i_rates.append(np.mean(interval_data))

        # Create x values for plotting
        x_values = np.arange(len(mean_raw_v2i_rates)) * 250 + (250 / 2)  # Midpoint of each interval
        plt.plot(x_values, mean_raw_v2i_rates, label=f'{n_vehicles} Vehicles')

    plt.xlabel('Time Step')
    plt.ylabel('Mean Raw V2I Rate (bps)')
    plt.title('Mean Raw V2I Rate vs Time for Different Vehicle Counts')
    plt.xlim(0, 2100)  # Limit x-axis to 1000 steps
    plt.legend()
    plt.grid(True)
    plt.savefig('mean_raw_v2i_rate_vs_time_multiple_vehicles.png')
  # up_lanes = [3.5/2,3.5/2 + 3.5,250+3.5/2, 250+3.5+3.5/2, 500+3.5/2, 500+3.5+3.5/2]
  # down_lanes = [250-3.5-3.5/2,250-3.5/2,500-3.5-3.5/2,500-3.5/2,750-3.5-3.5/2,750-3.5/2]
  # left_lanes = [16-2,20-2,24-2]
  # right_lanes = [2,2+4,2+8]
  # width = 10000
  # height = 24
  # Env = Environ(left_lanes,right_lanes, width, height)    
  # Env.new_random_game()
  # gpu_options = tf.GPUOptions(
  #     per_process_gpu_memory_fraction=calc_gpu_fraction(FLAGS.gpu_fraction))
  # config = tf.ConfigProto()
  # config.gpu_options.allow_growth = True
  
  
  # with tf.Session(config=config) as sess:
  #   config = []
  #   agent = Agent(config, Env, sess)
  #   #agent.play()
  #   agent.train()
  #   agent.play()


  # vehicle_counts = [30, 40, 50]  # The vehicle counts we want to test
  # raw_v2i_rates_all = {n: [] for n in vehicle_counts}  # Dictionary to store rates for each count

  # for n_vehicles in vehicle_counts:
  #       print(f"Running simulation with {n_vehicles} vehicles...")
  #       Env = Environ(left_lanes, right_lanes, width, height, n_Veh=n_vehicles)  # Set the number of vehicles
  #       Env.new_random_game()  # Initialize a new random game
  #       gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=calc_gpu_fraction(FLAGS.gpu_fraction))
  #       config = tf.ConfigProto()
  #       config.gpu_options.allow_growth = True

  #       with tf.Session(config=config) as sess:
  #           agent = Agent([], Env, sess)  # Initialize the agent
  #           agent.train()  # Train the agent
  #           agent.play()  # Play the game after training

  #           # Collect the raw V2I rates
  #           raw_v2i_rates_all[n_vehicles] = agent.raw_v2i_rates_over_time

  #   # Now we plot the raw V2I rates for each vehicle count
  # plt.figure(figsize=(10, 6))
  # for n_vehicles, raw_v2i_rates in raw_v2i_rates_all.items():
  #     plt.plot(raw_v2i_rates, label=f'{n_vehicles} Vehicles')

  # plt.xlabel('Time Step')
  # plt.ylabel('Mean Raw V2I Rate (bps)')
  # plt.title('Mean Raw V2I Rate vs Time for Different Vehicle Counts')
  # plt.legend()
  # plt.grid(True)
  # plt.savefig('mean_raw_v2i_rate_vs_time_multiple_vehicles.png')
    
if __name__ == '__main__':
    tf.app.run()
