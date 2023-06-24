import numpy as np
import copy, neat
import gymnasium as gym
import matplotlib.pyplot as plt
import visualize as vz


observation, reward, terminated, truncated, info = None, None, None, None, None
config_file = "config.INI"

controlEnv =  "CartPole-v1" #"MountainCar-v0"

env = gym.make(controlEnv, render_mode = "")
observation, info = env.reset()

def stepFunc(x):
    if x[0] > 0.5:
        return 1
    return 0

def eval_genomes(genomes, config):
    global observation, reward, terminated, truncated, info
    for genome_id, genome in genomes:
        terminations = 0
        genome.fitness = 0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        action = 0
        for i in range(1000):            
            observation, reward, terminated, truncated, info = env.step(action)
            output = net.activate(observation)
            action = stepFunc(output)
            if terminated or truncated:
                terminations += 1
                observation, info = env.reset()
        genome.fitness -= terminations

def main():
    global env
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)
    
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(30))

    # Run for up to 300 generations.
    winner = p.run(eval_genomes, 70)
    vz.plot_stats(stats)
    vz.plot_species(stats)
    vz.draw_net(config, winner)
    env.close()

    env2 = gym.make(controlEnv, render_mode = "human")
    observation, info = env2.reset()
    action2 = 0
    net2 = neat.nn.FeedForwardNetwork.create(winner, config)
    for i in range(1000):
        #env.action_space.sample()  # agent policy that uses the observation and info
        observation, reward, terminated, truncated, info = env2.step(action2)
        output2 = net2.activate(observation)
        action2 = stepFunc(output2)

        if terminated or truncated:
            observation, info = env2.reset()

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))


main()
