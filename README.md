
# Reinforcement Learning Problemes

This repository contains Python implementations of reinforcement learning solutions for three classic games: Blackjack, Lunar Lander, and Flappy Bird. The reinforcement learning algorithms used in this project aim to train agents to play these games effectively through trial and error, learning from their interactions with the environment.

## Games

### 1. Blackjack

- **Description:** A simple text-based implementation of the popular card game Blackjack.
- **Objective:** Train an agent to make optimal decisions to maximize rewards and win the game.
- **Observation Space:** The agent receives information about the current state, including the position, velocity, and orientation of the lunar module.


### 2. Lunar Lander

- **Description:** A 2D lunar landing simulation where the agent controls a spacecraft to safely land on the moon's surface.
- **Objective:** Train an agent to land the lunar module safely while considering fuel constraints and avoiding obstacles.
- **Observation Space:** The agent receives information about the current state, including the position, velocity, and orientation of the lunar module.


### 3. Flappy Bird

- **Description:** A simplified version of the popular Flappy Bird game where the agent controls a bird to navigate through pipes.
- **Objective:** Train an agent to learn the optimal timing for jumps to navigate through the pipes and achieve the highest score.
- **Observation Space:** The agent observes the current state of the game, including the bird's position, and the position of upcoming pipes.


## Reinforcement Learning Algorithms

The reinforcement learning solutions for these games are implemented using various algorithms, including:

- Q-Learning
- Deep Q-Networks (DQN)
- Policy Gradient Methods (such as REINFORCE)

The code is modular, allowing for easy experimentation with different algorithms and game environments.

## Acknowledgments

- The base implementations and environments for the games are inspired by various open-source projects.
- Reinforcement learning concepts and algorithms are based on literature.

## Contributing

Feel free to contribute by opening issues, proposing enhancements, or submitting pull requests. Your contributions are highly appreciated!

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
