Multi-Armed Bandits in Cricket Simulation

This project implements and compares four different Multi-Armed Bandit (MAB) algorithms—UCB1, Thompson Sampling, LinUCB, and a simple Neural Contextual Bandit—to optimize shot selection for a batsman in a simulated limited-overs cricket match (10 wickets, 60 balls).

The goal of the agents is to learn which shots (arms) yield the highest expected reward (runs) under various game contexts (wickets remaining, balls remaining, momentum).

🏏 Simulation Environment (CricketMDPEnvironment)

The environment simulates a single batting innings with the following key elements:

Arms (Shots): 8 distinct cricket shots (e.g., 'defensive push', 'cover drive', 'sweep'), each with properties like base_run_prob, aggression, and aerial_risk.

State/Context: The reward for a shot is influenced by:

wickets_left (0 to 10)

balls_left (0 to 60)

pressure_factor (derived from wickets left)

momentum (a simple tracker of recent success/failure)

Reward/Penalty:

Runs scored (0, 1, 2, 3, 4, 6) are positive rewards.

A wicket is treated as a negative reward of -5.

🧠 Algorithms Implemented

The project compares two classic MAB approaches (UCB1, Thompson Sampling) with two Contextual Bandit approaches (LinUCB, Neural Contextual Bandit) that leverage the game state.

1. UCB1 (Upper Confidence Bound 1)

This is a classic non-contextual bandit algorithm that relies on the optimism in the face of uncertainty principle. It balances exploitation (choosing the arm with the highest average reward) and exploration (choosing arms with high uncertainty).

Selection Rule:

$$i\_t = \arg\max\_i \left( \hat{\mu}\_i(t) + c \sqrt{\frac{\ln(t)}{N\_i(t)}} \right)

$$Where $\hat{\mu}_i(t)$ is the empirical mean reward for arm $i$, $N_i(t)$ is the number of times arm $i$ has been played, and $t$ is the total number of plays.

### 2\. Thompson Sampling (Beta-Bernoulli)

Thompson Sampling uses a Bayesian approach, maintaining a belief (a Beta distribution) over the true reward probability of each arm.

**Prior & Posterior Update (using an adjusted reward system):**
The reward $r$ for an arm $A_t$ is used to update the Beta distribution parameters $\alpha$ and $\beta$.$$

\theta_i \sim \text{Beta}(\alpha_i, \beta_i)
$$$$

\alpha_{A_t} \leftarrow \alpha_{A_t} + r

$$$$\
\beta_{A_t} \leftarrow \beta_{A_t} + (5 - r) \quad \text{(where 5 is the maximum reward mapping)}
$$**Arm Selection:** The arm is chosen by sampling from the posterior distribution of each arm's mean reward $\tilde{\theta}_i$ and selecting the maximum.
$$A\_t = \arg\max\_i \tilde{\theta}\_i$$

3. LinUCB (Contextual Bandit)

LinUCB is a linear model that assumes the reward of an arm is a linear function of the context vector $\mathbf{x}_t$. It applies the UCB principle in the parameter space.

Estimated Parameter:
$$

\hat{\theta}_a = A_a^{-1} b_a
$$Where $A_a$ is the covariance matrix and $b_a$ is the sum of rewards times context vectors for arm $a$.

Upper Confidence Bound (UCB) Calculation:
$$\text{UCB}a = \underbrace{\hat{\theta}a^\top x_t}{\text{Mean Prediction}} + \underbrace{\alpha \sqrt{x_t^\top A_a^{-1} x_t}}{\text{Confidence Interval}}

$$### 4. Neural Contextual Bandit

This method uses a simple 1-hidden-layer Neural Network to model the non-linear relationship between the game context $\mathbf{x}_t$ and the expected Q-value (reward) for each shot.

**Model:**
$$\
h = \text{ReLU}(x W_1)
 \\   
Q = h W\_2

$$**Arm Selection:** Selection is based on the maximum Q-value plus Gaussian exploration noise ($\epsilon$):$$

A_t = \arg\max_a \left( Q_a + \epsilon \right)
$$The network is trained using Stochastic Gradient Descent (SGD) on a replay buffer of past experiences.

📊 Results

The simulation runs one full innings (60 balls or 10 wickets) for each agent and compares their cumulative regret and final total runs scored.

Cumulative Regret Plot

The cumulative regret measures the difference between the maximum possible optimal score (if the best shot was played every time) and the runs actually scored by the agent. A lower, flatter regret curve is better.

The plot below shows the performance of the four agents over 60 balls:

Final Performance Comparison

Agent

Final Runs Scored

Final Regret

UCB1

26

-26.0

ThompsonSampling

0

0.0

LinUCBContextual

47

-47.0

NeuralContextualBandit

0

0.0

Conclusion:

The LinUCBContextual agent demonstrated the best performance, scoring the highest number of runs. This highlights the effectiveness of using the context (wickets, balls, momentum) when making sequential decisions in a non-stationary environment like a cricket innings, compared to non-contextual MABs (UCB1, TS) which struggle to adapt to changing game states.

Best Agent's Shot Distribution (LinUCB)

The most successful agent (LinUCB) learned to primarily rely on a balanced shot, avoiding the highly risky or overly defensive options:

shot
straight drive      24
defensive push       4
cover drive          2
reverse sweep        2
lofted off drive     1
pull/hook            1
sweep                1
scoop                1


🚀 Setup and Usage

Dependencies: Ensure you have Python and the necessary libraries installed:

pip install numpy pandas matplotlib seaborn


Run the Notebook: Execute the multi.ipynb notebook to run the simulations, train the models, and generate the results and the cricket_regret.png plot.

jupyter notebook multi.ipynb


Run the Script: Alternatively, you can run the python code directly.

# Assuming you converted the notebook to a script named 'cricket_bandits.py'
python cricket_bandits.py
