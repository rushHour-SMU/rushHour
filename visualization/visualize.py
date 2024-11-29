import matplotlib.pyplot as plt


def plot_rewards_and_steps(rewards, steps):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(rewards)
    plt.title("Total Reward per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")

    plt.subplot(1, 2, 2)
    plt.plot(steps)
    plt.title("Steps Taken per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Steps Taken")

    plt.tight_layout()
    plt.show()
