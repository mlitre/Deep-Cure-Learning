import matplotlib.pyplot as plt

def plot(env, average_returns = None, show_borders=True):
    fig = plt.figure()
    total = 4
    if average_returns is not None:
        total = 6

    n = len(env.hist_infected)
    ax0 = fig.add_subplot(total,1,1)
    ax0.set_title('Total number of infected/severe/dead')
    ax0.set_xlabel('time')
    ax0.set_ylabel('citizens')
    ax0.plot(range(n), env.hist_infected, label='infected')
    ax0.plot(range(n), env.hist_severe, label='severe')
    ax0.plot(range(n), env.hist_dead, label='dead')
    ax0.set_xlim(left=0,right=n)
    ax0.set_ylim(bottom=0)
    # for i,fc in enumerate(env.f_countries):
    #     ax0.plot(range(n), fc.hist_infected, label=f'Infected foreign {i}')
    ax0.legend()

    ax1 = fig.add_subplot(total,1,2)
    ax1.set_title('New number of infected/severe/dead')
    ax1.set_xlabel('time')
    ax1.set_ylabel('citizens')
    ax1.plot(range(n), env.hist_new_infected, label='new infected')
    ax1.plot(range(n), env.hist_new_severe, label='new severe')
    ax1.plot(range(n), env.hist_new_dead, label='new dead')
    ax1.plot(range(n), env.hist_border, label='new infected from borders')
    ax1.plot([0,n], [env.hospital_capacity, env.hospital_capacity], label='hospital capacity')
    ax1.set_xlim(left=0,right=n)
    ax1.set_ylim(bottom=0)
    ax1.legend()

    ax2 = fig.add_subplot(total,1,3)
    ax2.set_title('Reward')
    ax2.set_xlabel('time')
    ax2.set_ylabel('reward')
    ax2.plot(range(n), env.hist_reward)
    ax2.set_xlim(left=0,right=n)

    ax3 = fig.add_subplot(total,1,4)
    ax3.set_title('Actions')
    ax3.set_xlabel('time')
    ax3.set_ylabel('action')
    ax3.set_yticks([0,1,2,3])
    ax3.set_yticklabels(['no action', 'masks', 'curfew', 'mask & curfew'])
    actions = list()
    border = list()
    for is_mask,is_curfew,is_border_open in env.hist_action:
        border.append(is_border_open)
        if not is_mask and not is_curfew:
            actions.append(0)
        elif is_mask and not is_curfew:
            actions.append(1)
        elif not is_mask and is_curfew:
            actions.append(2)
        else:
            actions.append(3)
    ax3.plot(range(n), actions, label='actions')
    ax3.plot(range(n), border, label='border')
    ax3.set_xlim(left=0,right=n)
    ax3.legend()

    if average_returns is not None:
        ax5 = fig.add_subplot(5,1,5)
        ax5.plot(range(len(average_returns)), average_returns)

    fig.tight_layout()
    plt.show()
