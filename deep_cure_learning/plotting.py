import matplotlib.pyplot as plt

def plot(env):
    fig = plt.figure()

    n = len(env.hist_infected)
    ax0 = fig.add_subplot(5,1,1)
    ax0.plot(range(n), env.hist_infected, label='Infected')
    ax0.plot(range(n), env.hist_severe, label='severe')
    ax0.plot(range(n), env.hist_dead, label='dead')
    ax0.plot(range(n), env.hist_border, label='infected from borders')
    for i,fc in enumerate(env.f_countries):
        ax0.plot(range(n), fc.hist_infected, label=f'Infected foreign {i}')
    ax0.legend()

    ax1 = fig.add_subplot(5,1,2)
    ax1.plot(range(n), env.hist_internal_infection_rate, label='infection rate')
    ax1.plot([0,n], [env.v_base_infect_rate, env.v_base_infect_rate], label='base infection rate')
    ax1.legend()

    ax2 = fig.add_subplot(5,1,3)
    ax2.plot(range(n), env.hist_reward, label='Reward')
    ax2.legend()

    ax3 = fig.add_subplot(5,1,4)
    ax3.plot(range(n), env.hist_new_infected, label='new infected')
    ax3.plot(range(n), env.hist_new_severe, label='new severe')
    ax3.plot(range(n), env.hist_new_dead, label='new dead')
    ax3.plot([0,n], [env.hospital_capacity, env.hospital_capacity], label='hospital capacity')
    ax3.legend()

    ax4 = fig.add_subplot(5,1,5)
    ax4.plot(range(n), env.hist_action, label='actions')
    # ax4.yticks([0,1,2,3], ['no action', 'masks', 'curfew', 'all'])
    ax4.legend()

    plt.show()
