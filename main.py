import numpy as np
import random
from Retrieval import *
from Placement import *
from Scheduling import*
from Simulation import*
import math
from Placement_heuristic import *

from collections import OrderedDict

if __name__=="__main__":

    problem_dir='/output/problem_set/'
    if not os.path.exists(problem_dir):
        os.makedirs(problem_dir)
    model_dir='big/'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    history_dir='/output/history/'
    if not os.path.exists(history_dir):
        os.makedirs(history_dir)
    device = 'cuda'
    # small problem
    #input_list = [6, 4, (5, 5), (10, 12), (20, 21), 250, 100, 100, 300, [300, 500], 8, (1, 500), 3500, 500, 120, 20, 10]
    input_list = [9, 6, (5, 5), (10, 12), (30, 31), 250, 100, 100, 300, [300, 500], 12, (1, 500), 3500, 500, 120, 20, 10]
    #input_list = [12, 8, (5, 5), (10, 12), (40, 41), 250, 100, 100, 300, [300, 500], 8, (1, 500), 3500, 500, 120, 20, 10]

    # middle problem
    # problem, block: 60 pl:15 tp: 12
    # 0 factory number
    # 1 yard number
    # 2 yard size
    # 3 block number distribution per init yard
    # 4 source block per day
    # 5 storage_period_scale
    # 6 ready high
    # 7 gap
    # 8 tardy high
    # 9 TP capacity type
    # 10 TP number
    # 11 Weight distribution
    # 12 Dis high
    # 13 Dis low
    # 14 TP speed
    # 15 RT weight 15
    # 16 RT time 30
    learning_rate = 0.001
    lmbda = 0.95
    gamma = 1
    discount_factor = 1
    epsilon = 0.2
    alpha = 0.5
    beta = 0.01
    location_num = 10
    lookahead_block_num = 1
    grid_size = (5, 5)

    hidden_dim = 32
    transporter_type = 2
    feature_dim = 1 + transporter_type
    ppo = 0
    #mod = 'GCN2'
    placement = Placement(feature_dim+1, hidden_dim, lookahead_block_num, grid_size, learning_rate, lmbda, gamma, alpha,beta, epsilon, 'GAT').to('cuda')
    checkpoint = torch.load('Placement_network.pth', map_location=torch.device('cuda'))  # 파일에서 로드할 경우

    full_state_dict = checkpoint['model_state_dict']
    filtered_state_dict = OrderedDict({k: v for k, v in full_state_dict.items() if 'Critic_net' not in k})
    placement.load_state_dict(filtered_state_dict, strict=False)

    #placement = Heuristic(grid_size=(5,5),TP_type_len=transporter_type,mod='ASR')
    Simulation = Simulate_yard(input_list, ppo, placement)
    pd.DataFrame(Simulation.Dis).to_excel('yard_middle.xlsx')


    Simulation.ppo = PPO(learning_rate, lmbda, gamma, alpha, beta, epsilon, discount_factor, location_num,
                         transporter_type, Simulation.Dis)

    # small problem
    # problem, block: 40 pl:10 tp: 8
    # 0 factory number
    # 1 yard number
    # 2 yard size
    # 3 block number distribution per init yard
    # 4 source block per day
    # 5 storage_period_scale
    # 6 ready high
    # 7 gap
    # 8 tardy high
    # 9 TP capacity type
    # 10 TP number
    # 11 Weight distribution
    # 12 Dis high
    # 13 Dis low
    # 14 TP speed
    # 15 RT weight 15
    # 15 RT time 15
    scheduling_mode = 'RL_full'
    train_step=1000
    eval_num = 10
    eval_step = 40
    eval_set = []
    for _ in range(eval_num):
        eval_yard, eval_block = Simulation.Create_problem(10)
        eval_set.append([eval_yard.copy(), eval_block.copy()])
    history=np.zeros(1000)
    eval_history=np.zeros(int(train_step/eval_step+10))
    K = 2
    for step in range(train_step):
        data, reward_list, done_list, prob_list, action_list, ave_reward = Simulation.Run_simulation(simulation_day=10,
                                                                                                     scheduling_mode=scheduling_mode,
                                                                                                     init_yard=None,
                                                                                                     init_block=None,
                                                                                                     batch_step=20)
        #vessl.log(step=step, payload={'train_reward': ave_reward})
        print(step, ave_reward)
        history[step]=ave_reward
        for _ in range(K):
            ave_loss, v_loss, p_loss = Simulation.ppo.update(data, prob_list, reward_list, action_list, done_list, step,
                                                             model_dir)
        if step % eval_step == 0:
            eval_reward = 0
            for j in range(eval_num):
                data, reward_list, done_list, prob_list, action_list, ave_reward = Simulation.Run_simulation(
                    simulation_day=10, scheduling_mode=scheduling_mode, init_yard=eval_set[j][0].copy(),
                    init_block=eval_set[j][1].copy(), batch_step=5)
                eval_reward += ave_reward
            print('eval', step, eval_reward/eval_num)
            eval_history[int(step/eval_step)]=eval_reward/eval_num
            #vessl.log(step=step, payload={'eval_reward': eval_reward / eval_num})
    np.save('history_big.npy',history)
    np.save('eval_history_big.npy', eval_history)



