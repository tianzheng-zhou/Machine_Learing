import Dqn
import classes
import args
import surroundings_function
import multiprocessing
import time
import numpy as np
import torch
import copy
import random
if __name__=='__main__':
    classes.space=surroundings_function.import_maze("C:\\Users\\16329\\source\\repos\\LandingStar\\CST-Project\\maze game\\epsilon-greedy\\maze.txt")
    print(len(classes.space))
    print(args.max_row,args.max_column)

    init_state=random.randint(0,len(classes.space)-1)
    sign=1
    #process_pool=multiprocessing.Pool()
    for cnt_round in range(1000):
        epsilon_policy=classes.policy(args.epsilon_greedy(1))
        print(cnt_round)
        #epsilon_policy=classes.policy(args.random_choice(5))

        init_state=random.randint(0,len(classes.space)-1)
        eps=classes.episode(init_state,epsilon_policy,args.episode_length)
        #sample_return=classes.space[eps.track[-1][0]].actions[eps.track[-1][1]].action_value
        for ind in range(len(eps.track)-2,0,-1):
            s,a=eps.track[ind]

            #sample_return=classes.space[s].actions[a].action_reward+args.gamma*sample_return
            #classes.space[s].actions[a].update(sample_return,lambda x:0.001)
            classes.space[s].actions[a].update(classes.space[s].actions[a].action_reward+args.gamma*max(list(map(lambda x:x.action_value,classes.space[eps.track[ind+1][0]].actions))))
    surroundings_function.plot_maze(classes.space,"target.png")





    #Queue_Sample_to_TN=multiprocessing.Queue(3)
    Pipe_Sample_to_TN_out,Pipe_Sample_to_TN_in=multiprocessing.Pipe(0)
    Queue_Sample_to_TN=multiprocessing.Queue(3)
    #handle_recv,handle_send=multiprocessing.Pipe(0)
    Queue_copy_net=multiprocessing.Queue(5)
    handle=multiprocessing.Queue(maxsize=1)
    net_export_out,net_export_in=multiprocessing.Pipe(0)
    net_export=multiprocessing.Queue(1)
    stop_flag=multiprocessing.Value("i",0)

    Target_Net_Processing=multiprocessing.Process(target=Dqn.Target_Net_Processing,args=(Queue_Sample_to_TN,net_export,Queue_copy_net,stop_flag))
    Sample_Processing=multiprocessing.Process(target=Dqn.Sample_Processing,args=(classes.space,args.max_row,args.max_column,Queue_Sample_to_TN,handle,Queue_copy_net,stop_flag))
    Sample_Processing.start()
    Target_Net_Processing.start()

    net_export_in.close()

    for i in range(args.run_round):
        handle.put(1)
        print(i,flush=1)
    stop_flag.value=1
    handle.put(0)
    outcome_net=net_export.get()
    outcome_net.to(args.device)
    handle.close()
    #Target_Net_Processing.terminate()
    print("net got")
    state_num=len(classes.space)
    for s in range(state_num):
        
        x=np.float32(s%args.max_column)
        y=np.float32(s//args.max_column)
        action_values=list(outcome_net.forward(torch.tensor((x,y)).to(args.device)).to("cpu").detach())
        print(action_values)
        #classes.space[s].actions[a].action_value=np.float64(outcome_net.forward(torch.tensor((x,y,np.float32(a))).to(args.device)).detach().item())
        for a in range(len(classes.space[s].actions)):
            classes.space[s].actions[a].action_value=action_values[a]
            print(f"{s}_{a}/{state_num},action_value:{classes.space[s].actions[a].action_value}")

    surroundings_function.plot_maze(classes.space)
    print("maze printed")
    surroundings_function.out_figure()
    Target_Net_Processing.join()
    Sample_Processing.join()
    Target_Net_Processing.close()
