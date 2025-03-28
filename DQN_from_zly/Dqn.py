import multiprocessing.managers
import multiprocessing.queues
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
import torch
import torch.nn as nn
import multiprocessing
import random
import args
import classes
import numpy as np
import copy
import time
import matplotlib.pyplot as plt

Net_copy = multiprocessing.Queue(5)


def train(self, mini_batch, losses):
    #criterion = torch.nn.MSELoss(reduction='mean')   
    criterion = torch.nn.SmoothL1Loss()
    optimizer = torch.optim.SGD(self.parameters(), lr=args.lr_rate)
    batch_length = len(mini_batch)
    sa, action_value = zip(*mini_batch)
    sa = list(sa)
    action_value = list(action_value)

    #next_states = np.array(next_states)
    pred = []
    #pred=torch.zeros(batch_length,device=args.device)

    action_value = torch.cat(action_value)
    #pred=self(torch.cat(sa))
    for epoch in range(batch_length):
        pred.append(self(torch.tensor(sa[epoch]).to(args.device)))
        #print(f"TN sa:{sa[epoch]}")
        #print(f"TN pred:{pred}")
        #print(f"TN action_value{action_value[epoch]}")
        #print(f"TN av tensor{torch.tensor((action_value[epoch],)).to(args.device)}")
    pred = torch.cat(pred)
    pred.to(args.device)
    #loss = criterion(pred, torch.tensor(action_value).to(args.device))  
    loss = criterion(pred, action_value.to(args.device))
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(self.parameters(), 100)
    optimizer.step()
    losses.append(loss.item())

    #pred=self(torch.cat(sa))
    """for epoch in range(batch_length):
        #pred.append(self(torch.tensor(sa[epoch]).to(args.device)))
        #print(f"TN sa:{sa[epoch]}")
        #print(f"TN pred:{pred}")
        #print(f"TN action_value{action_value[epoch]}")
        #print(f"TN av tensor{torch.tensor((action_value[epoch],)).to(args.device)}")
        #loss = criterion(pred, torch.tensor(action_value).to(args.device))  
        loss = criterion(self(torch.tensor(sa[epoch]).to(args.device)), action_value[epoch].to(args.device))  
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.parameters(), 100)
        optimizer.step()
        losses.append(loss.item())"""
    return losses


def Target_Net_Processing(in_Pipe, export_net, queue_copy, stop_flag):
    time.sleep(1)
    #print(2)
    losses = []
    #raw_net=classes.QN(3,1)
    #net=classes.QN(3,1)
    net = classes.QN(2, 9)
    net.to(args.device)
    flag = 1
    """
    def train(self,mini_batch):
        criterion = torch.nn.MSELoss(reduction='mean')   
        optimizer = torch.optim.SGD(self.parameters(), lr=args.lr_rate)
        batch_length=len(mini_batch)
        sa, action_value= zip(*mini_batch)
        sa=list(sa)
        action_value=list(action_value)
        action_value=torch.cat(action_value)
        #next_states = np.array(next_states)
        pred=[]
        #pred=torch.zeros(batch_length,device=args.device)


        pred=self(torch.cat(sa))
        #for epoch in range(batch_length):
        #    pred.append(self(torch.tensor(sa[epoch],requires_grad=False).to(args.device)))
            #print(f"TN sa:{sa[epoch]}")
            #print(f"TN pred:{pred}")
            #print(f"TN action_value{action_value[epoch]}")
            #print(f"TN av tensor{torch.tensor((action_value[epoch],)).to(args.device)}")
        pred=torch.cat(pred)
        pred.to(args.device)
        #loss = criterion(pred, torch.tensor(action_value).to(args.device))  
        loss = criterion(pred, action_value.to(args.device))  
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.parameters(), 100)
        optimizer.step()
        losses.append(float(f"{loss}"))
        return 1
    """
    while flag:
        time.sleep(0.02)
        while in_Pipe.empty():
            #print(f"stop_flag in TN{stop_flag.value}")
            if stop_flag.value:
                flag = 0
                break
            time.sleep(0.01)
        if not flag:
            plt.plot(losses)
            #plt.ylim(1)
            plt.savefig("loss plot.png")
            plt.close()
            break
        try:
            mini_batch = in_Pipe.get()  #mini_batch=[((row,column,action),(action_values))]
        except EOFError:
            return 0
        timer_start = time.time()
        #if not net.train(mini_batch):
        #    break
        losses = train(net, mini_batch, losses)
        period = time.time() - timer_start
        print(f"train time {period}")
        #print("training happended")
        #print(f"is copy queue full: {queue_copy.full()}")
        if queue_copy.full():
            queue_copy.get()
        queue_copy.put(copy.deepcopy(net).to("cpu"), block=0)
        #net.to(args.device)
    #export_net.put(copy.deepcopy(net))
    net.to("cpu")
    export_net.put(net)
    plt.ylim(0, 1)
    plt.plot(losses)
    plt.savefig("loss plot.png")
    plt.close()
    print("NT_Processing over")


def Active_Net_Processing(queue_sa, queue_out, queue_copy, stop_flag):
    #net=classes.QN(3,1)
    net = classes.QN(2, 9)
    net.to(args.device)
    #print(args.device)
    flag = True
    batch = []
    round_cnt = 0
    while flag:
        batch = []
        got = 0
        #print(4)
        while not got:
            try:
                sa_pair = queue_sa.get(0, 0.02)
                got = 1
            except queue.Empty:
                if stop_flag.value:
                    print("AN over")
                    return
                time.sleep(0.02)
                continue
        for sa in sa_pair:
            #print(f"sa: {sa}")
            #t1=time.time()
            out = net(sa).detach()
            #print("net cal take:",time.time()-t1)
            #out=net(torch.tensor(sa).to(device=args.device)).detach()
            #print(f"net outcome: {out}")
            batch.append(out)
        #print(f"sa: {sa}")
        """out=net(torch.tensor(sa_pair).to(device=args.device).detach()).detach()
        queue_out.put(out)
        #print(f"net outcome: {out}")
        #"""
        #print(queue_out.full())
        queue_out.put(batch)
        #print("PN Put")
        round_cnt += 1
        if round_cnt > args.update_round:
            round_cnt = 0
            try:
                net = queue_copy.get(0, 0.2)
                net.to(args.device)
                print("net upgraded")
            except EOFError:
                queue_out.close()
                break
            #queue_copy=multiprocessing.Queue()
            except queue.Empty:
                print("Empty raised")
                if stop_flag.value:
                    print("AN over")
                    return
                continue
    print("AN over")
    return


def Sample_Processing(space, max_row, max_column, queue_out, handle, queue_copy, stop_flag,
                      treading_num=args.sample_threading_num):
    classes.space = space
    args.max_column = max_column
    args.max_row = max_row
    state_num = len(space)
    global cache
    cache = []
    global cache_init
    cache_init = 1
    global upgraded
    upgraded = 0

    def sub_processing():
        global cache_init
        global cache
        global upgraded
        #print(3)
        #manager=multiprocessing.Manager()
        #Net_in_queue=manager.Queue(10)
        #Net_out_queue=manager.Queue(10)
        Net_in_queue = multiprocessing.Queue(2)
        Net_out_queue = multiprocessing.Queue(2)
        net_processing = multiprocessing.Process(target=Active_Net_Processing,
                                                 args=(Net_in_queue, Net_out_queue, queue_copy, stop_flag))
        net_processing.start()
        #print(5)
        epsilon_policy = classes.policy(args.epsilon_greedy(1))
        ##print(5)
        ##print(f"SP:{stop_flag.value}")
        ##print(f"AN:{stop_flag.value}")
        ##print(f"TN:{args.stop_flag_TN.value}")
        while not stop_flag.value:
            out = []
            #print(7)
            #print(f"cache length in thread {len(cache)}")
            init_state = random.randint(0, state_num - 1)
            #print(f"init_state {init_state}")
            #print(f"state_num {len(classes.space)}")
            action_values = []
            next_states = []
            eps = classes.episode(init_state, epsilon_policy, args.episode_length)
            for ind in range(len(eps.track) - 2, -1, -1):
                s, a = eps.track[ind]
                #x=s%args.max_column
                #y=s//args.max_column
                #next_s_n=eps.track[ind+1][0]
                #next_s=classes.space[eps.track[ind+1][0]]
                #next_x=next_s_n%args.max_column
                #next_y=next_s_n//args.max_column
                #sa_pairs=list(map(lambda a:(np.float32(next_x),np.float32(next_y),np.float32(a)),range(len(next_s.actions))))

                for a in range(len(classes.space[s].actions)):
                    next_s = next(classes.space[s].actions[a])
                    next_x = next_s % args.max_column
                    next_y = next_s // args.max_column
                    next_states.append((np.float32(next_x), np.float32(next_y)))
            while Net_in_queue.full():
                #print("full")
                if stop_flag.value:
                    print("sample sub processing over")
                    net_processing.join()
                    return
                time.sleep(0.1)
            Net_in_queue.put(torch.tensor(next_states).to(args.device))
            while Net_out_queue.empty():
                #print("empty")
                if stop_flag.value:
                    net_processing.join()
                    print("sample sub processing over")
                    return
                time.sleep(0.1)
            next_action_values = copy.deepcopy(list(Net_out_queue.get()))
            for ind in range(len(eps.track) - 1):
                action_values = []
                s = eps.track[ind][0]
                x = s % args.max_column
                y = s // args.max_column
                for a in range(len(classes.space[s].actions) - 1, -1, -1):
                    values = list(next_action_values.pop())
                    #print(values)
                    this_action_value = space[s].actions[a].action_reward + args.gamma * max(values).item()
                    #this_action_value=(this_action_value+space[s].actions[a].action_value)/2
                    this_action_value = space[s].actions[a].action_value
                    action_values = [this_action_value, ] + action_values
                #print(torch.tensor(action_values).detach())
                #out.append(((np.float32(x),np.float32(y)),torch.tensor(action_values).detach()))
                out.append(((np.float32(x), np.float32(y)), torch.tensor(
                    [np.float32(space[s].actions[a].action_value) for a in range(len(space[s].actions))]).detach()))
                #print(f"the reward of action{s}_{a}:{space[s].actions[a].action_reward}")
            print(next_action_values)
            if cache_init:
                print("init round")
                cache += random.sample(out, len(out))
                #print(cache)
                if len(cache) > args.AN_cache_min_length:
                    cache_init = 0
                continue

            #print(f"cache:{len(cache)}")
            #print(f"out:{len(out)}")

            position = random.sample(range(len(cache)), len(out))
            for i in range(len(out)):
                cache[position[i]] = out[i]
            upgraded = 1
            print("upgraded")
        net_processing.join()
        print("sample sub processing over")

    pool = ThreadPoolExecutor(treading_num)
    threadings = []
    event = threading.Event()
    for i in range(treading_num):
        #print("threading add")
        threadings.append(threading.Thread(target=sub_processing))
        threadings[i].start()
        time.sleep(5)
    #    threadings.append(pool.submit(sub_processing,i))
    while 1:
        try:
            if not handle.get() or stop_flag.value:
                handle.close()
                queue_out.close()
                break
        except EOFError:
            handle.close()
            queue_out.close()
            break
        while len(cache) < args.batch_length:
            #print(f"cache length{len(cache)}")
            event.wait(0.5)
        while not upgraded:
            if stop_flag:
                break
            event.wait(0.5)
        upgraded = 0
        #print("is queue out full:",queue_out.full())
        queue_out.put(random.sample(cache, args.batch_length))
        #print("data sent")
    for i in threadings:
        i.join()
    print("sample processing over")
