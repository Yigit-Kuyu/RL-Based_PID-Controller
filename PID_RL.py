import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import copy
import time
from collections import deque
from torch.distributions.normal import Normal
#import matplotlib as plt
import matplotlib.pyplot as plt



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")





# Use Xavier initialization for the weights and initializes the biases to zero for linear layers.
# It sets the weights to values drawn from a Gaussian distribution with mean 0 and variance
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)



class ValueNetwork(nn.Module): # state-Value network
    def __init__(self, num_inputs, hidden_dim):
        super(ValueNetwork, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_) # Optional


    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


# stocastic policy
class Actor(nn.Module):
    def __init__(self, state_size, action_size, hidden_dim, high, low):
        super(Actor, self).__init__()
        self.linear1 = nn.Linear(state_size, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        
        self.mean = nn.Linear(hidden_dim, action_size)
        self.log_std = nn.Linear(hidden_dim, action_size)
        
        self.high = torch.tensor(high).to(device)
        self.low = torch.tensor(low).to(device)
        
        self.apply(weights_init_) # Optional
        
        # Action rescaling
        diff_scale=list(set(high) - set(low))
        sum_scale=list(map(sum, zip(high, low)))
        self.action_scale = torch.FloatTensor(list(map(lambda x: x/2.0, diff_scale))).to(device)
        self.action_bias = torch.FloatTensor(list(map(lambda x: x/2.0, sum_scale))).to(device)
    
    def forward(self, state):
        log_std_min=-20
        log_std_max=2
        x = state
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        m = self.mean(x)
        s = self.log_std(x)
        s = torch.clamp(s, min = log_std_min, max = log_std_max)
        return m, s
    
    def sample(self, state):
        noise=1e-6
        m, s = self.forward(state) 
        std = s.exp()
        normal = Normal(m, std)
        
        
        ## Reparameterization (https://spinningup.openai.com/en/latest/algorithms/sac.html)
        # There are two sample functions in normal distributions one gives you normal sample ( .sample() ),
        # other one gives you a sample + some noise ( .rsample() )
        a = normal.rsample() # This is for the reparamitization
        tanh = torch.tanh(a)
        action = tanh * self.action_scale + self.action_bias
        
        logp = normal.log_prob(a)
        # Comes from the appendix C of the original paper for scaling of the action:
        logp =logp-torch.log(self.action_scale * (1 - tanh.pow(2)) + noise)
        logp = logp.sum(1, keepdim=True)
        
        return action, logp


# Action-Value
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(Critic, self).__init__()

        # Critic-1: Q1 
        self.linear1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        # Critic-2: Q2 
        self.linear4 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_) # OpReplayMemorytional


    def forward(self, state, action):
        state_action = torch.cat([state, action], 1)

        q1 = F.relu(self.linear1(state_action))
        q1 = F.relu(self.linear2(q1))
        q1 = self.linear3(q1)

        q2 = F.relu(self.linear4(state_action))
        q2 = F.relu(self.linear5(q2))
        q2 = self.linear6(q2)
        return q1, q2
    
# Buffer
class ReplayMemory:
    def __init__(self, memory_capacity, batch_size):
        self.memory_capacity = memory_capacity
        self.batch_size = batch_size
        self.memory = []
        self.position = 0

    def push(self, element):
        if len(self.memory) < self.memory_capacity:
            self.memory.append(None)
        self.memory[self.position] = element
        self.position = (self.position + 1) % self.memory_capacity

    def sample(self):
        return list(zip(*random.sample(self.memory, self.batch_size)))

    def __len__(self):
        return len(self.memory)


class Sac_agent:
    def __init__(self, state_size, action_size, hidden_dim, high, low, memory_capacity, batch_size,
                 gamma, tau,num_updates, policy_freq, alpha):
        
         # Actor Network 
        self.actor = Actor(state_size, action_size,hidden_dim, high, low).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)
        

        # Critic Network and Target Network
        self.critic = Critic(state_size, action_size, hidden_dim).to(device)   
        
        self.critic_target = Critic(state_size, action_size, hidden_dim).to(device)        
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-4)
        
        # copy weights
        self.hard_update(self.critic_target, self.critic)
        
        # Value network and Target Network
        self.value = ValueNetwork(state_size, hidden_dim).to(device)
        self.value_optim =optim.Adam(self.value.parameters(), lr=1e-4)
        self.target_value = ValueNetwork(state_size, hidden_dim).to(device)
        
        # copy weights
        self.hard_update(self.target_value, self.value)
        
        self.state_size = state_size
        self.action_size = action_size
        
        self.memory = ReplayMemory(memory_capacity, batch_size)
        self.gamma = gamma
        self.tau = tau
        self.num_updates = num_updates
        self.iters = 0
        self.policy_freq=policy_freq
        
        ## For Dynamic Adjustment of the Parameter alpha (entropy coefficient) according to Gaussion policy (stochastic):
        self.target_entropy = -float(self.action_size) # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2, -11 for Reacher-v2)
        self.log_alpha = torch.zeros(1, requires_grad=True, device = device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=1e-4)
        self.alpha = alpha
        
        
    def hard_update(self, target, network):
        for target_param, param in zip(target.parameters(), network.parameters()):
            target_param.data.copy_(param.data)
            
    def soft_update(self, target, network):
        for target_param, param in zip(target.parameters(), network.parameters()):
            target_param.data.copy_(self.tau*param.data + (1-self.tau)*target_param.data)
            
    def learn(self, batch):
        for _ in range(self.num_updates):                
            state, action, reward, next_state, mask = batch

            state = torch.tensor(state).to(device).float()
            next_state = torch.tensor(next_state).to(device).float()
            reward = torch.tensor(reward).to(device).float().unsqueeze(1)
            action = torch.tensor(action).to(device).float()
            mask = torch.tensor(mask).to(device).int().unsqueeze(1)
                         
            value_current=self.value(state)
            value_next=self.target_value(next_state)
            act_next, logp_next = self.actor.sample(next_state)
                
            ## Compute targets
            Q_target_main = reward + self.gamma*mask*value_next # Eq.8 of the original paper

            ## Update Value Network
            Q_target1, Q_target2 = self.critic_target(next_state, act_next) 
            min_Q = torch.min(Q_target1, Q_target2)
            value_difference = min_Q - logp_next # substract min Q value from the policy's log probability of slelecting that action
            value_loss = 0.5 * F.mse_loss(value_current, value_difference) # Eq.5 from the paper
            # Gradient steps 
            self.value_optim.zero_grad()
            value_loss.backward(retain_graph=True)
            self.value_optim.step()
            
            ## Update Critic Network       
            critic_1, critic_2 = self.critic(state, action)
            critic_loss1 = 0.5*F.mse_loss(critic_1, Q_target_main) # Eq. 7 of the original paper
            critic_loss2 = 0.5* F.mse_loss(critic_2, Q_target_main) # Eq. 7 of the original paper
            total_critic_loss=critic_loss1+ critic_loss2 
            # Gradient steps
            self.critic_optimizer.zero_grad()
            total_critic_loss.backward() 
            self.critic_optimizer.step() 

            ## Update Actor Network with Entropy Regularized (look at the link for entropy regularization)
            act_pi, log_pi = self.actor.sample(state) # Reparameterize sampling
            Q1_pi, Q2_pi = self.critic(state, act_pi)
            min_Q_pi = torch.min(Q1_pi, Q2_pi)
            actor_loss =-(min_Q_pi-self.alpha*log_pi ).mean() # For minimization
            # Gradient steps
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            ## Dynamic adjustment of the Entropy Parameter alpha (look at the link for entropy regularization)
            alpha_loss = (-self.log_alpha * (log_pi.detach()) - self.log_alpha* self.target_entropy).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp()
            
            ## Soft Update Target Networks using Polyak Averaging
            if (self.iters % self.policy_freq == 0):         
                self.soft_update(self.critic_target, self.critic)
                self.soft_update(self.target_value, self.value)
        
    def act(self, state):
        state =  torch.tensor(state).unsqueeze(0).to(device).float()
        action, logp = self.actor.sample(state)
        return action.cpu().data.numpy()[0]
    
    def step(self):
        self.learn(self.memory.sample())
        
    def save(self):
        torch.save(self.actor.state_dict(), "pen_actor.pkl")
        torch.save(self.critic.state_dict(), "pen_critic.pkl")


class BicycleModel_Rear: # Rear axle model (Position of vehicle is defined as the position of the rear wheel)

    def __init__(self, delta_max=np.radians(30), L=2):
        self.delta_max = delta_max # [rad] max steering angle
        self.L = L # [m] Wheel base of vehicle

    def kinematics(self, state, inputs, dt):
        yaw= state[2]
        v=inputs[0]
        x=state[0]
        y=state[1]
        a, delta, _,  _,_,_= inputs #inputs: velocity, steering angle, CTE, yaw error, target index, current velocity error
        delta = np.clip(delta, -self.delta_max, self.delta_max) # steering angle

        dx = v * np.cos(yaw)*dt
        dy = v * np.sin(yaw)*dt

        x_new=x+dx
        y_new=y+dy

        
        yaw_new = v / self.L * np.tan(delta)*dt
        #yaw_new = np.mod(yaw_new,2.0*np.pi) # Normalize yaw at 2pi
        yaw_new=yaw+yaw_new
        yaw_new=normalize_angle(yaw_new)

        #rear_x=x_new-((self.L / 2) * np.cos(yaw_new))
        #rear_y=y_new-((self.L / 2) * np.sin(yaw_new))
        
        v_new = a*dt
        v_new=v+v_new

        #v_new=v

        state_new=[x_new, y_new, yaw_new]

        
        
        return state_new


def normalize_angle(angle):

    while angle > np.pi:
        angle -= 2.0 * np.pi

    while angle < -np.pi:
        angle += 2.0 * np.pi

    return angle


def path_sin():
    """Generate sinusoidal path along global y axis."""
    s = np.linspace(0, 4 * np.pi, 1000)
    pos = np.array([2 * np.sin(1 * s), 2 * s]).T

    # Calculate yaw angle of path points
    segments = np.diff(pos, axis=0)
    yaw = np.arctan2(segments[:, 1], segments[:, 0])
    path = np.hstack((pos[:-1], yaw.reshape(-1, 1)))

    return path


class StanleyController:


    def __init__(self, path_ref, v_ref, params):
        self.path_ref = path_ref
        self.v_ref = v_ref
        self.wheelbase = 2
        self.k = 0.5 # control gain
        self.k_p = params[0]
        self.k_i = params[1]
        self.k_d = params[2]

    def steering_angle(self, state,v, last_target_indx):
        calculation_option=1 # both options give same result
      
        pos = state[:2]
        yaw= state[2]
        #fx = pos[0] + self.wheelbase * np.cos(yaw) # front x
        #fy =pos[1] + self.wheelbase * np.sin(yaw)  # front y
        pos_fa = pos + self.wheelbase * np.array([np.cos(yaw), np.sin(yaw)])  # Position front axle

        # Find point on path nearest to front wheel
        dists = np.sum((pos_fa - self.path_ref[:, :2])**2, axis=1)
        # Search nearest point index
        current_target_indx = np.argmin(dists)
        
        if last_target_indx >= current_target_indx:
            current_target_indx = last_target_indx
       
        
        path_point_nearest = self.path_ref[current_target_indx]  # [x, y, yaw] of nearest path point

        # Yaw error term
        yaw_error = path_point_nearest[2] - yaw #yaw_ref- yaw_car
        #yaw_error = np.mod(yaw_error,2.0*np.pi)
        yaw_error=normalize_angle(yaw_error)

        if calculation_option==1: # main option
            # Cross-track error to nearest point on path
            e_ct = np.sqrt(dists[current_target_indx])
            # Calculate steering angle
            steering_angle = yaw_error + np.arctan2(self.k * e_ct,v)

            print('steering angle: ', steering_angle)
            return steering_angle, current_target_indx, e_ct, yaw_error
        
        else: # alternative option
            # second option for error calculation
            nearest_p_to_front_wheel = pos_fa - path_point_nearest[:2]  
            front_axle_vec_rot_90 = [-np.cos(yaw + np.pi / 2.0),-np.sin(yaw + np.pi / 2.0)]
            error_front_axle = np.dot(nearest_p_to_front_wheel, front_axle_vec_rot_90)
            # second option for calculation steering angle
            steering_angle = yaw_error + np.arctan2(self.k * error_front_axle,v)
            
            print('steering angle: ', steering_angle)
            return steering_angle, current_target_indx, error_front_axle, yaw_error
        
        

    def PID(self, action,dt,errors,total_errors): # PID for acceleration
        
        
        self.k_p=action[0]
        self.k_i=action[1]
        self.k_d=action[2]
        
        
        p=self.k_p*errors[1]
        i=self.k_i*total_errors*dt
        d=self.k_d*(errors[1]-errors[0])/dt
        print('PID P:', p, 'I: ',i, 'D: ',d)
        acceleration = p+i+d

        return acceleration

    def calculate_vel_steer(self, action, state, target_indx, dt,errors,total_errors):
        v= self.PID(action,dt,errors,total_errors)
        velocity_error=self.v_ref - v
        str_ang, target_index, e_ct, yaw_error=self.steering_angle(state, v, target_indx)
        print('velocity: ', v)
        return [v, str_ang, e_ct, yaw_error, target_index, velocity_error]
    
def reward_calculate(state_input):
        CTE=state_input[0]
        yaw_error=state_input[1]
        
        R=-1/2*(CTE+yaw_error)

        if CTE>2:
            done=True
        else:
            done=False

        return R, done

def sac(episodes):
    agent = Sac_agent(state_size = state_size, action_size = action_size, hidden_dim = hidden_dim, high = high, low = low, 
                  memory_capacity = memory_capacity, batch_size = batch_size, gamma = gamma, tau = tau, 
                  num_updates = num_updates, policy_freq =policy_freq, alpha = entropy_coef)
    time_start = time.time()
    reward_list = []
    avg_score_deque = deque(maxlen = 100)
    avg_scores_list = []
    mean_reward = -20000
    
    # Boundaries of the action size (P, I, D):
    #low = [0.5, 5e-1, 5e-6] 
    #high = [1, 1e-1, 5e-5]

    # Controller input
    path_ref = path_sin()
    v_ref = 30.0 / 3.6  # [m/s]
    wheelbase = 2
    delta_max = np.radians(30) # [rad] max steering angle
    dt_sampling = 0.1  # sampling time
    for i in range(episodes):
        #state = env.reset() 
        action=[1,1,1] # action    
        model_control = BicycleModel_Rear(delta_max, wheelbase)

        controller = StanleyController(path_ref, v_ref, action)
        
        # Initial state and input
        dt=1  # For PID
        total_velocity_errors=0
        action=[1,1,1] # P, I, D
        v=0
        target_index=0
        errors_velocity=[1e+1, 1e+1] # [previous velocity error, current velocity error]
        state_control = np.array([0, 0, np.radians(50)]) #state_control: x,y, yaw
        inputs = controller.calculate_vel_steer(action,state_control,target_index, dt,errors_velocity,total_velocity_errors) #inputs: velocity, steering angle, CTE, yaw error, target index, current velocity error
        state_RL=[inputs[2], inputs[3]] # RL state: CTE, yaw error
        total_velocity_errors+=inputs[-1]
        errors_velocity=[inputs[-1], inputs[-1]]
      
        state_hist = []
        inputs_hist = []
        it=0
        last_idx=len(path_ref)-1
        animate=1
    
        
        
        total_reward = 0
        done = False
        episode_steps = 0
        timenow=time.time()
        while not done or last_idx <= inputs[2]:
            time_previous=timenow
            episode_steps+=1 
            agent.iters=episode_steps
            if episode_steps < 10: # To increase exploration
                population_size=1
                dim=3
                for i in range(0, dim):
                    action[i] = random.uniform(low[i], high[i])  #action= P, I, D
                      
            else:
                action = agent.act(state_RL) # to sample the actions by Gaussian 
            
            it+=1
            state_control = model_control.kinematics(state_control, inputs, dt_sampling) #  state_control: x,y, yaw
            timenow=time.time()
            dt=timenow-time_previous # change in time
            print('action', action)
            inputs = controller.calculate_vel_steer(action,state_control,target_index, dt,errors_velocity,total_velocity_errors) #inputs: velocity, steering angle, CTE, yaw error, target index, current velocity error
            state_RL_next=[inputs[2], inputs[3]] # state RL: CTE, yaw error
            reward, done=reward_calculate(state_RL_next)
            print('state: ', state_RL_next, 'reward: ', reward)
            total_velocity_errors+=inputs[-1]
            errors_velocity[0]=errors_velocity[1]
            errors_velocity[1]=inputs[-1]
            

            if last_idx <= inputs[4]:
                 done=True
            
             # Ignore the "done" signal if it comes from hitting the time horizon.
            if episode_steps == max_episode_steps: # if the current episode has reached its maximum allowed steps
                mask = 1
            else:
                mask = float(not done)
            
            if (len(agent.memory) >= agent.memory.batch_size): 
                agent.step()
            
            

            total_reward += reward
            state_RL=state_RL_next

            state_hist.append(state_control)
            inputs_hist.append(inputs) 


            print(f"episode: {i+1}, steps:{episode_steps}, current reward: {total_reward}")
            agent.memory.push((state_RL, action, reward, state_RL_next, mask))
            #env.render()
            
            if animate:
                x=[s[0] for s in state_hist]
                y=[s[1] for s in state_hist]
                plt.cla()
                # for stopping simulation with the esc key.
                plt.gcf().canvas.mpl_connect('key_release_event',
                    lambda event: [exit(0) if event.key == 'escape' else None])
                plt.plot(path_ref[:, 0], path_ref[:, 1], ".r", label="reference")
                plt.plot(x, y, "-b", label="found traj")
                plt.plot(path_ref[last_idx,0], path_ref[last_idx,1], "xg", label="target")
                plt.axis("equal")
                plt.grid(True)
                plt.title("Speed[km/h]:" + str(inputs[0] * 3.6)[:4])
                plt.pause(0.001)

        reward_list.append(total_reward)
        avg_score_deque.append(total_reward)
        mean = np.mean(avg_score_deque)
        avg_scores_list.append(mean)
        
    plt.plot(episode_steps,reward_list)
    agent.save()
    print(f"episode: {i+1}, steps:{episode_steps}, current reward: {total_reward}, max reward: {np.max(reward_list)}")
    
                
    return reward_list, avg_scores_list



# Environment
action_size = 3 # P, I, D
print(f'size of each action = {action_size}')
state_size = 2 #  cte, yaw
print(f'size of state = {state_size}')
low = [0.5, 5e-1, 5e-6] 
high = [1, 1e-1, 5e-5]
print(f'low of each action = {low}')
print(f'high of each action = {high}')


max_episode_steps=1000 # if the agent does not reach "done" in "max_episode_steps", mask is 1
batch_size=20 # size that will be sampled from the replay memory that has maximum of "memory_capacity"
memory_capacity = 200 # 2000, maximum size of the memory
gamma = 0.99            
tau = 0.005               
num_of_train_episodes = 1500
num_updates = 1 # how many times you want to update the networks in each episode
policy_freq= 2 # lower value more probability to soft update,  policy frequency for soft update of the target network borrowed by TD3 algorithm
entropy_coef = 0.2 # For entropy regularization
num_of_test_episodes=200
hidden_dim=256

# Traning agent
reward, avg_reward = sac(num_of_train_episodes)



# Testing
new_env = make("Pendulum-v0")
best_actor = Actor(state_size, action_size, hidden_dim = hidden_dim, high = high, low = low)
best_actor.load_state_dict(torch.load("pen_actor.pkl"))        
best_actor.to(device) 
reward_test = []
for i in range(num_of_test_episodes):
    state = new_env.reset()
    local_reward = 0
    done = False
    while not done:
        state =  torch.tensor(state).to(device).float()
        action,logp = best_actor(state)        
        action = action.cpu().data.numpy()
        state, r, done, _ = new_env.step(action)
        local_reward += r
    reward_test.append(local_reward)


import plotly.graph_objects as go
x = np.array(range(len(reward_test)))
m = np.mean(reward_test)


fig = go.Figure()
fig.add_trace(go.Scatter(x=x, y=reward_test, name='test reward',
                                 line=dict(color="green", width=1)))

fig.add_trace(go.Scatter(x=x, y=[m]*len(reward_test), name='average reward',
                                 line=dict(color="red", width=1)))
    
fig.update_layout(title="SAC",
                           xaxis_title= "test",
                           yaxis_title= "reward")
fig.show()

print("average reward:", m)


