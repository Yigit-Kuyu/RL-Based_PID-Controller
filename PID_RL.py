import numpy as np
import matplotlib.pyplot as plt
import time

def calculate_curvature(path, x):
  """Calculates the curvature of a reference path at a given x-position.

  Args:
    path: A reference path.
    x: The x-position of the vehicle.

  Returns:
    The curvature of the reference path at the given x-position.
  """

  # Calculate the derivatives of the reference path
  dx = np.diff(path[:, 0])
  dy = np.diff(path[:, 1])

  # Calculate the curvature of the reference path
  curvature = (dy / dx) / (1.0 + (dy / dx)**2)

  # Return the curvature at the given x-position
  return curvature[x]


def calculate_direction(path, x):
  """Calculates the direction of a reference path at a given x-position.

  Args:
    path: A reference path.
    x: The x-position of the vehicle.

  Returns:
    The direction of the reference path at the given x-position.
  """

  # Calculate the derivatives of the reference path
  dx = np.diff(path[:, 0])
  dy = np.diff(path[:, 1])

  # Calculate the direction of the reference path
  direction = np.arctan2(dy, dx)[x]

  # Return the direction
  return direction


def determine_signs_of_cos_and_sin(curvature, direction):
  """Determines the signs of cos and sin in the vehicle normal based on the curvature and direction of the reference path.

  Args:
    curvature: The curvature of the reference path.
    direction: The direction of the reference path.

  Returns:
    A tuple containing the signs of cos and sin in the vehicle normal.
  """

  if curvature > 0:
    sign_of_cos = 1
    sign_of_sin = -1
  else:
    sign_of_cos = -1
    sign_of_sin = 1

  if direction > np.pi / 2:
    sign_of_cos *= -1
    sign_of_sin *= -1

  return sign_of_cos, sign_of_sin



class StanleyController:


    def __init__(self, path_ref, v_ref, params):
        self.path_ref = path_ref
        self.v_ref = v_ref
        self.wheelbase = params["wheelbase"]
        self.k = params["k"]
        self.k_p = params["k_p"]
        self.k_i = params["k_i"]
        self.k_d = params["k_d"]

    def steering_angle(self, state,last_target_indx):
        calculation_option=1 # both options give same result
      
        pos = state[:2]
        yaw, v = state[2:4]
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
            return steering_angle, current_target_indx,e_ct
        
        else: # alternative option
            # second option for error calculation
            nearest_p_to_front_wheel = pos_fa - path_point_nearest[:2]  
            front_axle_vec_rot_90 = [-np.cos(yaw + np.pi / 2.0),-np.sin(yaw + np.pi / 2.0)]
            error_front_axle = np.dot(nearest_p_to_front_wheel, front_axle_vec_rot_90)
            # second option for calculation steering angle
            steering_angle = yaw_error + np.arctan2(self.k * error_front_axle,v)
            
            print('steering angle: ', steering_angle)
            return steering_angle, current_target_indx,error_front_axle 
        
        

        

    def PID(self, state,dt,errors): # PID for acceleration
        
        v = state[3]
        #p=(self.v_ref - v) * self.k_p
        p=self.k_p*errors[1]
        i=self.k_i*sum(errors)*dt
        d=self.k_d*(errors[1]-errors[0])/dt
        acceleration = p+i+d

        return acceleration

    def calculate_vel_steer(self, state, target_indx,dt,errors):
        v= self.PID(state,dt,errors)
        str_ang, target_index, error=self.steering_angle(state, target_indx)
        
        return [v, str_ang, target_index, error]




class BicycleModel_Rear: # Rear axle model (Position of vehicle is defined as the position of the rear wheel)

    def __init__(self, delta_max=np.radians(30), L=2):
        self.delta_max = delta_max # [rad] max steering angle
        self.L = L # [m] Wheel base of vehicle

    def kinematics(self, state, inputs, dt):
        yaw, v = state[2:4]
        x=state[0]
        y=state[1]
        a, delta, target_index,  current_error= inputs
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

        state_new=[x_new, y_new, yaw_new, v_new]

        
        
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

def simulate():
    dt = 0.1  # sampling time
    t_max = 1000  # Max simulation time
    n_steps = int(t_max / dt)
    t_vec = np.linspace(0, t_max, n_steps)

    # Controller input
    path_ref = path_sin()
    v_ref = 30.0 / 3.6 # [m/s]

    # Bicycle model
    wheelbase = 2
    delta_max = np.radians(30) # [rad] max steering angle
    model = BicycleModel_Rear(delta_max, wheelbase)

    # Initialization
    params = {"wheelbase": wheelbase,
              "k": 0.5, # control gain
              "k_p": 1,
              "k_i": 1e-4,
              "k_d": 1e-4}
    controller = StanleyController(path_ref, v_ref, params)

    
   
    state_hist = []
    inputs_hist = []
    target_index=0
    

    # Initial state and input
    dt=1
    errors=[1e+6, 1e+6] # [previous error, current error]
    state = np.array([0, 0, np.radians(50), 0.0]) #x,y, steering, velocity,
    inputs = controller.calculate_vel_steer(state,target_index,dt,errors) #inputs: v, steering angle, target index, current error
    errors[1]=inputs[-1]
    
    it=0
    last_idx=len(path_ref)-1
    animate=1
    timenow=time.time()
    while True:
        time_previous=timenow
        state_hist.append(state)
        print('iteration: ', it)
        print('state.x: ', state[0], 'state.y: ', state[1])
        it+=1
        state_new = model.kinematics(state, inputs, dt)
        timenow=time.time()
        dt=timenow-time_previous # change in time
        inputs = controller.calculate_vel_steer(state_new,inputs[2],dt,errors) #inputs: v, steering angle, target index, current error
        errors[0]=errors[1]
        errors[1]=inputs[-1]
        state=state_new
        timestamps = np.arange(0, 3, 0.1)
        

        if last_idx <= inputs[2]:
            print('stop')
            break

        
        inputs_hist.append(inputs)
        

        
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
            plt.title("Speed[km/h]:" + str(state_new[-1] * 3.6)[:4])
            plt.pause(0.001)

    plot_trajectory(state_hist, path_ref, wheelbase)
    



def plot_trajectory(state, path_ref, L):
    x=[s[0] for s in state]
    y=[s[1] for s in state]
    yaw=[s[2] for s in state]
    fig, ax = plt.subplots(1, 1, figsize=(8, 8)) 
    ax.plot(x, y, 'k', label="rear wheel")
    ax.plot(x + L * np.cos(yaw), y + L * np.sin(yaw), 'k-.', label="front wheel")
    ax.plot(path_ref[:, 0], path_ref[:, 1], 'r--', label="path ref")
    ax.legend()
    ax.axis('equal')
    ax.grid()
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')

    return fig



if __name__ == "__main__":
    simulate()
    plt.show()