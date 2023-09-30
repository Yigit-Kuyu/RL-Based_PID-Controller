import numpy as np
import matplotlib.pyplot as plt

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
    """
    Path tracking controller using Stanley controller for lateral control and simple proportional
    controller for lateral longitudinal control.
    See: https://ieeexplore.ieee.org/document/4282788
    """

    def __init__(self, path_ref, v_ref, params):
        self.path_ref = path_ref
        self.v_ref = v_ref
        self.wheelbase = params["wheelbase"]
        self.k = params["k"]
        self.k_soft = params["k_soft"]
        self.k_p = params["k_p"]

    def steering_angle(self, state,last_target_indx):
        """
        Calculate control action for steering angle.
        """
        
        '''
        pos = state[:2]
        yaw, v = state[2:4]
        pos_fw = pos + self.wheelbase * np.array([np.cos(yaw), np.sin(yaw)])  # Position front wheel

        # Find point on path nearest to front wheel
        dists = np.sum((pos_fw - self.path_ref[:, :2])**2, axis=1)
        id_nearest = np.argmin(dists)
        path_point_nearest = self.path_ref[id_nearest]  # [x, y, yaw] of nearest path point

        # Yaw error term
        # TODO: Normalize angles correctly
        # See: https://stackoverflow.com/a/32266181
        yaw_error = path_point_nearest[2] - yaw 

        # Cross-track error to nearest point on path
        e_ct = np.sqrt(dists[id_nearest])

        # Cross-track error term has to be negative if we are on left side
        # of path and positive if we are on right side of path
        vehicle_normal = np.array([np.sin(yaw), -np.cos(yaw)])
        nearest_p_to_front_wheel = pos_fw - path_point_nearest[:2]
        dir_ct = np.sign(np.dot(vehicle_normal, nearest_p_to_front_wheel))

        # Final steering angle output
        steering_angle = yaw_error + dir_ct * np.arctan2(self.k * e_ct, (v + self.k_soft))

        return steering_angle

        '''

        pos = state[:2]
        yaw, v = state[2:4]
        fx = pos[0] + self.wheelbase * np.cos(yaw) # front x
        fy =pos[1] + self.wheelbase * np.sin(yaw)  # front y
        pos_fa = pos + self.wheelbase * np.array([np.cos(yaw), np.sin(yaw)])  # Position front axle

        dx = [fx - icx for icx in self.path_ref[:,0]]
        dy = [fy - icy for icy in self.path_ref[:,1]]
        d = np.hypot(dx, dy)
        current_target_indx = np.argmin(d)
        
        if last_target_indx >= current_target_indx:
            current_target_indx = last_target_indx
       
        
        # Find point on path nearest to front wheel
        dists = np.sum((pos_fa - self.path_ref[:, :2])**2, axis=1)
        # Search nearest point index
        id_nearest = np.argmin(dists)
        #path_point_nearest = self.path_ref[id_nearest]  # [x, y, yaw] of nearest path point
        path_point_nearest = self.path_ref[current_target_indx]


        nearest_p_to_front_wheel = pos_fa - path_point_nearest[:2]
        
        front_axle_vec_rot_90 = [-np.cos(yaw + np.pi / 2.0),-np.sin(yaw + np.pi / 2.0)]

        error_front_axle_notused=np.dot([dx[current_target_indx], dy[current_target_indx]], front_axle_vec_rot_90)
       
        error_front_axle = np.dot(nearest_p_to_front_wheel, front_axle_vec_rot_90)
        
       
        

        # Yaw error term
        #yaw_error = np.mod(path_point_nearest[2],2.0*np.pi) - yaw 
        yaw_error = path_point_nearest[2] - yaw #yaw_ref- yaw_car
        #yaw_error = np.mod(yaw_error,2.0*np.pi)
        yaw_error=normalize_angle(yaw_error)


        # Cross-track error to nearest point on path
        e_ct = np.sqrt(dists[id_nearest])

        #curvature_x=calculate_curvature(self.path_ref[:, :2], id_nearest)
        #direction_x =calculate_direction(self.path_ref[:, :2], id_nearest)
        #sign_of_cos, sign_of_sin=determine_signs_of_cos_and_sin(curvature_x, direction_x)

      

        # Cross-track error term has to be negative if we are on left side
        # of path and positive if we are on right side of path
        #vehicle_normal = np.array([sign_of_cos * np.cos(yaw), sign_of_sin * np.sin(yaw)]) # perpendicular to the bicycle's direction of travel
        #error_veh_normal=np.dot(vehicle_normal, nearest_p_to_front_wheel)
        #dir_ct = np.sign(np.dot(vehicle_normal, nearest_p_to_front_wheel))

        # Final steering angle output
        #steering_angle = yaw_error + dir_ct * np.arctan2(self.k * e_ct, (v + self.k_soft))
        #steering_angle = yaw_error + dir_ct * np.arctan2(self.k * e_ct, (v + self.k_soft))
        #steering_angle = yaw_error + np.arctan2(self.k * error_veh_normal, v)
        steering_angle = yaw_error + np.arctan2(self.k * error_front_axle_notused,v)
        
        print('steering angle: ', steering_angle)
        return steering_angle, current_target_indx

        

    def acceleration(self, state):
        """
        Calculate control action for acceleration.
        """
        v = state[3]
        acceleration = (self.v_ref - v) * self.k_p

        return acceleration

    def compute_controls(self, state, target_indx):
        v= self.acceleration(state)
        str_ang, target_index=self.steering_angle(state, target_indx)
        
        return [v, str_ang, target_index]



# Note: Position of vehicle is defined as the position of the rear wheel
class BicycleModel1WS:
    """
    Class representing bicycle model with front wheel steering.
    """

    def __init__(self, delta_max=np.radians(30), L=2):
        self.delta_max = delta_max # [rad] max steering angle
        self.L = L # [m] Wheel base of vehicle

    def kinematics(self, state, inputs, dt):
        """
        Kinematic model for Scipy's solve_ivp function.
        Note that the position [x, y] and velocity v of the bicycle correspond 
        to the position and velocity of the rear wheel.
        :param t: continuous time
        :param state: [x, y, yaw, v] state
        :param inputs: [a, delta] input
        """
        yaw, v = state[2:4]
        x=state[0]
        y=state[1]
        a, delta, target_index = inputs
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
    dt = 0.1  # sampling time, 0.5
    t_max = 1000  # Max simulation time
    n_steps = int(t_max / dt)
    t_vec = np.linspace(0, t_max, n_steps)

    # Controller input
    path_ref = path_sin()
    v_ref = 30.0 / 3.6 # [m/s]

    # Bicycle model
    wheelbase = 2
    delta_max = np.radians(30) # [rad] max steering angle
    model = BicycleModel1WS(delta_max, wheelbase)

    # Controller
    params = {"wheelbase": wheelbase,
              "k": 0.5, # control gain
              "k_soft": 2,
              "k_p": 1}
    controller = StanleyController(path_ref, v_ref, params)

    # Initialize histories for time, state and inputs
    t_hist = []
    state_hist = []
    inputs_hist = []
    target_index=0

    # Initial state and input
    state = np.array([0, 0, np.radians(50), 0.0]) #x,y, steering, velocity
    inputs = controller.compute_controls(state,target_index)
    

    # Simulate
    #for t in t_vec:
    it=0
    target_index=0
    last_idx=len(path_ref)-1
    while True:
        state_hist.append(state)
        print('iteration: ', it)
        print('state.x: ', state[0], 'state.y: ', state[1])
        it+=1
        state_new = model.kinematics(state, inputs, dt)
        
        #t_span = (t, t + dt)
        #t_eval = np.linspace(*t_span, 5)

        
        inputs = controller.compute_controls(state_new,inputs[2]) #v, steering angle, target index
        state=state_new

        
        if it>1000:
            print('Dur')
            #break
        
        if last_idx <= inputs[2]:
            print('Dur')
            break

        # Store state, inputs and time for analysis
        inputs_hist.append(inputs)
        #t_hist.append(sol.t)

        
        if reached_target(state, path_ref[-1, :2], wheelbase):
            print("Reached end of path.")
            break

    #state_hist = np.concatenate(state_hist, axis=1)
    #inputs_hist = np.vstack(inputs_hist).T
    #t_hist = np.concatenate(t_hist)

    plot_trajectory(state_hist, path_ref, wheelbase)
    # plot_state(t_hist, state_hist, v_ref=v_ref)

def reached_target(state, target, wheelbase):
    pos = state[:2]
    yaw = state[2]
    pos_fw = pos + wheelbase * np.array([np.cos(yaw), np.sin(yaw)])  # Position front wheel
    dist_to_target = np.sqrt(np.sum((pos_fw - target)**2))

    return dist_to_target < 0.05

def plot_trajectory(state, path_ref, L):
    x=[s[0] for s in state]
    y=[s[1] for s in state]
    yaw=[s[2] for s in state]
    #x, y, yaw = state[:,:3]
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

def plot_state(t, state, x_ref=None, y_ref=None, v_ref=None):
    fig, axs = plt.subplots(1, 4, figsize=(15, 5)) 
    axs[0].plot(t, state[0], 'k')
    axs[0].grid()
    axs[0].set_xlabel('t [s]')
    axs[0].set_ylabel('x [m]')

    axs[1].plot(t, state[1], 'k')
    axs[1].grid()
    axs[1].set_xlabel('t [s]')
    axs[1].set_ylabel('y [m/s]')

    axs[2].plot(t, np.rad2deg(state[2]) % (360), 'k')
    axs[2].grid()
    axs[2].set_xlabel('t [s]')
    axs[2].set_ylabel('yaw [deg]')

    v_ref_vec = v_ref * np.ones_like(t)
    axs[3].plot(t, state[3], 'k')
    axs[3].plot(t, v_ref_vec, 'r--')
    axs[3].grid()
    axs[3].set_xlabel('t [s]')
    axs[3].set_ylabel('speed [m/s]')

    fig.tight_layout()
    return fig


if __name__ == "__main__":
    simulate()
    plt.show()