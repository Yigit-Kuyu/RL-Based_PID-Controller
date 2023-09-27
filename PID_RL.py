import numpy as np
import matplotlib.pyplot as plt


class StanleyController:


    def __init__(self, path_ref, v_ref, params):
        self.path_ref = path_ref
        self.v_ref = v_ref
        self.wheelbase = params["wheelbase"]
        self.k = params["k"]
        self.k_soft = params["k_soft"]
        self.k_p = params["k_p"]

    def steering_angle(self, state):
        """
        Calculate control action for steering angle.
        """
        pos = state[:2]
        yaw, v = state[2:4]
        pos_fw = pos + self.wheelbase * np.array([np.cos(yaw), np.sin(yaw)])  # Position front wheel

        # Find point on path nearest to front wheel
        dists = np.sum((pos_fw - self.path_ref[:, :2])**2, axis=1)
        id_nearest = np.argmin(dists)
        path_point_nearest = self.path_ref[id_nearest]  # [x, y, yaw] of nearest path point

        # Yaw error term
        yaw_error = path_point_nearest[2] - yaw 

        # Cross-track error to nearest point on path
        e_ct = np.sqrt(dists[id_nearest])

        # Cross-track error term has to be negative if we are on left side
        # of path and positive if we are on right side of path
        vehicle_normal = np.array([np.sin(yaw), -np.cos(yaw)])
        nearest_p_to_front_wheel = pos_fw - path_point_nearest[:2]
        dir_ct = np.sign(np.dot(vehicle_normal, nearest_p_to_front_wheel))

        # Final steering angle output
        #steering_angle = yaw_error + dir_ct * np.arctan2(self.k * e_ct, (v + self.k_soft))
        #steering_angle = yaw_error + dir_ct * self.k * e_ct
        steering_angle =  dir_ct * 0.002 * e_ct
        
        return steering_angle

    def acceleration(self, state):
        """
        Calculate control action for acceleration.
        """
        v = state[3]
        acceleration = (self.v_ref - v) * self.k_p

        return acceleration

    def compute_controls(self, state):
        return [self.acceleration(state), self.steering_angle(state)]



# Note: Position of vehicle is defined as the position of the rear wheel
class BicycleModel1WS:
    """
    Class representing bicycle model with front wheel steering.
    """

    def __init__(self, delta_max=np.radians(90), L=2):
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
        a, delta = inputs
        delta = np.clip(delta, -self.delta_max, self.delta_max)

        dx = v * np.cos(yaw)*dt
        dy = v * np.sin(yaw)*dt

        x_new=x+dx
        y_new=y+dx
        
        yaw_new = v / self.L * np.tan(delta)*dt
        yaw_new=yaw_new+yaw
        #yaw_new = np.mod(yaw_new,2.0*np.pi) # Wrap theta at 2pi
        
        v_new = a*dt

        state_new=[x_new, y_new, yaw_new, v_new]

        
        
        return state_new


class BicycleModel2WS:
    """
    Class representing bicycle model with front and back wheel steering.
    """

    def __init__(self, delta_max=np.radians(10), L=2):
        self.delta_max = delta_max # [rad] max steering angle
        self.L = L # [m] Wheel base of vehicle

    def kinematics(self, t, state, inputs):
        """
        Kinematic model for Scipy's solve_ivp function.
        Note that the position [x, y] and velocity v of the bicycle correspond 
        to the position and velocity of the rear wheel.
        :param t: continuous time
        :param state: [x, y, yaw, v] state
        :param inputs: [a, delta] input
        """
        yaw, v = state[2:4]
        a, delta = inputs
        delta = np.clip(delta, -self.delta_max, self.delta_max)

        x_dot = v * np.cos(yaw - delta)
        y_dot = v * np.sin(yaw - delta)
        yaw_dot = 2 * v / self.L * np.sin(delta)
        v_dot = a

        return np.array([x_dot, y_dot, yaw_dot, v_dot])




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
    v_ref = 0.2

    # Bicycle model
    wheelbase = 2
    delta_max = np.radians(30)
    model = BicycleModel1WS(delta_max, wheelbase)

    # Controller
    params = {"wheelbase": wheelbase,
              "k": 0.02,
              "k_soft": 2,
              "k_p": 0.2}
    controller = StanleyController(path_ref, v_ref, params)

    # Initialize histories for time, state and inputs
    t_hist = []
    state_hist = []
    inputs_hist = []

    # Initial state and input
    state = np.array([-4.0, -2.0, np.radians(-90), 0.0]) #x,y, steering, velocity
    inputs = controller.compute_controls(state)

    # Simulate
    #for t in t_vec:
    it=0
    while True:
        
        print('iteration: ', it)
        print('state.x: ', state[0], 'state.y: ', state[1])
        it+=1
        state_new = model.kinematics(state, inputs, dt)
        
        #t_span = (t, t + dt)
        #t_eval = np.linspace(*t_span, 5)

        if it==100000:
            print('Dur')
            break

        inputs = controller.compute_controls(state_new)
        state=state_new

        # Store state, inputs and time for analysis
        state_hist.append(state)
        inputs_hist.append(inputs)
        #t_hist.append(sol.t)

        
        if reached_target(state, path_ref[-1, :2], wheelbase):
            print("Reached end of path.")
            break

    #state_hist = np.concatenate(state_hist, axis=1)
    inputs_hist = np.vstack(inputs_hist).T
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
    x, y, yaw = state[:3]
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