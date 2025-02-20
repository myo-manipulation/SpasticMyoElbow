##################################################
# In this script, we integrated the spasticity model into the elbow model in the MyoSuite simulator.
# You can set the control mode to 'dual', 'joint', 'position', 'velocity' or 'none' to apply different external movements on the model.
# The trajectory of the movement can be set to minijerk_trajectory or ramphold_trajectory.
##################################################

import os
from typing import Any, Dict, List, Tuple
import csv
import time
import datetime
import gym
import myosuite
import mujoco as mj
import numpy as np
import pickle
import matplotlib.pyplot as plt

class Simulation():
    """Simulation class for the MuJoCo simulation."""

    def __init__(self, env):
        # self.actuator_type = "servo"
        # if you set this to True, the camera configuration will be printed
        self.frame_skip = 1 # TODO under testing
        self.print_camera_config = False
        self.model = env.sim.model
        self.data = env.sim.data
        self.mujoco_init()
        self.centre = 0.773 # radian
        
        self.start_time = 1
        self.motion_duration = 1
        self.end_time = 3
        self.forearm_id = self.model.body_name2id("r_ulna_radius_hand")

        self.control_mode = 'none'
        self.KP = 1
        self.KI = 0
        self.KD = 0
        self.stretch = 0
        
    @property
    def dt(self):
        return self.model.opt.timestep * self.frame_skip
        
    def mujoco_init(self) -> None:
        """Mujoco initilization."""
        # turn off the gravity
        self.model.opt.gravity[:] = [0, 0, 0]
        # remove skin
        geom_1_indices = np.where(self.model.geom_group == 1)
        self.model.geom_rgba[geom_1_indices, 3] = 0
        self.check_all_link()

    def get_model_info(self) -> None:
        """Get joints and muscles info of model."""
        print("Muscles:")
        for i in range(len(self.model.name_actuatoradr)):
            print([i, self.model.name_actuatoradr[i]])
        # print(mjcModel.joint_names)
        print("\nJoints:")
        for i in range(len(self.model.name_jntadr)):
            print([i, self.model.name_jntadr[i]])

    def reset(self):
        # set all joint angles to 0.0
        self.data.qpos[:] = self.centre
    
        # # joint position
        # self.jointPos = [np.array([self.centre])]
        # # time
        # self.timeVals = [np.array([0.0])]
        # # muscle act
        # self.muscleAct = [self.data.act[:].copy()]
        # # muscle length
        # self.muscleLen = [self.data.actuator_length[1:].copy()]
        # # muscle length velocity
        # self.muscleVel = [self.data.actuator_velocity[1:].copy()]
        
        # joint position
        self.jointPos = []
        # time
        self.timeVals = []
        # muscle act
        self.muscleAct = []
        # muscle length
        self.muscleLen = []
        # muscle length velocity
        self.muscleVel = []
        # control commands
        self.commands = []
        
        # applied force or torque
        self.torques = []
        self.torque = np.array([0.0])
        # position error
        self.errors = []
        
        mj.set_mjcb_control(self.controller)

    def check_all_link(self) -> None:
        """Check all links in the model."""
        for i in range(self.model.nbody):
            print([i, self.model.body(i).name])

    def check_link_id(self, link_name: str) -> int:
        """Check the link id of a specific link."""
        try:
            link_id = self.model.body(link_name).id
        except Exception as e:
            print(f"Error: {e}")
        else:
            # print(f"Link ID for '{link_name}': {link_id}")
            return link_id
    
    def get_current_motor_torque(self):
        actuator_id = self.model.actuator_name2id("Exo")
        return self.data.actuator_force[actuator_id]
    
    def get_current_motor_position(self):
        motor_position = self.data.qpos[:]
        return motor_position
    
    def get_obs_dict(self):
        obs_dict = {}
        obs_dict['time'] = np.array([self.data.time])
        obs_dict['qpos'] = self.data.qpos[:].copy()
        obs_dict['qvel'] = self.data.qvel[:].copy()*self.dt
        obs_dict['act'] = self.data.act[:].copy() if self.model.na>0 else np.zeros_like(obs_dict['qpos'])
        obs_dict['m_length'] = self.data.actuator_length[1:].copy()
        obs_dict['m_velocity'] = self.data.actuator_velocity[1:].copy()
        obs_dict['motor_force'] = self.torque
        return obs_dict
        
    def log_data(self):
        # get the current observation
        info = self.get_obs_dict()

        # record data
        self.timeVals.append(info['time'])
        self.jointPos.append(info['qpos'])
        self.muscleAct.append(info['act'])
        self.muscleLen.append(info['m_length'])
        self.muscleVel.append(info['m_velocity'])
        self.torques.append(info['motor_force'])
        self.commands.append(self.data.ctrl[:].copy())
    
    def external_force(self, force, torque):
        R = self.data.xmat[self.forearm_id].reshape(3, 3)
        
        # define force and torque in local frame
        F_local = force*np.array([1,0,0])
        T_local = torque*np.array([0,0,1])
        # Transform force and torque to world frame
        F_world = np.dot(R, F_local)
        T_world = np.dot(R, T_local)
        perturbation = np.hstack((F_world, T_world))
        
        self.data.xfrc_applied[self.forearm_id][:] = perturbation
        
    def minijerk_trajectory(self, curr_t, start_t, duration, stretch):
        ini_pos = self.centre
        target_pos = ini_pos + stretch
        t = curr_t - start_t
        T = duration
        t = max(min(t, T), .0)
        tn= max(min(t/T, 1.0), .0)
        tn3=pow(tn,3.)
        tn4=tn*tn3
        tn5=tn*tn4
        Xd = ini_pos + ( (ini_pos-target_pos) * (15.*tn4-6.*tn5-10.*tn3) )
        if tn==0 or tn==1:
            dXd = 0
        else:
            dXd = (ini_pos-target_pos) * (4.*15.*tn4-5.*6.*tn5-10.*3*tn3)/t
        return Xd, dXd, tn
    
    def ramphold_trajectory(self, curr_t, start_t, duration, stretch):
        ini_pos = self.centre
        target_pos = ini_pos + stretch
        t = curr_t - start_t
        T = duration
        t = max(min(t, T), .0)
        tn= max(min(t/T, 1.0), .0)
        Xd = ini_pos + ( (target_pos - ini_pos) * tn )
        if tn==0 or tn==1:
            dXd = 0
        else:
            dXd = (target_pos-ini_pos)/duration
        return Xd, dXd, tn
        
    def controller(self, model, data: mj.MjData):
        """Controller for the simulation."""
        
        # update the current time and position
        current_t = np.array([self.data.time])
        current_pos = self.data.qpos[:].copy()
        current_vel = self.data.qvel[:].copy()
        if len(self.timeVals) < 2:
            d_error = 0
        else:
            d_error = (self.errors[-1] - self.errors[-2])/self.dt
        
        # dual loop control of velocity and position
        if self.control_mode == 'dual':
            # todo: not real dual loop control, still a joint control
            # target_pos, target_vel, status = self.minijerk_trajectory(current_t, self.start_time, self.motion_duration, self.stretch)
            target_pos, target_vel, status = self.ramphold_trajectory(current_t, self.start_time, self.motion_duration, self.stretch)
            pos_error = target_pos - current_pos
            self.errors.append(pos_error)
            target_vel = pos_error/self.dt
            vel_error = target_vel - current_vel
            self.torque = 50*vel_error + 0*d_error + 200*np.sum(self.errors)
        
        # joint control of velocity and position       
        elif self.control_mode == 'joint':
            target_pos, target_vel, status = self.minijerk_trajectory(current_t, self.start_time, self.motion_duration, self.stretch)
            # target_pos, target_vel, status = self.ramphold_trajectory(current_t, self.start_time, self.motion_duration, self.stretch)
            error = target_pos - current_pos
            self.errors.append(error)
            # self.torque = self.KP*error + self.KD*(target_vel - current_vel) + self.KI*np.sum(self.errors)*self.dt
            self.torque = 200*error + 50*(target_vel - current_vel) + 200*np.sum(self.errors)*self.dt

        # position control
        elif self.control_mode == 'position':
            # target_pos, target_vel, status = self.minijerk_trajectory(current_t, self.start_time, self.motion_duration, self.stretch)
            target_pos, target_vel, status = self.ramphold_trajectory(current_t, self.start_time, self.motion_duration, self.stretch)
            error = target_pos - current_pos
            self.errors.append(error)
            # self.torque = self.KP*error + self.KD*d_error + self.KI*np.sum(self.errors)*self.dt
            self.torque = 100*error + 5*d_error + 200*np.sum(self.errors)*self.dt
        
        # velocity control    
        elif self.control_mode == 'velocity':
            # target_pos, target_vel, status = self.minijerk_trajectory(current_t, self.start_time, self.motion_duration, self.stretch)
            target_pos, target_vel, status = self.ramphold_trajectory(current_t, self.start_time, self.motion_duration, self.stretch)
            error = target_vel - current_vel
            self.errors.append(error)
            # self.torque = self.KP*error + self.KD*d_error + self.KI*np.sum(self.errors)*self.dt
            self.torque = 50*error + 0.01*d_error + 100*np.sum(self.errors)*self.dt
            
        else: # no control, test the model in a static state
            self.torque = np.array([0.0])
            self.errors.append(np.array([0.0]))
            
        if abs(self.torque) > 50:
            self.torque = 50*np.sign(self.torque)
        
        self.external_force(0, self.torque)
        self.log_data()
    
def muscle_spasticity(sim, type='none', muscle_dynamics=False, tau=0.03, g=2.0, b=1.0, threshold=0):
    current_time = sim.timeVals[-1]
    if type == 'ml':
        enable = True
        sensory_feedback = np.array(sim.muscleLen)
    elif type == 'mv':
        enable = True
        sensory_feedback = np.array(sim.muscleVel)
    else:
        enable = False
        sensory_feedback = np.array(sim.muscleAct)
        
    current_feedback = sensory_feedback[-1,:]
    if enable:
        if current_time <= tau:
            spastic_activation = 0*np.ones(len(sensory_feedback[-1,:]))
        else:   
            if muscle_dynamics:
                # intergration to get the current excitation of the muscle
                current_excitation = sim.muscleAct[-1]
                excitation_derivative = spastic_excitation_with_dynamics(current_time, current_excitation, current_feedback, tau, g, b, threshold)
                spastic_activation = current_excitation + excitation_derivative * 0.02
            else:
                delayed_feedback = sensory_feedback[-1 - int(tau/sim.dt), :]
                spastic_activation = spastic_excitation_without_dynamics(delayed_feedback, g, threshold)
    else:
        spastic_activation = 0*np.ones(len(sensory_feedback[-1,:]))
    
    # Explicitely project actuator space (0,1) to normalized space (-1,1), because of the setp function in MyoSuite base_v0.py
    spastic_activation[spastic_activation<0] = 0
    spastic_activation[spastic_activation>1] = 1
    
    a_max = 0.999999
    a_min = sigmoid(-1)
    a_01 = spastic_activation*(a_max - a_min) + a_min
    a_11 = inverse_sigmoid(a_01)
    spastic_activation = a_11

    return spastic_activation

def sigmoid(x, k=5.0, x0=0.5):
    return 1.0 / (1.0 + np.exp(-k * (x - x0)))

def inverse_sigmoid(y, k=5.0, x0=0.5):
    return x0 - (1 / k) * np.log((1 - y) / y)
     
# spasticity model with muscle dynamics
def spastic_excitation_with_dynamics(t, curr_exc, current_feedback, tau, g, b, thre):
    d_act = np.empty(len(current_feedback))
    for i in range(len(current_feedback)):
        c1 = -curr_exc[i] / tau
        c2 = g / tau
        f = 0.5 * np.tanh(b * (current_feedback[i] - thre))
        d_act[i] = c1 + c2 * current_feedback[i] * (f + 0.5)
    return d_act

# spasticity model without muscle dynamics
def spastic_excitation_without_dynamics(delayed_feedback, g, threshold):
    act = np.empty(len(delayed_feedback))
    for i in range(len(delayed_feedback)):
        if delayed_feedback[i] > threshold:
            act[i] = g * (delayed_feedback[i] - threshold)
        else:
            act[i] = 0.0
    return act 
    
def plot_curve(timeVals, data, ylabel, legend=None, ignore=False, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
        
    data = np.array(data)
    
    if ignore:
        data = data[:,1:]
    
    for i in range(data.shape[1]):
        ax.plot(timeVals, data[:, i], label=(legend[i] if legend else None))
        
    ax.set_xlabel('time [s]')
    ax.set_ylabel(ylabel)
    ax.grid(True)
    if legend:
        ax.legend()
    
        
def save_data(name, timeVals, jointPos, muscleAct, motorTorque):
    directory = './data/'
    # Create directory if it doesn't exist
    if not os.path.exists(directory):
        os.makedirs(directory)

    full_path = os.path.join(directory, name)
    with open(full_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['timeVals', 'jointPos', 'muscleAct', 'motorTorque'])
        for i in range(len(timeVals)):
            writer.writerow([timeVals[i], jointPos[i], muscleAct[i], motorTorque[i]])

def spasticity_experiment(env, sim, pi, No, random_policy, spasticity_type='none', muscle_dynamics=False, tau=0.03, g=2.0, b=1.0, threshold=0.0):
    obs = env.reset()
    sim.reset() # clean data for plotting
    total_time = sim.end_time
    total_steps = int(total_time/0.02)
    spas_a = []
    
    # initial observation
    a = pi.get_action(obs)[0]
    a = -1*np.ones(len(a))
    a[0] = 0
    obs, r, done, ifo = env.step(a)

    for i in range(total_steps-2):
        # observe the simulation
        # time.sleep(0.1)
        env.mj_render()
        
        if random_policy:
            a = env.action_space.sample()
        else:
            a = pi.get_action(obs)[0]
        
        sim.log_data()
        # tau is the latency of the spasticity model, g is the gain of the spasticity model, threshold is the threshold of the spasticity model
        spastic_a = muscle_spasticity(sim, spasticity_type, muscle_dynamics, tau, g, b, threshold)
        # record the spastic activation, which is [-1, 1]. When it is -1, the mujoco muscle actuator will finally get 0. 
        spas_a.append(sigmoid(spastic_a))

        # deactivate the muscles
        a = -1*np.ones(len(a))
        # add the spastic activation to the muscle activation. The spasticity activation is added with 1 to conpensate the default -1 in the muscle activation.
        a = a + np.append([0], spastic_a+1) 
        # set the activation of motor to 0
        a[0] = 0
        
        obs, r, done, ifo = env.step(a)
        
    name = f"dataset_{No}.csv"
    save_data(name, sim.timeVals, sim.jointPos, sim.muscleAct, sim.torques)
    
    # Create a figure with multiple subplots
    fig, axes = plt.subplots(3, 2, figsize=(14, 18), sharex=True)
    # Flatten axes array for easier indexing
    axes = axes.flatten()
    
    plot_curve(sim.timeVals, sim.jointPos, 'elbow joint angle [rad]', ax=axes[0])
    plot_curve(sim.timeVals, sim.torques, 'motor torque [Nm]', ax=axes[1])
    
    muscleName = ['TRIlong','TRIlat','TRImed','BIClong','BICshort','BRA']
    plot_curve(sim.timeVals, sim.muscleAct, 'muscle activation', muscleName, ax=axes[2])
    plot_curve(sim.timeVals, sim.muscleLen, 'muscle length', muscleName, ax=axes[3])
    plot_curve(sim.timeVals, sim.muscleVel, 'muscle velocity', muscleName, ax=axes[4])
    plot_curve(sim.timeVals, sim.commands, 'control signal', muscleName, ignore=True, ax=axes[5])
    
    # For the spastic activation plot, we'll need an additional axis
    t = np.linspace(0, total_time, len(spas_a))
    plot_curve(t, spas_a, 'spastic activation', muscleName)
    
    # for the error plot
    t = np.linspace(0, total_time, len(sim.errors))
    plot_curve(t, sim.errors, 'position error [rad]')
        
    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.show()

def load_policy():
    pth = "./myosuite/agents/baslines_NPG/"
    policy_path = pth+"myoElbowPose1D6MExoRandom-v0/2022-02-26_21-16-27/36_env=myoElbowPose1D6MExoRandom-v0,seed=1/iterations/best_policy.pickle"
    
    pi = pickle.load(open(policy_path, 'rb'))
    return pi

def main():
    # load the environment and policy
    env = gym.make('myoElbowPose1D6MExoRandom-v0', reset_type='init', target_type = 'fixed', weight_range=(0,0))    
    # env = gym.make('myoElbowPose1D6MFixed-v0', reset_type='init', target_type = 'fixed')
    sim = Simulation(env)
    
    pi = None # random policy
    random_policy = False
    if not random_policy:
        pi = load_policy()
    
    # set the initial joint angle
    env.env.target_jnt_value = [np.deg2rad(0)] # joint range is (0, 130)
    env.env.update_target(restore_sim=True)
    
    for i in range(1):
        print(f"Experiment {i}")
        
        # sim.KP = 50
        # sim.KI = 100
        # sim.KD = 0.01
        sim.control_mode = 'dual' # 'none', 'dual', 'joint', 'position', 'velocity'
        sim.stretch = 1.5
        
        sim.start_time = 0.5
        sim.motion_duration = 1
        sim.end_time = 2

        # the type can be 'ml' for muscle length, 'mv' for muscle velocity, 'none' for no spasticity
        spasticity_experiment(env, sim, pi, i, random_policy, spasticity_type='mv', muscle_dynamics=True, tau=0.03, g=4.0, b=20, threshold=0.025)
        
    env.close()
    
if __name__ == "__main__":
    main()