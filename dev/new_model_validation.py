##################################################
# In this script, we conducted a series of experiments to select the optimal spasticity model.
# This script is based on the test_spasticity_model.py script, but we disabled the function of generating and following trajectories.
# Instead, the trajectory is loaded from the reference dataset.
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
        self.frame_skip = 1
        self.print_camera_config = False
        self.model = env.sim.model
        self.data = env.sim.data
        self.mujoco_init()
        self.centre = 0.773 # modified model: -0.666 # radian
        
        self.start_time = 1
        self.motion_duration = 1
        self.end_time = 3
        self.forearm_id = self.model.body_name2id("r_ulna_radius_hand")

        # custom controller
        self.KP = 1
        self.KI = 0
        self.KD = 0
        self.stretch = 0
        self.alpha = 0.1 # filter coefficient of the velocity derivative error, smaller value means more filtering
        self.filtered_d_error = 0
        self.filtered_vd_error = 0
        
        # reference trajectory
        self.refTime = np.array([0.0])
        self.refAngle = np.array([0.0])
        self.refTorque = np.array([0.0])
        
        # compensation for the passive torque of the elbow model
        self.compensation = False
        self.passiveAngle = []
        self.passiveTorque = []
        
        # muscle parameters for normalising the muscle feedback
        self.starting_length = np.array([0.0, 0.0, 0.0, 0.36, 0.28, 0.12])
        self.stretch_length = np.array([0.0, 0.0, 0.0, 0.054, 0.054, 0.024])
        self.max_velocity = np.array([0.0, 0.0, 0.0, 0.18, 0.20, 0.13])
        self.max_force = np.array([0.0, 0.0, 0.0, 729, 508, 1171])
        
    @property
    def dt(self):
        dt = self.model.opt.timestep * self.frame_skip
        return dt
        
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
        # muscle velocity
        self.muscleVel = []
        # muscle force
        self.muscleForce = []
        # control commands
        # self.commands = []
        
        # applied force or torque
        self.torques = []
        self.torque = np.array([0.0])
        # position error
        self.p_errors = []
        self.v_errors = []
        
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
        obs_dict['m_force'] = -self.data.actuator_force[1:].copy()
        obs_dict['motor_force'] = -self.torque
        return obs_dict
        
    def log_data(self):
        # get the current observation
        info = self.get_obs_dict()

        # record data
        if len(self.timeVals) == 0 or self.timeVals[-1] != info['time']: # skip the data with the same time
            self.timeVals.append(info['time'])
            self.jointPos.append(info['qpos'])
            self.muscleAct.append(info['act'])
            self.muscleLen.append(info['m_length'])
            self.muscleVel.append(info['m_velocity'])
            self.muscleForce.append(info['m_force'])
            self.torques.append(info['motor_force'])
            # self.commands.append(self.data.ctrl[:].copy())
    
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
    
    def reference_trajectory(self, curr_t, start_t):
        t = curr_t - start_t
        idx = np.where(self.refTime == t)[0]
        if len(idx) == 0:
            idx = np.where(self.refTime > t)[0]
        if len(idx) == 0:
            idx = len(self.refTime) - 1
        else:
            idx = idx[0]
        return self.refAngle[idx]
        
    def controller(self, model, data: mj.MjData):
        """Controller for the simulation."""
        
        # update the current time and position
        current_t = np.array([self.data.time])
        current_pos = self.data.qpos[:].copy()
        current_vel = self.data.qvel[:].copy()
        
        if self.compensation:
            # compensate the passive torque of the elbow model
            passive_torque = np.interp(current_pos, self.passiveAngle, self.passiveTorque)
        
        if len(self.timeVals) < 2:
            d_error = 0
            vel_d_error = 0
        else:
            d_error = (self.p_errors[-1] - self.p_errors[-2])/self.dt
            vel_d_error = (self.v_errors[-1] - self.v_errors[-2])/self.dt
            self.filtered_d_error = self.alpha * d_error + (1 - self.alpha) * self.filtered_d_error
            self.filtered_vd_error = self.alpha * vel_d_error + (1 - self.alpha) * self.filtered_vd_error
            
        # dual loop control of velocity and position
        self.alpha = 0.1
        # position loop control
        target_pos = self.reference_trajectory(current_t, 0)
        pos_error = target_pos - current_pos
        self.p_errors.append(pos_error)
        target_vel = pos_error*30 - 0.01*self.filtered_d_error + 100*np.sum(self.p_errors)*self.dt
        
        # velocity loop control
        vel_error = target_vel - current_vel
        self.v_errors.append(vel_error)
        self.torque = 5*vel_error - 0.01*self.filtered_vd_error + 400*np.sum(self.v_errors)*self.dt
            
        if abs(self.torque) > 30:
            self.torque = 30*np.sign(self.torque)
        
        self.external_force(0, self.torque+passive_torque)
        self.log_data()
    
def muscle_spasticity(sim, mus_select, muscle_dynamics, tau, gains, thresholds):
    current_time = sim.timeVals[-1]
    if current_time > tau:
        if muscle_dynamics:
            index = -1 - int(tau/sim.dt)
        else:
            index = -1
        len_feedback = sim.muscleLen[index]
        vel_feedback = sim.muscleVel[index]
        force_feedback = sim.muscleForce[index]
        
        len_reflex = np.zeros(len(len_feedback))
        vel_reflex = np.zeros(len(vel_feedback))
        force_reflex = np.zeros(len(force_feedback))
        
        for i, enable in enumerate(mus_select):
            if enable == 1:
                if muscle_dynamics:
                    b = 200 # to make the hyperbolic tangent function steep enough
                    f = 0.5 * np.tanh(b * (len_feedback[i]-thresholds[0][i])) + 0.5
                    len_reflex[i] = f*(len_feedback[i]-sim.starting_length[i])/(sim.starting_length[i]+sim.stretch_length[i])
                    f = 0.5 * np.tanh(b * (vel_feedback[i]-thresholds[1][i])) + 0.5
                    vel_reflex[i] = f*vel_feedback[i]/sim.max_velocity[i]*sim.stretch_length[i]/(sim.starting_length[i]+sim.stretch_length[i])
                    f = 0.5 * np.tanh(b * (force_feedback[i]-thresholds[2][i])) + 0.5
                    force_reflex[i] = f*force_feedback[i]/sim.max_force[i]
                else:
                    if len_feedback[i] > thresholds[0][i]:
                        len_reflex[i] = (len_feedback[i]-thresholds[0][i])/(sim.starting_length[i]+sim.stretch_length[i])
                    if vel_feedback[i] > thresholds[1][i]:
                        vel_reflex[i] = (vel_feedback[i]-thresholds[1][i])/sim.max_velocity[i]*sim.stretch_length[i]/(sim.starting_length[i]+sim.stretch_length[i])*3
                    if force_feedback[i] > thresholds[2][i]:
                        force_reflex[i] = (force_feedback[i]-thresholds[2][i])/sim.max_force[i]*sim.stretch_length[i]/(sim.starting_length[i]+sim.stretch_length[i])*3
                            
        current_reflexes = np.vstack((len_reflex, vel_reflex, force_reflex))
            
        if muscle_dynamics:
            # intergration to get the current excitation of the muscle
            current_excitation = sim.muscleAct[-1]
            excitation_derivative = spastic_excitation_with_dynamics2(current_time, current_excitation, current_reflexes, tau, gains)
            spastic_activation = current_excitation + excitation_derivative * 0.02
        else:
            spastic_activation = spastic_excitation_without_dynamics2(current_reflexes, gains)
    else:
        spastic_activation = 0*np.ones(len(mus_select))
    
    # Explicitely project actuator space (0,1) to normalized space (-1,1), because of the setp function in MyoSuite base_v0.py
    spastic_activation = spastic_activation*np.array(mus_select)
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
def spastic_excitation_with_dynamics(t, curr_exc, current_feedback, tau, g, thre):
    b = 200 # to make the hyperbolic tangent function steep enough
    d_act = np.empty(len(current_feedback))
    for i in range(len(current_feedback)):
        c1 = -curr_exc[i] / tau
        c2 = g / tau
        f = 0.5 * np.tanh(b * (current_feedback[i] - thre[i])) + 0.5
        d_act[i] = c1 + c2 * current_feedback[i] * f
    return d_act

# spasticity model without muscle dynamics
def spastic_excitation_without_dynamics(delayed_feedback, g, thre):
    act = np.empty(len(delayed_feedback))
    for i in range(len(delayed_feedback)):
        if delayed_feedback[i] > thre[i]:
            act[i] = g * (delayed_feedback[i] - thre[i])
        else:
            act[i] = 0.0
    return act 

# spasticity model with muscle dynamics, multiple feedbacks
def spastic_excitation_with_dynamics2(t, curr_exc, afferent_reflexes, tau, gains):
    d_act = np.zeros(np.shape(afferent_reflexes)[1]) # the derivative of the muscle activation, six muscles
    for j in range(np.shape(afferent_reflexes)[0]): # loop three times for three feedbacks
        afferent_reflex = afferent_reflexes[j]
        g = gains[j]
        for i in range(len(afferent_reflex)):
            c1 = -curr_exc[i] / tau
            c2 = g / tau
            d_act[i] = d_act[i] + c1 + c2 * afferent_reflex[i]
    return d_act

# spasticity model without muscle dynamics, multiple feedbacks
def spastic_excitation_without_dynamics2(afferent_reflexes, gains):
    act = np.zeros(np.shape(afferent_reflexes)[1]) # the muscle activation, six muscles
    for j in range(np.shape(afferent_reflexes)[0]): # loop three times for three feedbacks
        afferent_reflex = afferent_reflexes[j]
        g = gains[j] 
        for i in range(len(afferent_reflex)):
            act[i] = act[i] + g * afferent_reflex[i]
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
    directory = './data/sim_data2/'
    # Create directory if it doesn't exist
    if not os.path.exists(directory):
        os.makedirs(directory)

    full_path = os.path.join(directory, name)
    with open(full_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['timeVals', 'jointPos', 'muscleAct', 'motorTorque'])
        for i in range(len(timeVals)):
            writer.writerow([timeVals[i], jointPos[i], muscleAct[i], motorTorque[i]])
            
def plot_data(sim):
    # Create a figure with multiple subplots
    fig, axes = plt.subplots(3, 2, figsize=(14, 18), sharex=True)
    # Flatten axes array for easier indexing
    axes = axes.flatten()
    
    plot_curve(sim.timeVals, sim.jointPos, 'elbow joint angle [rad]', ax=axes[0])
    ax=axes[0]
    ax.plot(sim.refTime, sim.refAngle)
    ax.legend(['sim', 'ref'])
    plot_curve(sim.timeVals, sim.torques, 'motor torque [Nm]', ax=axes[1])
    ax=axes[1]
    ax.plot(sim.refTime, sim.refTorque)
    ax.legend(['sim', 'ref'])
    
    muscleName = ['TRIlong','TRIlat','TRImed','BIClong','BICshort','BRA']
    plot_curve(sim.timeVals, sim.muscleAct, 'muscle activation', muscleName, ax=axes[2])
    plot_curve(sim.timeVals, sim.muscleLen, 'muscle length', muscleName, ax=axes[3])
    plot_curve(sim.timeVals, sim.muscleVel, 'muscle velocity', muscleName, ax=axes[4])
    plot_curve(sim.timeVals, sim.muscleForce, 'muscle force', muscleName, ax=axes[5])
    # plot_curve(sim.timeVals, sim.commands, 'control commands', muscleName, ax=axes[5])
        
    # for the error plot
    # t = np.linspace(0, sim.timeVals[-1], len(sim.v_errors))
    # plot_curve(t, sim.v_errors, 'position error [rad]')
        
    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.show()

def spasticity_experiment(env, sim, No, pi, random_policy, mus_select = [0,0,0,0,0,0], muscle_dynamics=False, tau=0.03, g=[0, 0, 0], thre=[1, 1, 1]):
    obs = env.reset()
    sim.reset() # clean data for plotting
    # total_time = sim.end_time
    # total_steps = int(total_time/0.02)
    total_steps = int(len(sim.refTime)/10)
    spas_a = []
    
    # initial observation
    a = pi.get_action(obs)[0]
    a = -1*np.ones(len(a))
    a[0] = 0
    obs, r, done, ifo = env.step(a)
    
    # initialise the thresholds of the spasticity model
    len_thre = sim.starting_length + thre[0]*sim.stretch_length
    vel_thre = thre[1]*sim.max_velocity
    force_thre = thre[2]*sim.max_force
    thresholds = np.vstack((len_thre, vel_thre, force_thre))
    
    indices = [index for index, value in enumerate(g) if value > 0]
    spasticity_types = ['ml', 'mv', 'mf']
    if len(indices) == 1:
        spasticity_type = spasticity_types[indices[0]]
        Gain = g[indices[0]]
    elif len(indices) == 2:
        spasticity_type = 'lv'
        Gain = g[1]
    else:
        spasticity_type = 'none'
        Gain = g[0]
    
    start_time = time.time()
    for i in range(total_steps-2):
        # observe the simulation
        # time.sleep(0.05)
        # env.mj_render()
                    
        if random_policy:
            a = env.action_space.sample()
        else:
            a = pi.get_action(obs)[0]
            
        sim.log_data()
            
        # tau is the latency of the spasticity model, g is the gain of the spasticity model, threshold is the threshold of the spasticity model
        spastic_a = muscle_spasticity(sim, mus_select, muscle_dynamics, tau, g, thresholds)
        # record the spastic activation, which is [-1, 1]. When it is -1, the mujoco muscle actuator will finally get 0. 
        # spas_a.append(sigmoid(spastic_a))

        # deactivate the muscles
        a = -1*np.ones(len(a))
        # add the spastic activation to the muscle activation. The spasticity activation is added with 1 to conpensate the default -1 in the muscle activation.
        a = a + np.append([0], spastic_a+1) 
        # set the activation of motor to 0
        a[0] = 0
        
        obs, r, done, ifo = env.step(a)

    end_time = time.time()
    print(f"Time cost: {end_time-start_time}")
    
    if mus_select == [0, 0, 0, 0, 0, 1]:
        muscle_type = 'BRA'
    elif mus_select == [0, 0, 0, 0, 1, 0]:
        muscle_type = 'BICS'
    elif mus_select == [0, 0, 0, 1, 0, 0]:
        muscle_type = 'BICL'
    else:
        muscle_type = 'BBB'
    
    name = muscle_type + "_" + spasticity_type + f"_{Gain}" + f"_{No}"+ ".csv"
    print(name)
    save_data(name, sim.timeVals, sim.jointPos, sim.muscleAct, sim.torques)
    # plot_data(sim)

def load_policy():
    pth = "./myosuite/agents/baslines_NPG/"
    policy_path = pth+"myoElbowPose1D6MExoRandom-v0/2022-02-26_21-16-27/36_env=myoElbowPose1D6MExoRandom-v0,seed=1/iterations/best_policy.pickle"
    
    pi = pickle.load(open(policy_path, 'rb'))
    return pi

def load_data():
    file_path = './data/ref_data_0_1.pkl'
    data = pickle.load(open(file_path, 'rb'))
    return data

def main():
    # load the environment and policy
    env = gym.make('myoElbowPose1D6MExoRandom-v0', reset_type='init', target_type = 'fixed', weight_range=(0,0))    
    # env = gym.make('myoElbowPose1D6MFixed-v0', reset_type='init', target_type = 'fixed')
    sim = Simulation(env)
    
    pi = None # random policy
    random_policy = False
    if not random_policy:
        pi = load_policy()
        
    compensation = True # compensate the passive force of the elbow model
    if compensation:
        sim.compensation = True
        calibration_data = np.load('./dev/validation_data/passive_force.npy', allow_pickle=True)
        sim.passiveAngle = calibration_data[0]
        sim.passiveTorque = calibration_data[1]
        # plt.plot(sim.passiveAngle, sim.passiveTorque)

    # set the initial joint angle
    env.env.target_jnt_value = [0.774] # joint range is (0, 130)
    env.env.update_target(restore_sim=True)
    
    # load the data of refrence trajectory
    refData = load_data()
    ref_names = refData.keys()
    
    for model_index in range(0,3):
        for No, ref_name in enumerate(ref_names):
            sim.refTime = refData[ref_name]['time']
            # sim.refAngle = np.pi - (refData[ref_name]['angle']+60)/180*np.pi # use this if you are using the reference trajectroy from the literature, ref_data_0.pkl
            sim.refAngle = np.pi-(refData[ref_name]['angle'])/180*np.pi # use thsi if you are using the reference trajectory from Vincent, ref_data_0_1.pkl and ref_data_2_1.pkl
            sim.refTorque = refData[ref_name]['torque']
            sim.centre = sim.refAngle[0]
            
            for i in range(1, 4, 1):
                index = i + No*len(ref_names)
                print(f"Experiment {index}")
                
                # the gains and thre_factors for three feebacks: muscle length, muscle velocity, muscle force
                gains = [0, 0, 0]
                gains[model_index] = i # the gain will determine the spasticity type, can be 'ml' for muscle length, 'mv' for muscle velocity, 'mf' for muscle force, 'lv' for multiple feedbacks, 'none' for no spasticity
                
                thre_factors = [0.1, 0.1, 0]
                muscle_selection = [0, 0, 0, 1, 1, 1] # select the muscle to be activated, 1 for activated, 0 for deactivated, the order is TRIlong, TRIlat, TRImed, BIClong, BICshort, BRA
                
                # the tau is the latency of the spasticity model, g is the gain of the spasticity model, threshold is the threshold of the spasticity model
                # tau is 30 ms set accroding to the literature, threshold is 0.05 rad/s which is considered a velocity that won't cause spasticity in literature
                spasticity_experiment(env, sim, No, pi, random_policy, mus_select=muscle_selection, muscle_dynamics=False, tau=0.03, g=gains, thre=thre_factors)
    env.close()
    
if __name__ == "__main__":
    main()