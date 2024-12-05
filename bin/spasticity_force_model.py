##################################################
# This script is a simple implementation of the spasticity force model
##################################################

import numpy as np
import random
import matplotlib.pyplot as plt

# model parameters
tau = 0.1
g = 1.0
b = 100
threshold = 0.5

# time and sampling rate
t_start = 0.0
t_end = 2.0
sampling_rate = 500
t = np.linspace(t_start, t_end, int((t_end - t_start) * sampling_rate))

# sensory information for muscle activation
dF = np.zeros_like(t)
sensory_info = np.zeros_like(t)

# motion simulation
def motion_simulation_step(curr_t, curr_exc):
    # test: step activation
    if 0.5 <= curr_t <= 0.6:
        return 1.0
    return 0.0

# spasticity model with muscle dynamics
def spastic_excitation_with_dynamics(t, curr_exc, info, tau, g, b, thre):
    c1 = -curr_exc / tau
    c2 = g / tau
    f = 0.5 * np.tanh(b * (info - thre))
    d_act = c1 + c2 * info * (f + 0.5)
    return d_act

# spasticity model without muscle dynamics
def spastic_excitation_without_dynamics(delayed_feedback, g, threshold):
    if delayed_feedback > threshold:
        act = g * (delayed_feedback - threshold)
    else:
        act = 0.0
    return act

def get_muscle_excitation():
    exc = random.random()
    return exc
    
# main loop

e0 = 0.0  # initial excitation of muscle
current_activation = e0 # activation is commands from the brain to motor unit
current_excitation = e0 # excitation is the real muscle activity
muscle_dynamics = True

if muscle_dynamics:
    for i, current_time in enumerate(t):
        sensory_feedback = motion_simulation_step(current_time, current_excitation)
        sensory_data[i] = sensory_feedback
        if i == 0:
            estimated_emg[i] = e0
        else:
            # intergration to get the current excitation of the muscle
            dt = t[i] - t[i-1]
            excitation_derivative = spastic_excitation_with_dynamics(current_time, current_excitation, sensory_feedback, tau, g, b, threshold)
            current_excitation = current_excitation + excitation_derivative * dt
            estimated_emg[i] = current_excitation

else:
    for i, current_time in enumerate(t):
        sensory_feedback = motion_simulation_step(current_time, current_activation)
        sensory_data[i] = sensory_feedback
        current_excitation = get_muscle_excitation()
        estimated_emg[i] = current_excitation
        if current_time <= tau:
            current_activation = current_activation
        else:
            delayed_feedback = sensory_data[i - int(tau * sampling_rate)]
            current_activation = spastic_excitation_without_dynamics(delayed_feedback, g, threshold)
            

# plot
plt.figure(figsize=(10, 5))
plt.plot(t, edFf, label='Muscle Activation (edFf)')
plt.axhline(y=threshold, color='r', linestyle='--', label='Threshold')
plt.xlabel('Time (s)')
plt.ylabel('Value')
plt.legend()
plt.title('Real-time Muscle Activation Dynamics')
plt.show()
