##################################################
# Please use this script to test if your environment is working correctly
##################################################

""" Load a policy and run it on the myo elbow exo environment, note errors with mujoco and mjrl in MacOS """
import gym
import myosuite
import numpy as np

env = gym.make('myoElbowPose1D6MExoRandom-v0', reset_type='random')

# mjrl error with mujoco
import pickle
policy_path = "./myosuite/agents/baslines_NPG/myoElbowPose1D6MExoRandom-v0/2022-02-26_21-16-27/36_env=myoElbowPose1D6MExoRandom-v0,seed=1/iterations/best_policy.pickle"
pi = pickle.load(open(policy_path, 'rb'))

AngleSequence = [60, 30, 30, 60, 80, 80, 60, 30, 80, 30, 80, 60]
env.reset()
frames = []
for ep in range(len(AngleSequence)):
    print("Ep {} of {} testing angle {}".format(ep, len(AngleSequence), AngleSequence[ep]))
    env.env.target_jnt_value = [np.deg2rad(AngleSequence[int(ep)])]
    env.env.target_type = 'fixed'
    env.env.weight_range=(0,0)
    env.env.update_target()
    for _ in range(40):
        # frame = env.sim.renderer.render_offscreen(
        #                 width=400,
        #                 height=400,
        #                 camera_id=0)
        # frames.append(frame)
        env.mj_render()
        o = env.get_obs()
        a = pi.get_action(o)[0]
        next_o, r, done, ifo = env.step(a) # take an action based on the current observation
env.close()