import numpy as np
from Ynaka_planner import PolicyIterationPlanner
from tqdm import tqdm


class MaxEntIRL():

    def __init__(self, env):
        self.env = env
        self.planner = PolicyIterationPlanner(env)

    def estimate(self, trajectories, epoch=20, learning_rate=0.01, gamma=0.9):
        # one-hot vecの基底を作成，縦に積む．
        state_features = np.vstack([self.env.state_to_feature(s)
                                   for s in self.env.states])
        theta = np.random.uniform(size=state_features.shape[1])
        
        # 行動をone-hotのエピソード平均へ 
        # e.g.) trajectories = [[12, 13, 14, 14, 15, 11, 7, 3, 3], [12, 13, 14, 15, 11, 10, 14, 15, 11, 7, 3, 3]]
        teacher_features = self.calculate_expected_feature(trajectories)
        
        # chk
        #print('state_features',state_features)
        #print('theta',theta)
        #print('teacher_features',teacher_features)
            
        for e in tqdm(range(epoch)):
            # Estimate reward.
            rewards = state_features.dot(theta.T)

            # Optimize policy under estimated reward.
            self.planner.reward_func = lambda s: rewards[s]
            self.planner.plan(gamma=gamma)

            # Estimate feature under policy.
            features = self.expected_features_under_policy(
                                self.planner.policy, trajectories)

            # Update to close to teacher.
            update = teacher_features - features.dot(state_features)
            theta += learning_rate * update
            
            print('-----------------------------')
            print('theta')
            print(theta)
            print('teacher_features')
            print(teacher_features)
            print('features')
            print(features)
            print('state_features')     
            print(state_features)
            print('features.dot(state_features)')
            print(features.dot(state_features))
            
        #chk
        #print('feature',features)
        #print('features.dot(state_features)',features.dot(state_features))

        estimated = state_features.dot(theta.T)
        estimated = estimated.reshape(self.env.shape)
        #return estimated
        return estimated

    def calculate_expected_feature(self, trajectories):
        features = np.zeros(self.env.observation_space.n)
        for t in trajectories:
            for s in t:
                features[s] += 1

        features /= len(trajectories)
        return features

    def expected_features_under_policy(self, policy, trajectories):
        t_size = len(trajectories)
        states = self.env.states
        transition_probs = np.zeros((t_size, len(states)))

        initial_state_probs = np.zeros(len(states))
        for t in trajectories:
            initial_state_probs[t[0]] += 1
        initial_state_probs /= t_size
        transition_probs[0] = initial_state_probs # この迷路では必ず12のみに1が立つ
        
        #chk
        print('before')
        print('transition_probs\n',transition_probs)
        print('initial_state_probs\n',initial_state_probs)

        
        # なぜこうなるのか分からない？？ https://qiita.com/yasufumy/items/4fe5d12e54b84a435d6a#%E8%A3%9C%E8%B6%B32
        # sum_a sum_s' mu_p(s')*plicy(a|s')*P(s|a,s')
        for t in range(1, t_size): # number of episode
            for prev_s in states: 
                prev_prob = transition_probs[t - 1][prev_s]
                a = self.planner.act(prev_s) # policy : state 2 action
                probs = self.env.transit_func(prev_s, a) # dict
                print('t','prev_s','a','probs')
                #print(t,prev_s,act2str(a),probs)
                print(t,prev_s,a,probs)
                for s in probs:
                    transition_probs[t][s] += prev_prob * probs[s]
                    
        print('after')
        print('transition_probs',transition_probs)

        total = np.mean(transition_probs, axis=0)
        return total
    
    def act2srt(self, a):
        if a == 0:
            return 'Left'
        elif a == 1:
            return 'Down'
        elif a == 2:
            return 'Right'
        else:
            return 'Up'



if __name__ == "__main__":
    def test_estimate():
        from environment import GridWorldEnv
        env = GridWorldEnv(grid=[
            [0, 0, 0, 1],
            [0, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, 0, 0],
        ])
        # Train Teacher
        teacher = PolicyIterationPlanner(env)
        teacher.plan()
        trajectories = []
        print("Gather demonstrations of teacher.")
        for i in range(20):
            s = env.reset()
            done = False
            steps = [s]
            while not done:
                a = teacher.act(s)
                n_s, r, done, _ = env.step(a)
                steps.append(n_s)
                s = n_s
            trajectories.append(steps)

        print("Estimate reward.")
        irl = MaxEntIRL(env)
        rewards = irl.estimate(trajectories, epoch=100)
        print(rewards)
        env.plot_on_grid(rewards)

    test_estimate()
