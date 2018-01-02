from __future__ import division
import argparse

from PIL import Image
import numpy as np
import gym

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Convolution2D, Permute, LSTM, TimeDistributed, Input, Lambda
from keras.optimizers import Adam
import keras.backend as K

from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, BoltzmannQPolicy, EpsGreedyQPolicy
from rl.core import Processor
from rl.callbacks import FileLogger, ModelIntervalCheckpoint


INPUT_SHAPE = (84, 84)

from rl.core import Agent
from rl.agents.dqn import AbstractDQNAgent
from rl.util import *
from rl.memory import *

def mean_q(y_true, y_pred):
    return K.mean(K.max(y_pred, axis=-1))

INPUT_SHAPE = (84, 84)
WINDOW_LENGTH = 4


EpisodicTimestep = namedtuple('EpisodicTimestep', 'observation, action, reward, terminal')


class RecurrentDQNAgent(AbstractDQNAgent):
    def __init__(self, model, policy=EpsGreedyQPolicy(), enable_double_dqn=True,
                 target_model=None, policy_model=None,
                 nb_max_steps_recurrent_unrolling=100, *args, **kwargs):
        super(RecurrentDQNAgent, self).__init__(*args, **kwargs)

        # Validate (important) input.
        if hasattr(model.output, '__len__') and len(model.output) > 1:
            raise ValueError('Model "{}" has more than one output. DQN expects a model that has a single output.'.format(model))
        if model.output._keras_shape[-1] != self.nb_actions:
            raise ValueError('Model output "{}" has invalid shape. DQN expects a model that has one dimension for each action, in this case {}.'.format(model.output, self.nb_actions))

        # Validate settings for recurrent DQN.
        self.is_recurrent = True
        if self.is_recurrent:
            if enable_double_dqn:
                raise ValueError('DoubleDQN (`enable_double_dqn = True`) is currently not supported for recurrent Q learning.')
            memory = kwargs['memory']
            if not memory.is_episodic:
                raise ValueError('Recurrent Q learning requires an episodic memory. You are trying to use it with memory={} instead.'.format(memory))
            if nb_max_steps_recurrent_unrolling and not model.stateful:
                raise ValueError('Recurrent Q learning with max. unrolling requires a stateful model.')
            if policy_model is None or not policy_model.stateful:
                raise ValueError('Recurrent Q learning requires a separate stateful policy model with batch_size=1. Please refer to an example to see how to properly set it up.')

        # Parameters.
        self.enable_double_dqn = enable_double_dqn
        self.nb_max_steps_recurrent_unrolling = nb_max_steps_recurrent_unrolling

        # Related objects.
        self.model = model
        self.target_model = target_model
        self.policy_model = policy_model if policy_model is not None else model
        self.policy = policy

        # State.
        self.reset_states()

    def get_config(self):
        config = super(RecurrentDQNAgent, self).get_config()
        config['enable_double_dqn'] = self.enable_double_dqn
        config['nb_max_steps_recurrent_unrolling'] = self.nb_max_steps_recurrent_unrolling
        config['model'] = get_object_config(self.model)
        config['policy'] = get_object_config(self.policy)
        if self.compiled:
            config['target_model'] = get_object_config(self.target_model)
        return config

    def compile(self, optimizer, metrics=[]):
        metrics += [mean_q]  # register default metrics

        # We never train the target model, hence we can set the optimizer and loss arbitrarily.
        if self.target_model is None:
            self.target_model = clone_model(self.model, self.custom_model_objects)
        self.target_model.compile(optimizer='sgd', loss='mse')
        self.model.compile(optimizer='sgd', loss='mse')

        # Compile model.
        updates = []
        if self.target_model_update < 1.:
            # We use the `AdditionalUpdatesOptimizer` to efficiently soft-update the target model.
            updates += get_soft_target_model_updates(self.target_model, self.model, self.target_model_update)
        if self.policy_model is not self.model:
            # Update the policy model after every training step.
            updates += get_soft_target_model_updates(self.policy_model, self.model, 1.)
        if len(updates) > 0:
            optimizer = AdditionalUpdatesOptimizer(optimizer, updates)

        def clipped_masked_mse(args):
            y_true, y_pred, mask = args
            delta = K.clip(y_true - y_pred, self.delta_clip[0], self.delta_clip[1])
            delta *= mask  # apply element-wise mask
            loss = K.mean(K.square(delta), axis=-1)
            # Multiply by the number of actions to reverse the effect of the mean.
            loss *= float(self.nb_actions)
            return loss

        # Create trainable model. The problem is that we need to mask the output since we only
        # ever want to update the Q values for a certain action. The way we achieve this is by
        # using a custom Lambda layer that computes the loss. This gives us the necessary flexibility
        # to mask out certain parameters by passing in multiple inputs to the Lambda layer.
        input_shape = (None, self.nb_actions) if self.is_recurrent else (self.nb_actions,)
        output_shape = (None, 1) if self.is_recurrent else (1,)

        y_pred = self.model.output
        y_true = Input(name='y_true', shape=input_shape)
        mask = Input(name='mask', shape=input_shape)
        loss_out = Lambda(clipped_masked_mse, output_shape=output_shape, name='loss')([y_pred, y_true, mask])
        ins = [self.model.input] if type(self.model.input) is not list else self.model.input
        trainable_model = Model(input=ins + [y_true, mask], output=[loss_out, y_pred])
        assert len(trainable_model.output_names) == 2
        combined_metrics = {trainable_model.output_names[1]: metrics}
        losses = [
            lambda y_true, y_pred: y_pred,  # loss is computed in Lambda layer
            lambda y_true, y_pred: K.zeros_like(y_pred),  # we only include this for the metrics
        ]
        trainable_model.compile(optimizer=optimizer, loss=losses, metrics=combined_metrics)
        self.trainable_model = trainable_model

        self.update_target_model_hard()
        self.compiled = True

    def load_weights(self, filepath):
        self.model.load_weights(filepath)
        self.update_target_model_hard()

    def save_weights(self, filepath, overwrite=False):
        self.model.save_weights(filepath, overwrite=overwrite)

    def reset_states(self):
        self.recent_action = None
        self.recent_observation = None
        if self.compiled:
            self.model.reset_states()
            self.target_model.reset_states()
            self.policy_model.reset_states()

    def update_target_model_hard(self):
        self.target_model.set_weights(self.model.get_weights())
        if self.policy_model is not None:
            self.policy_model.set_weights(self.model.get_weights())

    def compute_q_values(self, state):
        batch = self.process_state_batch([state])
        if self.is_recurrent:
            # Add time axis.
            batch = batch.reshape((1,) + batch.shape)  # (1, 1, ...)
        q_values = self.policy_model.predict_on_batch(batch).flatten()
        assert q_values.shape == (self.nb_actions,)
        return q_values

    def forward(self, observation):
        # Select an action.
        state = self.memory.get_recent_state(observation)
        q_values = self.compute_q_values(state)
        action = self.policy.select_action(q_values=q_values)
        if self.processor is not None:
            action = self.processor.process_action(action)

        # Book-keeping.
        self.recent_observation = observation
        self.recent_action = action

        return action

    def backward(self, reward, terminal):
        # Store most recent experience in memory.
        if self.step % self.memory_interval == 0:
            self.memory.append(self.recent_observation, self.recent_action, reward, terminal,
                               training=self.training)

        metrics = [np.nan for _ in self.metrics_names]
        if not self.training:
            # We're done here. No need to update the experience memory since we only use the working
            # memory to obtain the state over the most recent observations.
            return metrics

        # Train the network on a single stochastic batch.
        if self.step > self.nb_steps_warmup and self.step % self.train_interval == 0:
            experiences = self.memory.sample(self.batch_size)
            assert len(experiences) == self.batch_size

            if self.is_recurrent:
                lengths = [len(seq) for seq in experiences]
                maxlen = np.max(lengths)

                # Start by extracting the necessary parameters (we use a vectorized implementation).
                state0_batch = [[] for _ in range(len(experiences))]
                reward_batch = [[] for _ in range(len(experiences))]
                action_batch = [[] for _ in range(len(experiences))]
                terminal1_batch = [[] for _ in range(len(experiences))]
                state1_batch = [[] for _ in range(len(experiences))]
                for sequence_idx, sequence in enumerate(experiences):
                    for e in sequence:
                        state0_batch[sequence_idx].append(e.state0)
                        state1_batch[sequence_idx].append(e.state1)
                        reward_batch[sequence_idx].append(e.reward)
                        action_batch[sequence_idx].append(e.action)
                        terminal1_batch[sequence_idx].append(0. if e.terminal1 else 1.)

                    # Apply padding.
                    state_shape = state0_batch[sequence_idx][-1].shape
                    while len(state0_batch[sequence_idx]) < maxlen:
                        state0_batch[sequence_idx].append(np.zeros(state_shape))
                        state1_batch[sequence_idx].append(np.zeros(state_shape))
                        reward_batch[sequence_idx].append(0.)
                        action_batch[sequence_idx].append(0)
                        terminal1_batch[sequence_idx].append(1.)

                state0_batch = self.process_state_batch(state0_batch)
                state1_batch = self.process_state_batch(state1_batch)
                terminal1_batch = np.array(terminal1_batch)
                reward_batch = np.array(reward_batch)
                assert reward_batch.shape == (self.batch_size, maxlen)
                assert terminal1_batch.shape == reward_batch.shape
                assert len(action_batch) == len(reward_batch)
            else:
                # Start by extracting the necessary parameters (we use a vectorized implementation).
                state0_batch = []
                reward_batch = []
                action_batch = []
                terminal1_batch = []
                state1_batch = []
                for e in experiences:
                    state0_batch.append(e.state0)
                    state1_batch.append(e.state1)
                    reward_batch.append(e.reward)
                    action_batch.append(e.action)
                    terminal1_batch.append(0. if e.terminal1 else 1.)

                # Prepare and validate parameters.
                state0_batch = self.process_state_batch(state0_batch)
                state1_batch = self.process_state_batch(state1_batch)
                terminal1_batch = np.array(terminal1_batch)
                reward_batch = np.array(reward_batch)
                assert reward_batch.shape == (self.batch_size,)
                assert terminal1_batch.shape == reward_batch.shape
                assert len(action_batch) == len(reward_batch)

            # Compute Q values for mini-batch update.
            if self.enable_double_dqn:
                # Double DQN relies on the model for additional predictions, which we cannot use
                # since it must be stateful (we could save the state and re-apply, but this is
                # messy).
                assert not self.is_recurrent

                # According to the paper "Deep Reinforcement Learning with Double Q-learning"
                # (van Hasselt et al., 2015), in Double DQN, the online network predicts the actions
                # while the target network is used to estimate the Q value.
                if self.is_recurrent:
                    self.model.reset_states()
                q_values = self.model.predict_on_batch(state1_batch)
                assert q_values.shape == (self.batch_size, self.nb_actions)
                actions = np.argmax(q_values, axis=1)
                assert actions.shape == (self.batch_size,)

                # Now, estimate Q values using the target network but select the values with the
                # highest Q value wrt to the online model (as computed above).
                if self.is_recurrent:
                    self.target_model.reset_states()
                target_q_values = self.target_model.predict_on_batch(state1_batch)
                assert target_q_values.shape == (self.batch_size, self.nb_actions)
                q_batch = target_q_values[range(self.batch_size), actions]
            else:
                # Compute the q_values given state1, and extract the maximum for each sample in the batch.
                # We perform this prediction on the target_model instead of the model for reasons
                # outlined in Mnih (2015). In short: it makes the algorithm more stable.
                target_q_values = self.target_model.predict_on_batch(state1_batch)
                if self.is_recurrent:
                    assert target_q_values.shape == (self.batch_size, maxlen, self.nb_actions)
                else:
                    assert target_q_values.shape == (self.batch_size, self.nb_actions)
                q_batch = np.max(target_q_values, axis=-1)
            if self.is_recurrent:
                assert q_batch.shape == (self.batch_size, maxlen)
            else:
                q_batch = q_batch.flatten()
                assert q_batch.shape == (self.batch_size,)

            if self.is_recurrent:
                targets = np.zeros((self.batch_size, maxlen, self.nb_actions))
                dummy_targets = np.zeros((self.batch_size, maxlen, 1))
                masks = np.zeros((self.batch_size, maxlen, self.nb_actions))
            else:
                targets = np.zeros((self.batch_size, self.nb_actions))
                dummy_targets = np.zeros((self.batch_size,))
                masks = np.zeros((self.batch_size, self.nb_actions))

            # Compute r_t + gamma * max_a Q(s_t+1, a) and update the target targets accordingly,
            # but only for the affected output units (as given by action_batch).
            discounted_reward_batch = self.gamma * q_batch
            # Set discounted reward to zero for all states that were terminal.
            discounted_reward_batch *= terminal1_batch
            assert discounted_reward_batch.shape == reward_batch.shape
            Rs = reward_batch + discounted_reward_batch
            if self.is_recurrent:
                for batch_idx, (inner_targets, inner_masks, inner_Rs, inner_action_batch, length) in enumerate(zip(targets, masks, Rs, action_batch, lengths)):
                    for idx, (target, mask, R, action) in enumerate(zip(inner_targets, inner_masks, inner_Rs, inner_action_batch)):
                        target[action] = R  # update action with estimated accumulated reward
                        dummy_targets[batch_idx, idx] = R
                        if idx < length:  # only enable loss for valid transitions
                            mask[action] = 1.  # enable loss for this specific action

            else:
                for idx, (target, mask, R, action) in enumerate(zip(targets, masks, Rs, action_batch)):
                    target[action] = R  # update action with estimated accumulated reward
                    dummy_targets[idx] = R
                    mask[action] = 1.  # enable loss for this specific action

            targets = np.array(targets).astype('float32')
            masks = np.array(masks).astype('float32')
            ins = [state0_batch] if type(self.model.input) is not list else state0_batch

            # In the recurrent case, we support splitting the sequences into multiple
            # chunks. Each chunk is then used as a training example. The reason for this is that,
            # for too long episodes, the unrolling in time during backpropagation can exceed the
            # memory of the GPU (or, to a lesser degree, the RAM if training on CPU).
            if self.is_recurrent and self.nb_max_steps_recurrent_unrolling:
                assert targets.ndim == 3
                steps = targets.shape[1]  # (batch_size, steps, actions)
                nb_chunks = int(np.ceil(float(steps) / float(self.nb_max_steps_recurrent_unrolling)))
                chunks = []
                for chunk_idx in range(nb_chunks):
                    start = chunk_idx * self.nb_max_steps_recurrent_unrolling
                    t = targets[:, start:start + self.nb_max_steps_recurrent_unrolling, ...]
                    m = masks[:, start:start + self.nb_max_steps_recurrent_unrolling, ...]
                    iss = [i[:, start:start + self.nb_max_steps_recurrent_unrolling, ...] for i in ins]
                    dt = dummy_targets[:, start:start + self.nb_max_steps_recurrent_unrolling, ...]
                    chunks.append((iss, t, m, dt))
            else:
                chunks = [(ins, targets, masks, dummy_targets)]

            metrics = []
            if self.is_recurrent:
                # Reset states before training on the entire sequence.
                self.trainable_model.reset_states()
            for i, t, m, dt in chunks:
                # Finally, perform a single update on the entire batch. We use a dummy target since
                # the actual loss is computed in a Lambda layer that needs more complex input. However,
                # it is still useful to know the actual target to compute metrics properly.
                ms = self.trainable_model.train_on_batch(i + [t, m], [dt, t])
                ms = [metric for idx, metric in enumerate(ms) if idx not in (1, 2)]  # throw away individual losses
                metrics.append(ms)
            metrics = np.mean(metrics, axis=0).tolist()
            metrics += self.policy.metrics
            if self.processor is not None:
                metrics += self.processor.metrics

        if self.target_model_update >= 1 and self.step % self.target_model_update == 0:
            self.update_target_model_hard()

        return metrics

    @property
    def metrics_names(self):
        # Throw away individual losses and replace output name since this is hidden from the user.
        assert len(self.trainable_model.output_names) == 2
        dummy_output_name = self.trainable_model.output_names[1]
        model_metrics = [name for idx, name in enumerate(self.trainable_model.metrics_names) if idx not in (1, 2)]
        model_metrics = [name.replace(dummy_output_name + '_', '') for name in model_metrics]

        names = model_metrics + self.policy.metrics_names[:]
        if self.processor is not None:
            names += self.processor.metrics_names[:]
        return names

    @property
    def policy(self):
        return self.__policy

    @policy.setter
    def policy(self, policy):
        self.__policy = policy
        self.__policy._set_agent(self)



class EpisodicMemory(Memory):
    def __init__(self, limit, **kwargs):
        super(EpisodicMemory, self).__init__(**kwargs)

        self.limit = limit
        self.episodes = RingBuffer(limit)
        self.terminal = False

    def sample(self, batch_size, batch_idxs=None):
        if len(self.episodes) <= 1:
            # We don't have a complete episode yet ...
            return []

        if batch_idxs is None:
            # Draw random indexes such that we never use the last episode yet, which is
            # always incomplete by definition.
            batch_idxs = sample_batch_indexes(0, self.nb_entries - 1, size=batch_size)
        assert np.min(batch_idxs) >= 0
        assert np.max(batch_idxs) < self.nb_entries
        assert len(batch_idxs) == batch_size

        # Create sequence of experiences.
        sequences = []
        for idx in batch_idxs:
            episode = self.episodes[idx]
            while len(episode) == 0:
                idx = sample_batch_indexes(0, self.nb_entries, size=1)[0]

            # Bootstrap state.
            running_state = deque(maxlen=self.window_length)
            for _ in range(self.window_length - 1):
                running_state.append(np.zeros(episode[0].observation.shape))
            assert len(running_state) == self.window_length - 1

            states, rewards, actions, terminals = [], [], [], []
            terminals.append(False)
            for idx, timestep in enumerate(episode):
                running_state.append(timestep.observation)
                states.append(np.array(running_state))
                rewards.append(timestep.reward)
                actions.append(timestep.action)
                terminals.append(timestep.terminal)  # offset by 1, see `terminals.append(False)` above
            assert len(states) == len(rewards)
            assert len(states) == len(actions)
            assert len(states) == len(terminals) - 1

            # Transform into experiences (to be consistent).
            sequence = []
            for idx in range(len(episode) - 1):
                state0 = states[idx]
                state1 = states[idx + 1]
                reward = rewards[idx]
                action = actions[idx]
                terminal1 = terminals[idx + 1]
                experience = Experience(state0=state0, state1=state1, reward=reward, action=action, terminal1=terminal1)
                sequence.append(experience)
            sequences.append(sequence)
            assert len(sequence) == len(episode) - 1
        assert len(sequences) == batch_size
        return sequences

    def append(self, observation, action, reward, terminal, training=True):
        super(EpisodicMemory, self).append(observation, action, reward, terminal, training=training)

        # This needs to be understood as follows: in `observation`, take `action`, obtain `reward`
        # and weather the next state is `terminal` or not.
        if training:
            timestep = EpisodicTimestep(observation=observation, action=action, reward=reward, terminal=terminal)
            if len(self.episodes) == 0:
                self.episodes.append([])  # first episode
            self.episodes[self.episodes.length-1].append(timestep)
            if self.terminal:
                self.episodes.append([])
            self.terminal = terminal

    @property
    def nb_entries(self):
        return len(self.episodes)

    def get_config(self):
        config = super(SequentialMemory, self).get_config()
        config['limit'] = self.limit
        return config

    @property
    def is_episodic(self):
        return True


class AtariProcessor(Processor):
    def process_observation(self, observation):
        assert observation.ndim == 3  # (height, width, channel)
        img = Image.fromarray(observation)
        img = img.resize(INPUT_SHAPE).convert('L')  # resize and convert to grayscale
        processed_observation = np.array(img)
        assert processed_observation.shape == INPUT_SHAPE
        return processed_observation.astype('uint8')  # saves storage in experience memory

    def process_state_batch(self, batch):
        # We could perform this processing step in `process_observation`. In this case, however,
        # we would need to store a `float32` array instead, which is 4x more memory intensive than
        # an `uint8` array. This matters if we store 1M observations.
        processed_batch = batch.astype('float32') / 255.
        return processed_batch

    def process_reward(self, reward):
        return np.clip(reward, -1., 1.)


parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['train', 'test'], default='train')
parser.add_argument('--env-name', type=str, default='BreakoutDeterministic-v4')
parser.add_argument('--weights', type=str, default=None)
args = parser.parse_args()

# Get the environment and extract the number of actions.
env = gym.make(args.env_name)
np.random.seed(123)
env.seed(123)
nb_actions = env.action_space.n

# We patch the environment to be closer to what Mnih et al. actually do: The environment
# repeats the action 4 times and a game is considered to be over during training as soon as a live
# is lost.
'''
def _step(a):
    reward = 0.0
    action = env._action_set[a]
    lives_before = env.ale.lives()
    for _ in range(4):
        reward += env.ale.act(action)
    ob = env._get_obs()
    done = env.ale.game_over() or (args.mode == 'train' and lives_before != env.ale.lives())
    return ob, reward, done, {}
env._step = _step
'''

def build_model(stateful, batch_size=None):
    # Next, we build our model. We use the same model that was described by Mnih et al. (2015).
    # TODO: fix TF
    if stateful:
        input_shape = (batch_size, None, 1) + INPUT_SHAPE
    else:
        input_shape = (None, 1) + INPUT_SHAPE
    model = Sequential()
    if K.image_data_format() == 'channels_last':
        # (width, height, channels)
        if stateful:
            model.add(Permute((1, 3, 4, 2), batch_input_shape=input_shape))
        else:
            model.add(Permute((1, 3, 4, 2), input_shape=input_shape))
    elif K.image_data_format() == 'channels_first':
        # (channels, width, height)
        if stateful:
            model.add(Permute((1, 2, 3, 4), batch_input_shape=input_shape))
        else:
            model.add(Permute((1, 2, 3, 4), input_shape=input_shape))
    else:
        raise RuntimeError('Unknown image_dim_ordering.')
    model.add(TimeDistributed(Convolution2D(32, 8, 8, subsample=(4, 4))))
    model.add(Activation('relu'))
    model.add(TimeDistributed(Convolution2D(64, 4, 4, subsample=(2, 2))))
    model.add(Activation('relu'))
    model.add(TimeDistributed(Convolution2D(64, 3, 3, subsample=(1, 1))))
    model.add(Activation('relu'))
    model.add(TimeDistributed(Flatten()))
    model.add(TimeDistributed(Dense(32)))
    model.add(LSTM(32, return_sequences=True, stateful=stateful))
    model.add(TimeDistributed(Dense(nb_actions)))
    model.add(Activation('linear'))
    return model

batch_size = 32
model = build_model(stateful=True, batch_size=batch_size)
policy_model = build_model(stateful=True, batch_size=1)
print(model.summary())

# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
memory = EpisodicMemory(limit=10000, window_length=1)
processor = AtariProcessor()

# Select a policy. We use eps-greedy action selection, which means that a random action is selected
# with probability eps. We anneal eps from 1.0 to 0.1 over the course of 1M steps. This is done so that
# the agent initially explores the environment (high eps) and then gradually sticks to what it knows
# (low eps). We also set a dedicated eps value that is used during testing. Note that we set it to 0.05
# so that the agent still performs some random actions. This ensures that the agent cannot get stuck.
policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=.05,
                              nb_steps=1000000)

# The trade-off between exploration and exploitation is difficult and an on-going research topic.
# If you want, you can experiment with the parameters or use a different policy. Another popular one
# is Boltzmann-style exploration:
# policy = BoltzmannQPolicy(tau=1.)
# Feel free to give it a try!

dqn = RecurrentDQNAgent(model=model, nb_actions=nb_actions, policy=policy, memory=memory,
               processor=processor, nb_steps_warmup=50000, gamma=.99, delta_clip=(-1., 1.),
               target_model_update=10000, train_interval=500, policy_model=policy_model,
               enable_double_dqn=False, batch_size=batch_size)
dqn.compile(Adam(lr=.00025), metrics=['mae'])

if args.mode == 'train':
    # Okay, now it's time to learn something! We capture the interrupt exception so that training
    # can be prematurely aborted. Notice that you can the built-in Keras callbacks!
    weights_filename = 'dqn_{}_weights.h5f'.format(args.env_name)
    checkpoint_weights_filename = 'dqn_' + args.env_name + '_weights_{step}.h5f'
    log_filename = 'dqn_{}_log.json'.format(args.env_name)
    callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename, interval=250000)]
    callbacks += [FileLogger(log_filename, interval=100)]
    dqn.fit(env, callbacks=callbacks, nb_steps=1750000, log_interval=10000)

    # After training is done, we save the final weights one more time.
    dqn.save_weights(weights_filename, overwrite=True)

    # Finally, evaluate our algorithm for 10 episodes.
    dqn.test(env, nb_episodes=10, visualize=False)
elif args.mode == 'test':
    weights_filename = 'dqn_{}_weights.h5f'.format(args.env_name)
    if args.weights:
        weights_filename = args.weights
    dqn.load_weights(weights_filename)
    dqn.test(env, nb_episodes=10, visualize=True)
