import torch
import numpy as np


class ReplayBuffer:
    def __init__(self, pool_size=1000000, frame_history_len=4):
        # the total size of the Buffer
        self.pool_size = pool_size
        # the number of memories of each observation
        self.frame_history_len = frame_history_len

        # stored memories (the list of dict)
        self.memories = None
        self.obs_shape = None

        self.number_of_memories = 0
        self.next_idx = 0

    def _check_idx(self, cur_idx):
        """
        if memory pool cannot meet "frame_history_len" frames, then padding 0.

        situation 1: cur_idx < frame_history_len and memory pool is not full      --> padding 0
        situation 2: cur_idx < frame_history_len and memory pool is full          --> no padding
        situation 3: appear "stop" flag (check from end to start)                 --> padding 0
        situation 4: other                                                        --> no padding

        :return: idx_flag, missing_context, start_idx, end_idx
        """
        end_idx = cur_idx + 1  # exclusive
        start_idx = end_idx - self.frame_history_len  # inclusive
        is_sit_3 = False

        # situation 1 or 2 or 3
        if start_idx < 0:
            start_idx = 0
            missing_context = self.frame_history_len - (end_idx - start_idx)

            # situation 1
            if self.number_of_memories != self.pool_size:
                # not check end frame
                for idx in range(start_idx, end_idx-1):
                    # 0, 1|, 0, 0|, ...
                    if self.memories[idx % self.pool_size]['done']:
                        start_idx = idx + 1
                        is_sit_3 = True

                    if is_sit_3:
                        missing_context = self.frame_history_len - (end_idx - start_idx)
                        return 3, missing_context, start_idx, end_idx

                return 1, missing_context, start_idx, end_idx

            # situation 2
            else:
                for idx in range(start_idx, end_idx-1):
                    if self.memories[idx % self.pool_size]['done']:
                        start_idx = idx + 1
                        is_sit_3 = True

                    if is_sit_3:
                        missing_context = self.frame_history_len - (end_idx - start_idx)
                        return 3, missing_context, start_idx, end_idx

                # not check end frame
                for i in range(missing_context, 0, -1):
                    idx = self.pool_size - i
                    if self.memories[idx % self.pool_size]['done']:
                        start_idx = (idx + 1) % self.pool_size
                        is_sit_3 = True

                    if is_sit_3:
                        # ..., end_idx|, ..., |start_idx, ., end
                        if start_idx > end_idx:
                            missing_context = self.frame_history_len - (self.pool_size - start_idx + end_idx)
                        else:
                            missing_context = self.frame_history_len - (end_idx - start_idx)
                        return 3, missing_context, start_idx, end_idx

                start_idx = self.pool_size - missing_context
                return 2, 0, start_idx, end_idx

        # situation 3: appear "stop" flag
        for idx in range(start_idx, end_idx-1):
            if self.memories[idx % self.pool_size]['done']:
                start_idx = idx + 1
                is_sit_3 = True

            if is_sit_3:
                missing_context = self.frame_history_len - (end_idx - start_idx)
                return 3, missing_context, start_idx, end_idx

        return 4, 0, start_idx, end_idx

    def _encoder_observation(self, cur_idx):
        """
        concatenate recent "frame_history_len" frames
        obs: (c, h, w) => (frame_history_len*c, h, w)
        :param cur_idx: current frame's index
        :return: tensor
        """

        encoded_observation = []

        idx_flag, missing_context, start_idx, end_idx = self._check_idx(cur_idx)

        if missing_context > 0:
            for i in range(missing_context):
                encoded_observation.append(np.zeros_like(self.memories[0]['obs']))

        # situation 3 in situation 2
        if start_idx > end_idx:
            for idx in range(start_idx, self.pool_size):
                encoded_observation.append(self.memories[idx % self.pool_size]['obs'])
            for idx in range(end_idx):
                encoded_observation.append(self.memories[idx % self.pool_size]['obs'])
        else:
            for idx in range(start_idx, end_idx):
                encoded_observation.append(self.memories[idx % self.pool_size]['obs'])

        # encoded_observation: [k, c, h, w] => [k*c, h, w]
        encoded_observation = np.concatenate(encoded_observation, 0)
        return encoded_observation

    def encoder_recent_observation(self):
        """
        concatenate recent "frame_history_len" frames
        :return:
        """
        assert self.number_of_memories > 0

        current_idx = self.next_idx - 1
        # when next_idx == 0
        if current_idx < 0:
            current_idx = self.pool_size - 1

        return self._encoder_observation(current_idx)

    def sample_memories(self, batch_size):
        """
        choose randomly "batch_size" memories (batch_size, )
        :param batch_size:
        :return:
        """
        # ensure s_{i+1} is exist
        sample_idxs = np.random.randint(0, self.number_of_memories-1, [batch_size])

        # [batch_size, frame_history_len*c, h, w]
        obs_batch = np.zeros(
            [batch_size, self.obs_shape[0] * self.frame_history_len, self.obs_shape[1], self.obs_shape[2]])
        next_obs_batch = np.copy(obs_batch)
        action_batch = np.zeros([batch_size, 1])  # [batch_size, ]
        reward_batch = np.zeros([batch_size, 1])  # [batch_size, ]
        done_batch = []

        for i in range(batch_size):
            obs_batch[i] = self._encoder_observation(sample_idxs[i])
            next_obs_batch[i] = self._encoder_observation(sample_idxs[i] + 1)
            action_batch[i] = self.memories[sample_idxs[i]]['action']
            reward_batch[i] = self.memories[sample_idxs[i]]['reward']
            done_batch.append(self.memories[sample_idxs[i]]['done'])

        return obs_batch, next_obs_batch, action_batch, reward_batch, done_batch

    def store_memory_obs(self, frame):
        """
        store observation of memory
        :param frame: numpy array
                      Array of shape (img_h, img_w, img_c) and dtype np.uint8
        :return:
        """
        # obs is a image (h, w, c)
        frame = frame.transpose(2, 0, 1)  # c, w, h

        if self.obs_shape is None:
            self.obs_shape = frame.shape

        if self.memories is None:
            self.memories = [dict() for i in range(self.pool_size)]

        self.memories[self.next_idx]['obs'] = frame
        index = self.next_idx

        self.next_idx = (self.next_idx + 1) % self.pool_size
        self.number_of_memories = min([self.number_of_memories + 1, self.pool_size])

        return index

    def store_memory_effect(self, index, action, reward, done):
        """
        store other information of memory
        :param action: scalar
        :param done: bool
        :param reward: scalar
        :return:
        """
        self.memories[index]['action'] = action
        self.memories[index]['reward'] = reward
        self.memories[index]['done'] = done
