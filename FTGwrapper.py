import os
import cv2
import gym
import signal
from collections import deque
from gym import spaces
import numpy as np
os.environ.setdefault('PATH', '')
cv2.ocl.setUseOpenCL(False)


class timeout:
    def __init__(self, seconds=1, error_message='Timeout'):
        self.seconds = seconds
        self.error_message = error_message

    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)

    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)

    def __exit__(self, type, value, traceback):
        signal.alarm(0)


class LazyFrames(object):
    def __init__(self, frames):
        """This object ensures that common frames between the observations are only stored once.
        It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay
        buffers.

        This object should only be converted to numpy array before being passed to the model.

        You'd not believe how complex the previous solution was."""
        self._frames = frames
        self._out = None

    def _force(self):
        if self._out is None:
            self._out = np.concatenate(self._frames, axis=-1)
            self._frames = None
        return self._out

    def __array__(self, dtype=None):
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self._force())

    def __getitem__(self, i):
        return self._force()[i]

    def count(self):
        frames = self._force()
        return frames.shape[frames.ndim - 1]

    def frame(self, i):
        return self._force()[..., i]


class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        """Stack k last frames.

        Returns lazy array, which is much more memory efficient.

        See Also
        --------
        baselines.common.atari_wrappers.LazyFrames
        """
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = spaces.Box(low=0, high=255, shape=(shp[:-1] + (shp[-1] * k,)), dtype=env.observation_space.dtype)

    def reset(self):
        ob = self.env.reset()
        for _ in range(self.k):
            self.frames.append(ob)
        return self._get_ob()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.k
        return LazyFrames(list(self.frames)).__array__()


class FTGWrapper(gym.Wrapper):
    def __init__(self, env, p2):
        gym.Wrapper.__init__(self, env)
        self.p2 = p2

    def reset(self):
        while True:
            try:
                with timeout(seconds=30):
                    s = self.env.reset(p2=self.p2)
                    if isinstance(s, list):
                        continue
                    if isinstance(s, np.ndarray):
                        break
            except TimeoutError:
                print("Time out to reset env")
                self.env.close()
                continue
        return s

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        # if info.get('isControl', True):
        if info.get('no_data_receive', False):
            self.env.close()
            done = False
        return ob, reward, done, info


class FTGNonstationWrapper(gym.Wrapper):
    def __init__(self, env, p2_list, total_episode=1000, stable=False):
        gym.Wrapper.__init__(self, env)
        self.p2_list = p2_list
        self.shuffled_p2 = self.p2_list
        self.total_episode = total_episode
        self.current_episode = 1
        self.p2_num = len(self.p2_list)
        self.p2 = None
        self.stable = stable
        self.create_order(self.stable)

    def create_order(self, stable=False):
        if stable:
            random_list = [self.total_episode//self.p2_num for i in range(self.p2_num)]
        else:
            np.random.shuffle(self.shuffled_p2)
            self.p2 = self.shuffled_p2[0]
            random_list = np.random.uniform(0, 1, self.p2_num)
            random_list = (random_list / np.sum(random_list) * self.total_episode).astype("int")
        random_list[-1] += self.total_episode - np.sum(random_list)
        self.random_list = random_list
        print("Mode:{}, Shuffled p2 list: {} \n p2_counters:{}".format("stable" if stable else "random", self.shuffled_p2, self.random_list))

    def reset(self):
        if self.current_episode > np.sum(self.random_list):
            self.current_episode = 1
            self.create_order(self.stable)
        for index, p2 in enumerate(self.shuffled_p2):
            if self.current_episode <= np.sum(self.random_list[0:index + 1]):
                if self.p2 != p2:
                    # need to close the env otherwise the p2 will not be changed.
                    self.env.close()
                self.p2 = p2
                break
        print("current p2: {}, current episode: {}".format(self.p2, self.current_episode))
        while True:
            try:
                with timeout(seconds=30):
                    s = self.env.reset(p2=self.p2)
                    if isinstance(s, list):
                        continue
                    if isinstance(s, np.ndarray):
                        self.current_episode += 1
                        break
            except TimeoutError:
                print("Time out to reset env")
                self.env.close()
                continue
        return s

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        # if info.get('isControl', True):
        if info.get('no_data_receive', False):
            self.env.close()
            done = False
        return ob, reward, done, info


class ImageToPyTorch(gym.ObservationWrapper):
    def __init__(self, env):
        super(ImageToPyTorch, self).__init__(env)
        old_shape = self.observation_space.shape
        new_shape = (old_shape[-1], old_shape[0], old_shape[1])
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=new_shape, dtype=np.float32)

    def observation(self, observation):
        return np.moveaxis(observation, 2, 0)


def make_ftg_display(env_name, p2, port=4000, java_env_path="."):
    env = gym.make(env_name, java_env_path=java_env_path, port=port)
    env = FTGWrapper(env, p2)
    env = FrameStack(env, 10)
    env = ImageToPyTorch(env)
    return env


def make_ftg_ram(env_name, p2, port=None, java_env_path="."):
    if port is None:
        env = gym.make(env_name, java_env_path=java_env_path)
    else:
        env = gym.make(env_name, java_env_path=java_env_path, port=port)
    env = FTGWrapper(env, p2)
    return env


def ftg_creator(env_config: dict):
    if env_config["port"] is None:
        env = gym.make(env_config["env_name"], java_env_path=env_config["java_env_path"])
    else:
        env = gym.make(env_config["env_name"], java_env_path=env_config["java_env_path"], port=env_config["port"] )
    env = FTGWrapper(env, env_config["p2"])
    return env


def make_ftg_ram_nonstation(env_name, p2_list, total_episode=100, port=None, java_env_path=".",stable=False):
    if port is None:
        env = gym.make(env_name, java_env_path=java_env_path)
    else:
        env = gym.make(env_name, java_env_path=java_env_path, port=port)
    env = FTGNonstationWrapper(env, p2_list, total_episode,stable)
    return env


