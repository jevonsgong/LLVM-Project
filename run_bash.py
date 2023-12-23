import readline
from intercode.envs import BashEnv

if __name__ == '__main__':
    env = BashEnv("intercode-nl2bash", data_path="./data/nl2bash/nl2bash_fs_1.json", traj_dir="logs/", verbose=True)

    try:
        for idx in range(3):
            env.reset()
            obs, done = env.observation, False
            while not done:
                action = input('> ')
                obs, reward, done, info = env.step(action)
    except KeyboardInterrupt:
        print("Keyboard interrupt detected")
    finally:
        env.close()