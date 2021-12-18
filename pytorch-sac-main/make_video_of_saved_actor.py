import gym
from gym.wrappers import Monitor
from params_pool import ParamsPool
from action_wrappers import ScalingActionWrapper
import argparse

def make_video_of_saved_actor(save_dir: str, run_id:int, filename: str) -> None:


    env_raw = gym.make('Pendulum-v0')

    env = Monitor(
        ScalingActionWrapper(
            env_raw,
            scaling_factors=env_raw.action_space.high
        ),
        #directory=f'results/trained_policies_video/{run_id}',
        directory=f'results/trained_policies_video/' + str(run_id),
        force=True,
        video_callable=False
    )

    print('hello1')
    print(save_dir)
    print(filename)

    param = ParamsPool(
        input_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0]
    )
    param.load_actor(save_dir=save_dir, filename=filename)  # critics are not loaded and act does not depend on them

    
    print('hello2')

    obs = env.reset()

    while True:
        next_obs, reward, done, _ = env.step(param.act(obs))
        if done: break
        obs = next_obs

    env.close()

if __name__ == '__main__':
    make_video_of_saved_actor(
        save_dir='results/trained_policies_pth/',
        #filename=f'{run_id}.pth'
        run_id = 3,
        filename='3.pth'
    )