from gym.envs.registration import register
register(
  id='powerworld-env-v0',
  entry_point='powerworld_env.envs:PowerWorldEnv',
)
