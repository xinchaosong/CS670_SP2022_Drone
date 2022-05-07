from gym.envs.registration import (
    registry,
    register,
    make,
    spec,
    load_env_plugins as _load_env_plugins,
)

# Hook to load plugins from entry points
_load_env_plugins()

# Classic
# ----------------------------------------

register(
    id="DroneBulletEnv-v0",
    entry_point="gym.envs.drone:DroneBulletEnv",
    max_episode_steps=100,
    reward_threshold=0.0,
)

register(
    id="DroneRenderEnv-v0",
    entry_point="gym.envs.drone:DroneRenderEnv",
    max_episode_steps=100,
    reward_threshold=0.0,
)

register(
    id="DroneTelloEnv-v0",
    entry_point="gym.envs.drone:DroneTelloEnv",
    max_episode_steps=100,
    reward_threshold=0.0,
)
