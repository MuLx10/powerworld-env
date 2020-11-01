import gym
from gym import spaces
import numpy as np
import pandas as pd
import logging
from esa import SAW, PowerWorldError
from typing import Union, Tuple, List
from copy import copy, deepcopy
import itertools
import os
import matplotlib.pyplot as plt
from matplotlib.image import AxesImage
from PIL import Image

# Get full path to this directory.
THIS_DIR = os.path.dirname(os.path.abspath(__file__))

# When generating scenarios, we're drawing random generation to meet
# the load. There will be some rounding error, so set a reasonable
# tolerance. Note this is in MW. In the power flow, the slack bus will
# pick up this rounding error - that's okay.
GEN_LOAD_DELTA_TOL = 0.001

# For safety we'll have a maximum number of loop iterations.
ITERATION_MAX = 100

# Constants related to PowerWorld (for convenience and cleanliness):
# Constant power portion of PowerWorld loads.
LOAD_P = ['LoadSMW', 'LoadSMVR']

# Constant current and constant impedance portions of PowerWorld
# loads.
LOAD_I_Z = ['LoadIMW', 'LoadIMVR', 'LoadZMW', 'LoadZMVR']

# Assumed transmission system losses as a fraction of energy delivered.
LOSS = 0.03

# Specify bus voltage bounds.
LOW_V = 0.95
HIGH_V = 1.05
NOMINAL_V = 1.0
V_TOL = 0.0001

# Lines which are allowed to be opened in the 14 bus case for some
# environments.
LINES_TO_OPEN_14 = ((1, 5, '1'), (2, 3, '1'), (4, 5, '1'), (7, 9, '1'))

# Maps for open/closed states.
STATE_MAP = {
  1: 'Closed',
  0: 'Open',
  True: 'Closed',
  False: 'Open',
  1.0: 'Closed',
  0.0: 'Open'
}
STATE_MAP_INV = {'Closed': True, 'Open': False}

# Some environments may reject scenarios with a certain voltage range.
MIN_V = 0.7
MAX_V = 1.2

# Some environments may min/max scale voltages.
MIN_V_SCALED = 0
MAX_V_SCALED = 1
# https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html
V_SCALER = (MAX_V_SCALED - MIN_V_SCALED) / (MAX_V - MIN_V)
V_ADD_TERM = MIN_V_SCALED - MIN_V * V_SCALER

SCENARIO_INIT_ATTRIBUTES = ['total_load_mw', 'loads_mw', 'loads_mvar',
                                'shunt_states', 'gen_mw', 'gen_v',
                                'branches_to_open']


INIT_FIELDS = {
  'gen': ['BusCat', 'GenMW', 'GenMVR', 'GenVoltSet', 'GenMWMax','GenMWMin', 'GenMVRMax', 'GenMVRMin', 'GenStatus','GenRegNum'],
  'load': LOAD_P + LOAD_I_Z,
  'bus': [],
  'branch': [],
  'shunt': ['AutoControl', 'SSStatus'],
  'ltc': ['XFAuto', 'XFRegMin', 'XFRegMax', 'XFTapMin','XFTapMax', 'XFStep', 'XFTapPos', 'XFTapPos:1','XFTapPos:2']
}

OBS_FIELD = {
  'gen': ['GenMW', 'GenMWMax', 'GenMVA', 'GenMVR', 'GenStatus', 'GenVoltSet'],
  'load': LOAD_P + ['PowerFactor', ],
  'bus': ['BusPUVolt', ],
  'branch': [],
  'shunt': ['SSStatus'],
  'ltc': ['XFTapPos']
}

RESET_FIELD = {
  'gen': ['GenMW', 'GenStatus', 'GenVoltSet'],
  'load': LOAD_P,
  'bus': [],
  'branch': [],
  'shunt': []
}


REWARDS = {
  # Negative reward (penalty) given for taking an action.
  "action": -10,
  # If no action is taken, give a reward if all voltages are in
  # bounds, otherwise penalize.
  "no_op": 50,
  # Reward per 0.01 pu voltage movement in the right direction
  # (i.e. a voltage below the lower bound moving upward).
  "v_delta": 1,
  # Bonus reward for moving a voltage that was not in-bounds
  # in-bounds. This is set to be equal to the action penalty so
  # that moving a single bus in-bounds makes an action worth it.
  "v_in_bounds": 10,
  # Penalty for moving a voltage that was in-bounds out-of-bounds.
  "v_out_bounds": -10,
  # Reward per 1% increase in generator var reserves (or penalty
  # per 1% decrease).
  "gen_var_delta": 1,
  # Penalty for taking an action which causes the power flow to
  # fail to converge (or voltages get below the MIN_V threshold).
  "fail": -1000
}

def _scale_voltages(arr_in):
  return V_SCALER * arr_in + V_ADD_TERM

def _get_voltage_masks(v_prev, v_now, low_v, high_v):
  low_v_prev = v_prev < low_v
  high_v_prev = v_prev > high_v
  low_v_now = v_now < low_v
  high_v_now = v_now > high_v

  in_prev = (~low_v_prev) & (~high_v_prev)    # in bounds before
  out_prev = low_v_prev | high_v_prev         # out of bounds before
  in_now = (~low_v_now) & (~high_v_now)       # in bounds now
  out_now = low_v_now | high_v_now            # out of bounds now

  v_diff = v_prev - v_now
  moved_up = v_diff < 0
  moved_down = v_diff > 0

  in_out = in_prev & out_now              # in before, out now
  out_in = out_prev & in_now              # out before, in now
  in_out_low = in_prev & low_v_now        # in before, low now
  in_out_high = in_prev & high_v_now      # in before, high now
  out_right_d = (out_prev & ((high_v_prev & moved_down) | (low_v_prev & moved_up)))
  out_wrong_d = (out_prev & ((high_v_prev & moved_up) | (low_v_prev & moved_down)))

  return  {
    # 'low_v_prev': low_v_prev, 'high_v_prev': high_v_prev,
    # 'low_v_now': low_v_now, 'high_v_now': high_v_now,
    # 'in_prev': in_prev, out_prev: 'out_prev', 'in_now': in_now,
    # 'out_now': out_now, 'moved_up': moved_up, 'moved_down': moved_down,
    'in_out': in_out, 'out_in': out_in, 'in_out_low': in_out_low,
    'in_out_high': in_out_high, 'out_right_d': out_right_d,
    'out_wrong_d': out_wrong_d,   # 'over_under_shoot': over_under_shoot
  }


class PowerWorldEnv(gym.Env):
  """docstring for PowerWorldEnv"""
  def __init__(self, 
              pwb_path,
              num_scenarios = 0,
              max_load_factor = None,
              min_load_factor = None,
              min_load_pf = 0.8,
              lead_pf_probability = 0.1,
              load_on_probability = 0.8,
              shunt_closed_probability = 0.6,
              num_gen_voltage_bins = 5,
              gen_voltage_range = (0.9, 1.1),
              seed = None,
              log_level=logging.INFO,
              rewards = None,
              dtype = np.float32,
              low_v = LOW_V,
              high_v = HIGH_V,
              oneline_axd = None,
              contour_axd = None,
              image_dir = None,
              render_interval = 1.0,
              log_buffer = 10000,
              csv_logfile = 'log.csv',
              truncate_voltages = False,
              scale_voltage_obs = False,
              clipped_reward = False,
              vtol = V_TOL,
              no_op_flag = False):
    super(PowerWorldEnv, self).__init__()

    self.log = logging.getLogger(self.__class__.__name__)
    self.log.setLevel(log_level)
    self.log.debug('PowerWorld case loaded.')

    self.rng = np.random.default_rng(seed)

    self.pwb_path = pwb_path
    self.saw = SAW(self.pwb_path, early_bind=True)

    self.dtype = dtype

    self.num_scenarios = num_scenarios

    self.scenario_idx = 0

    self.low_v = low_v - vtol
    self.high_v = high_v + vtol

    self.log_buffer = log_buffer
    self.csv_logfile = csv_logfile

    self.min_load_factor = min_load_factor
    self.max_load_factor = max_load_factor

    self.shunt_closed_probability = shunt_closed_probability

    self.reset_successes = 0
    self.reset_failures = 0

    self.no_op_action = None

    self.last_action = None

    self.no_op_flag = no_op_flag

    self.oneline_axd = oneline_axd
    self.contour_axd = contour_axd

    self.oneline_name = 'my oneline'

    if image_dir is None:
      self.image_dir = 'render_images'
    else:
      self.image_dir = image_dir

    self.render_interval = render_interval
    self._render_flag = False

    self.truncate_voltages = truncate_voltages
    self.scale_voltage_obs = scale_voltage_obs
    self.clipped_reward = clipped_reward

    self.gen_bins = np.linspace(gen_voltage_range[0], gen_voltage_range[1], num_gen_voltage_bins)
    
    
    self.nums = {}
    self.init_data = {}
    self.obs_data = {}
    self.reset_data = {}
    
    self.obs_data_prev = None
    self.nums['gen_voltage_bins'] = num_gen_voltage_bins

    self.init_attributes()

    self.action_space = self._get_action_space()
    self.observation_space = self._get_obs_space()


    self.saw.SolvePowerFlow()
    self.saw.SaveState()


  def init_attributes(self):
    self.current_reward = np.nan
    self.action_count = 0
    self.cumulative_reward = 0
    self.rewards = deepcopy(REWARDS)

    for obj in ('gen', 'load', 'bus', 'shunt'):
      self.nums[obj] = 0
      kf = self.saw.get_key_field_list(obj)
      if(obj in list(INIT_FIELDS.keys())):
        data = self.saw.GetParametersMultipleElement(
            ObjectType=obj, ParamList=kf+INIT_FIELDS[obj])
        if data is not None:
          self.init_data[obj] = data
          self.nums[obj] = data.shape[0]

    self.gen_mw_capacity = self.init_data['gen']['GenMWMax'].sum()
    self.gen_mvar_produce_capacity = self.init_data['gen']['GenMVRMax'].sum()
    self.gen_mvar_consume_capacity = self.init_data['gen']['GenMVRMin'].sum()


    if (self.init_data['load'][LOAD_I_Z] != 0.0).any().any():
      self.log.warning('The given PowerWorld case has loads with '
                       'non-zero constant current and constant impedance'
                       ' portions. These will be zeroed out.')
      self.init_data['load'].loc[:, LOAD_I_Z] = 0.0
      kf = self.saw.get_key_field_list('load')
      self.saw.change_and_confirm_params_multiple_element('Load', self.init_data['load'].loc[:, kf + LOAD_I_Z])

    if self.nums['shunt'] > 0:
      self.init_data['shunt']['AutoControl'] = 'NO'
      self.saw.change_parameters_multiple_element_df('shunt', self.init_data['shunt'])

    if self.max_load_factor is not None:
      self.max_load_mw = self.init_data['load']['LoadSMW'].sum() * self.max_load_factor
      if self.max_load_mw * (1 + LOSS) >= self.gen_mw_capacity:
        raise MaxLoadAboveMaxGenError(str(self.gen_mw_capacity) + ' ' + str(self.max_load_mw * (1 + LOSS)))
      gen_factor = self.gen_mw_capacity / self.max_load_mw
      if gen_factor >= 1.5:
          self.log.warning('The given generator capacity '+str(gen_factor))   
    else:
      self.max_load_mw = self.gen_mw_capacity

    if self.min_load_factor is not None:
      self.min_load_mw = self.init_data['load']['LoadSMW'].sum() * self.min_load_factor
      min_gen = self.init_data['gen']['GenMWMin'].min()
      if self.min_load_mw < min_gen:
        raise MinLoadBelowMinGenError(str(self.min_load_factor)+' '+str(self.min_load_mw)+' '+str(min_gen))
    else:
      self.min_load_mw = self.init_data['gen']['GenMWMin'].min()


    self.gen_dup_reg = self.init_data['gen'].duplicated('GenRegNum', 'first')
    self.nums['gen_reg_buses'] = (~self.gen_dup_reg).sum()

    self.gen_action_array = np.zeros(shape=(self.nums['gen_reg_buses'] * self.nums['gen_voltage_bins'], 2), dtype=int)

    self.gen_action_array[:, 0] = np.tile(self.init_data['gen'].loc[~self.gen_dup_reg, 'BusNum'].to_numpy(), self.nums['gen_voltage_bins'])
    for i in range(self.nums['gen_voltage_bins']):
      s_idx = i * self.nums['gen_reg_buses']
      e_idx = (i + 1) * self.nums['gen_reg_buses']
      self.gen_action_array[s_idx:e_idx, 1] = i

    self.action_cap = 2 * self.nums['gen'] + 2 * self.nums['shunt']
    self.action_cap *= 100



  def _get_action_space(self):
    self.nums['action'] = int(self.nums['gen_reg_buses'] * self.nums['gen_voltage_bins'] + self.nums['shunt'] + 1)
    return spaces.Discrete(self.nums['action'])

  def _get_obs_space_shunts(self):
    self.nums['observation'] = self.nums['bus'] + self.nums['gen'] + self.nums['shunt']
    low = np.zeros(self.nums['observation'], dtype=self.dtype)
    high = np.ones(self.nums['observation'], dtype=self.dtype)
    if not self.scale_voltage_obs:
        high[0:self.nums['bus']] = 2

    return spaces.Box(low=low, high=high, dtype=self.dtype)


  def _get_obs_space(self):
    self.nums['observation'] = self.nums['bus'] + 3 * self.nums['gen'] + 3 * self.nums['load']
    low = np.zeros(self.nums['observation'], dtype=self.dtype)
    bus_high = np.ones(self.nums['bus'], dtype=self.dtype)
    if not self.scale_voltage_obs:
      bus_high += 1
    rest_high = np.ones(3 * self.nums['gen'] + 3 * self.nums['load'], dtype=self.dtype)
    return spaces.Box(low=low, high=np.concatenate((bus_high, rest_high)),dtype=self.dtype)


  def _get_observation(self):
    self.obs_data['load']['lead'] = (self.obs_data['load']['LoadSMVR'] < 0).astype(self.dtype)

    bus_pu_volt_arr = self.obs_data['bus']['BusPUVolt'].to_numpy(dtype=self.dtype)

    if self.scale_voltage_obs:
      bus_pu_volt_arr = _scale_voltages(bus_pu_volt_arr)

    return np.concatenate([
        bus_pu_volt_arr,
        (self.obs_data['gen']['GenMW'] / self.obs_data['gen']['GenMWMax']).to_numpy(dtype=self.dtype),
        (self.obs_data['gen']['GenMW'] / self.obs_data['gen']['GenMVA']).fillna(1).to_numpy(dtype=self.dtype),
        self.obs_data['gen']['GenMVR'].to_numpy(dtype=self.dtype) / 100,
        (self.obs_data['load']['LoadSMW'] / self.max_load_mw).to_numpy(dtype=self.dtype),
        self.obs_data['load']['PowerFactor'].to_numpy(dtype=self.dtype),
        self.obs_data['load']['lead'].to_numpy(dtype=self.dtype)
        #(self.obs_data['shunt']['SSStatus'] == 'Closed').to_numpy(dtype=self.dtype)
        #(self.obs_data['branch']['LineStatus'] == 'Closed').to_numpy(dtype=self.dtype)
    ])


  def _solve_and_observe(self):
    self.saw.SolvePowerFlow()
    self.obs_data_prev = copy(self.obs_data)
    for obj in ['bus', 'gen', 'load', 'shunt']:
      kf = self.saw.get_key_field_list(obj)
      self.obs_data[obj] = self.saw.GetParametersMultipleElement(
        ObjectType=obj, ParamList=kf + OBS_FIELD[obj])
    return self._get_observation()


  def _take_action_gens(self, action):
    gen_bus = self.gen_action_array[action, 0]
    voltage = self.gen_bins[self.gen_action_array[action, 1]]

    gen_init_data_mi = self.init_data['gen'].set_index(['BusNum', 'GenID'])
    gens = gen_init_data_mi.loc[(gen_bus,), :]

    if gens.shape[0] == 1:
      self.saw.ChangeParametersSingleElement(
        ObjectType='gen',
        ParamList=['BusNum', 'GenID', 'GenVoltSet'],
        Values=[gen_bus, gens.index[0], voltage]
      )
    else:
      self.saw.ChangeParametersMultipleElement(
        ObjectType='gen',
        ParamList=['BusNum', 'GenID', 'GenVoltSet'],
        ValueList=[[gen_bus, gen_id, voltage] for gen_id in
                   gens.index]
      )

  def _take_action_shunts(self, action):
    shunt_status_arr = (self.obs_data['shunt']['SSStatus'] == 'Closed').to_numpy(dtype=self.dtype)[action]
    new = STATE_MAP[not shunt_status_arr]

    kf = self.saw.get_key_field_list('shunt')
    kfv = self.init_data['shunt'].iloc[action][kf].tolist()

    self.saw.ChangeParametersSingleElement(
        ObjectType='shunt',
        ParamList=kf + ['SSStatus'],
        Values=kfv + [new]
    )

  def _take_action_ltcs(self, action):
    pass

  def _take_action(self, action):
    if action == self.no_op_action:
      return

    action -= 1
    if action < self.gen_action_array.shape[0] and action > -1:
      self._take_action_gens(action)
      return

    action -= self.gen_action_array.shape[0]
    if action < self.nums['shunt'] and action > -1:
      self._take_action_shunts(action)
      return

    action -= self.nums['shunt']
    if action > -1:
      self._take_action_ltcs(action)


  

  def _compute_reward_volt_change(self):
    if self.last_action == self.no_op_action:
      if self._check_congestion():
        return self.rewards['no_op']
      else:
        return -self.rewards['no_op']

    reward = self.rewards['action']
    v_prev = self.obs_data_prev['bus']['BusPUVolt']
    v_now = self.obs_data['bus']['BusPUVolt']
    nom_delta_diff = ((v_prev - NOMINAL_V).abs() - (v_now - NOMINAL_V).abs()).abs() * 100
    d = _get_voltage_masks(
        v_prev=v_prev.to_numpy(dtype=self.dtype),
        v_now=v_now.to_numpy(dtype=self.dtype),
        low_v=self.low_v,
        high_v=self.high_v)
    reward += (nom_delta_diff[d['out_right_d']] * self.rewards['v_delta']).sum()
    reward -= (nom_delta_diff[d['out_wrong_d']] * self.rewards['v_delta']).sum()
    reward += ((v_now[d['in_out_low']] - LOW_V) / 0.01 * self.rewards['v_delta']).sum()
    reward += ((HIGH_V - v_now[d['in_out_high']]) / 0.01 * self.rewards['v_delta']).sum()
    reward += d['in_out'].sum() * self.rewards['v_out_bounds']
    reward += d['out_in'].sum() * self.rewards['v_in_bounds']
    return reward

  def _compute_reward_volt_change_clipped(self):
    if self.last_action == self.no_op_action:
        return 0.0
    if self._check_congestion():
        return 1.0
    v_prev = self.bus_pu_volt_arr_prev
    v_now = self.bus_pu_volt_arr
    d = _get_voltage_masks(v_prev=v_prev, v_now=v_now, low_v=self.low_v, high_v=self.high_v)
    out_in_sum = d['out_in'].sum()
    in_out_sum = d['in_out'].sum()
    if (out_in_sum > in_out_sum) and (out_in_sum > 0):
        net = out_in_sum - in_out_sum
        if net > 1:            
          return 0.75
        else:            
          return 0.5
    if (in_out_sum > out_in_sum) and (in_out_sum > 0):
        net = in_out_sum - out_in_sum
        if net > 1:            
          return -0.75
        else:            
          return -0.5
    right_d_sum = d['out_right_d'].sum()
    wrong_d_sum = d['out_wrong_d'].sum()
    if (right_d_sum > wrong_d_sum) and (right_d_sum > 0):
        return 0.25
    if (wrong_d_sum > right_d_sum) and (wrong_d_sum > 0):
        return -0.25
    return -0.1


  def _compute_reward(self, failed = False):
    if self.clipped_reward:
      if failed:
        return -1.0
      return self._compute_reward_volt_change_clipped()

    if failed:
      return self.rewards['fail'] + self.rewards['action']
    return self._compute_reward_volt_change()

  def _check_congestion(self):
    return self.obs_data['bus']['BusPUVolt'].between(self.low_v, self.high_v, inclusive=True).all()

  def _check_done(self):
    if self.action_count >= self.action_cap:
      return True
    if self._check_congestion(): # self.all_v_in_range:
      return True
    return False


  def _add_to_log(self, action):
    # self.log_array[self.log_idx, 0] = self.scenario_idx - 1
    # self.log_array[self.log_idx, 1] = action
    # self.log_array[self.log_idx, 2] = self.current_reward
    # self.log_array[self.log_idx, 3:] = np.concatenate((self.bus_pu_volt_arr, self.gen_volt_set_arr))
    # self.log_idx += 1

    # if self.log_idx == self.log_buffer:
    #   self._flush_log()
    pass

  def step(self, action):
    self.action_count += 1
    self.last_action = action
    self._take_action(action)
    info = dict()
    try:
      obs = self._solve_and_observe()
    except PowerWorldError:
      obs = self._get_observation()
      obs[0:self.nums['bus']] = 0.0
      done = True
      reward = self._compute_reward(failed=True)
      info['is_success'] = False
      info['error'] = 'PowerWorldError'
    else:
      reward = self._compute_reward()
      done = self._check_done()
      if done and self._check_congestion(): #self.all_v_in_range:
        info['is_success'] = True
      else:
        info['is_success'] = False

    if self.no_op_flag and (action == self.no_op_action):
      reward = 0.0
      done = True

    self.cumulative_reward += reward

    info['cumulative_reward'] = self.cumulative_reward
    if done:
      eor = 0 # self._compute_end_of_episode_reward()
      if eor is not None:
        reward += eor
        self.cumulative_reward += eor

    self.current_reward = reward
    info['reward'] = self.current_reward
    self._add_to_log(action=action)
    return obs, reward, done, info

  def reset(self):
    self.action_count = 0
    self.current_reward = np.nan
    self.cumulative_reward = 0
    self.last_action = None

    done = False
    obs = None
    
    self.saw.exit()
    
    self.saw = SAW(self.pwb_path, early_bind=True)
    self.log.debug('PowerWorld case loaded.')

    # self.saw.LoadState()
    # self.saw.SolvePowerFlow()

    try:
      obs = self._solve_and_observe()
    except PowerWorldError as exc:
      obs = None
    else:
      done = True
    finally:
      pass
    self._add_to_log(action=np.nan)
    return obs

  def render(self, mode='human', close=False):
    if (self.oneline_axd is None) or (self.contour_axd is None):
      self.log.error('Cannot render without providing the "oneline_axd" '
                     'and "contour_axd" parameters during environment '
                     'initialization. Rendering will not occur.')
      return

    # plt.ion()
    # plt.show()
    # plt.axis('off')

    # self.fig, self.ax = plt.subplots(frameon=False)
    # self.fig.canvas.manager.full_screen_toggle()
    # self.ax.set_axis_off()

    # self.saw.RunScriptCommand('LoadAXD("'+self.oneline_axd+'", "'+self.oneline_name+'");')

    # try:
    #   os.mkdir(self.image_dir)
    # except FileExistsError:
    #   self.log.warning('The directory '+self.image_dir+' already exists. Existing images will be overwritten.')

    # self.saw.RunScriptCommand('LoadAXD("'+self.contour_axd+'", "'+self.oneline_name+'");')

    # self.image_path = os.path.join(self.image_dir, 'episode_action_'+str(self.action_count)+'.bmp')


    # self.saw.RunScriptCommand(
    #   fr'ExportOneline("{self.image_path}", "{self.oneline_name}", BMP, ' + r'"", YES, YES)'
    # )

    # self.image = Image.open(self.image_path)

    # if self.image_axis is None:
    #   self.image_axis = self.ax.imshow(self.image)
    # else:
    #   self.image_axis.set_data(self.image)

    # txt = (f'Episode: {self.scenario_idx}, Action Count: '
    #        f'{self.action_count}, Current Reward: {self.current_reward}')

    # self.ax.set_title(txt, fontsize=32, fontweight='bold')
    # # self.fig.suptitle(txt, fontsize=32, fontweight='bold')
    # # self.ax.text(
    # #     0.1, 0.9, txt, color='black',
    # #     bbox=dict(facecolor='white', edgecolor='black'))
    # plt.tight_layout()
    # plt.draw()
    # plt.pause(self.render_interval)




class Error(Exception):
  """Base class for exceptions in this module."""
  pass


class MinLoadBelowMinGenError(Error):
  """Raised if an environment's minimum possible load is below the
  minimum possible generation.
  """
  pass


class MaxLoadAboveMaxGenError(Error):
  """Raised if an environment's maximum possible load is below the
  maximum possible generation.
  """


class OutOfScenariosError(Error):
  """Raised when an environment's reset() method is called to move to
  the next episode, but there are none remaining.
  """


class ComputeGenMaxIterationsError(Error):
  """Raised when generation for a given scenario/episode cannot be
  computed within the given iteration limit.
  """