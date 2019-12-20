from asyncio import Queue
from enum import Enum
import numpy as np
import torch
import math
import csv
from datetime import datetime

import numpy as np

producer_queue = Queue(maxsize=1)
consumer_queue = Queue(maxsize=1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ActionSpace(Enum):
  STOP = 0
  STRONGFORWARD = 1
  WEAKFORWARD = 2
  STRONGBACK = 3
  WEAKBACK = 4
  STRONGLEFT = 5
  WEAKLEFT = 6
  STRONGRIGHT = 7
  WEAKRIGHT = 8

MOTOR_SPEED_CHANGE = 1

class Commands(Enum):
  STOP = 0
  STRONGFORWARD = 1
  WEAKFORWARD = 2
  STRONGBACK = 3
  WEAKBACK = 4
  STRONGLEFT = 5
  WEAKLEFT = 6
  STRONGRIGHT = 7
  WEAKRIGHT = 8
  RESET = 9
  WINDRESET = 10

class VelStates(Enum):
  NO_ACCEL = 0
  STRONGFORWARD = 1
  WEAKFORWARD = 2
  STRONGBACK = 3
  WEAKBACK = 4
  STRONGLEFT = 5
  WEAKLEFT = 6
  STRONGRIGHT = 7
  WEAKRIGHT = 8

class AngVelStates(Enum):
  NONE = 0
  STRONGLEFT = 1
  STRONGRIGHT = 2
  WEAKLEFT = 3
  WEAKRIGHT = 4

# abbrevations for motors, U = up, N = neutral, D = down
class Actions(Enum):
  UU = 0
  UN = 1
  UD = 2
  NU = 3
  NN = 4
  ND = 5
  DU = 6
  DN = 7
  DD = 8

def changeMotorSpeed(left_motor_speed, right_motor_speed, action):
  if action == Actions.UU:
    left_motor_speed += MOTOR_SPEED_CHANGE
    right_motor_speed += MOTOR_SPEED_CHANGE
  elif action == Actions.UN:
    left_motor_speed += MOTOR_SPEED_CHANGE
  elif action == Actions.UD:
    left_motor_speed += MOTOR_SPEED_CHANGE
    right_motor_speed -= MOTOR_SPEED_CHANGE

  elif action == Actions.NU:
    right_motor_speed += MOTOR_SPEED_CHANGE
  elif action == Actions.ND:
    right_motor_speed -= MOTOR_SPEED_CHANGE

  elif action == Actions.DU:
    left_motor_speed -= MOTOR_SPEED_CHANGE
    right_motor_speed += MOTOR_SPEED_CHANGE
  elif action == Actions.DN:
    left_motor_speed -= MOTOR_SPEED_CHANGE
  elif action == Actions.DD:
    left_motor_speed -= MOTOR_SPEED_CHANGE
    right_motor_speed -= MOTOR_SPEED_CHANGE

  if left_motor_speed >= 127:
    left_motor_speed = 127
  elif left_motor_speed <= -127:
    left_motor_speed = -127

  if right_motor_speed >= 127:
    right_motor_speed = 127
  elif right_motor_speed <= -127:
    right_motor_speed = -127
  return (left_motor_speed, right_motor_speed)

COMMAND_MAP = [
  {
    "leftPowerLevel": '0',
    "rightPowerLevel": '0'
  },
  {
    "leftPowerLevel": '127',
    "rightPowerLevel": '127'
  },
  {
    "leftPowerLevel": '64',
    "rightPowerLevel": '64'
  },
  {
    "leftPowerLevel": '-127',
    "rightPowerLevel": '-127'
  },
  {
    "leftPowerLevel": '-64',
    "rightPowerLevel": '-64'
  },
  {
    "leftPowerLevel": '-127',
    "rightPowerLevel": '127'
  },
  {
    "leftPowerLevel": '-64',
    "rightPowerLevel": '64'
  },
  {
    "leftPowerLevel": '127',
    "rightPowerLevel": '-127'
  },
  {
    "leftPowerLevel": '64',
    "rightPowerLevel": '-64'
  },
  {
    "reset": True
  },
  {
    "windReset": True
  }
]

def calculateStateAndReward(acceleration, angularVelocity, linearVelocity):
    speed = math.sqrt(math.pow(linearVelocity['x'], 2) + math.pow(linearVelocity['y'], 2))
    accmag = math.sqrt(math.pow(acceleration['x'], 2) + math.pow(acceleration['y'], 2))
    reward = 0.0
    # print(speed)
    
    state = np.array([acceleration['x'], acceleration['y'], linearVelocity['x'], 
        linearVelocity['y'], angularVelocity, speed])
    # if speed < 0.1:
    #     reward = 100.0
    # elif speed < 0.25:
    reward = 3.0 - (4 * speed) ** 2 - (10000 * accmag) ** 2 - (100 * angularVelocity) ** 2

    return (state, reward, speed < 0.1 and abs(angularVelocity) < 0.0005)

async def sendCommand(command):
    await producer_queue.put(command)

async def getNextState(lms, rms):
    await producer_queue.put({
        "leftPowerLevel": str(lms),
        "rightPowerLevel": str(rms),
    })
    data_package = await consumer_queue.get()
    state, reward, done = calculateStateAndReward(**data_package)
    return (state, reward, done, {})

async def getState():
    data_package = await consumer_queue.get()
    state, _, _ = calculateStateAndReward(**data_package)
    return state


def now_str(str_format='%Y%m%d%H%M'):
    return datetime.now().strftime(str_format)


def idx2mask(idx, max_size):
    mask = np.zeros(max_size)
    mask[idx] = 1.0
    return mask


class RecordHistory:
    def __init__(self, csv_path, header):
        self.csv_path = csv_path
        self.header = header

    def generate_csv(self):
        with open(self.csv_path, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(self.header)

    def add_histry(self, history):
        history_list = [history[key] for key in self.header]
        with open(self.csv_path, 'a') as f:
            writer = csv.writer(f)
            writer.writerow(history_list)

    def add_list(self, array):
        with open(self.csv_path, 'a') as f:
            writer = csv.writer(f)
            writer.writerow(array)
