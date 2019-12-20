Using pytorch to implement [Deep Deterministic Policy Gradient(DDPG)](https://arxiv.org/abs/1509.02971).

## Denpendency
- python 3.6
- pytorch 0.4+

## Setup using Docker

Setup with Docker is not required to run this program; simply installing the correct dependencies will work also.

If `nvidia-docker` is already installed, then simply running `docker build` will install all the correct dependencies. Once the build is complete (might take a while), running `sudo bash ./linux_run_docker.sh` will start docker with the correct ports open. 

## Train
```
python main.py --train --cuda
```
Parameters:

|Parameters    | description        |
|---------     |-----------         |
|  --train     |  train model       |
|  --test      |  test model        |
|  --retrain   |      retrain model |
|  --retrain_model |   retrain model path   |
|  --episodes  | train episodes             |
|  --eps_decay | noise epsilon decay        |
|  --cuda      |                use cuda    |
|  --model_path|    if test mode, import the model |
|  --record_ep_interval   | record episodes interval |
|  --checkpoint           |  use model checkpoint    |
|  --checkpoint_interval  |    checkpoint interval   |

(more parameters see the file)

## Test
You can test your model with `--test` like this:
```
main.py --test --model_path out/MountainCarContinuous-v0-run0
```
Note that this isn't re-implemented yet.

## Running with the Blimp Simulator

If the blimp simulator is already running, (running `yarn run start` in the repo's folder will start it), then running the simulator with the above command and refreshing the blimp simulator web page will start the simulation. The simulation will only connect once at the start, so don't refresh the web page while it's running. Using other tabs is fine however.

## Result
It turns out that tuning parameters are very important, especially `eps_decay`.  I use the simple linear noise decay such as `epsilon -= eps_decay` every episode.

```
python main.py --train --cuda --eps_decay 0.01
```

## Reference
- paper [Continuous control with deep reinforcement learning](https://arxiv.org/abs/1509.02971)



