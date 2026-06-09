#!/bin/bash

qsub -v SCRIPT=cnn_cifar.py,CONFIG=cnn_cifar10/cnn_config_adamw.yaml submit.sh
qsub -v SCRIPT=cnn_cifar.py,CONFIG=cnn_cifar10/cnn_config_muon.yaml submit.sh
qsub -v SCRIPT=cnn_cifar.py,CONFIG=cnn_cifar10/cnn_config_sgd.yaml submit.sh 

qsub -l select=1:ngpus=1:gpu_cap=compute=89 -v SCRIPT=gpt_tinyshakespeare.py,CONFIG=gpt_tinyshakespeare/gpt_config_adamw.yaml submit.sh
qsub -l select=1:ngpus=1:gpu_cap=compute=89 -v SCRIPT=gpt_tinyshakespeare.py,CONFIG=gpt_tinyshakespeare/gpt_config_muon.yaml submit.sh
qsub -l select=1:ngpus=1:gpu_cap=compute=89 -v SCRIPT=gpt_tinyshakespeare.py,CONFIG=gpt_tinyshakespeare/gpt_config_sgd.yaml submit.sh

qsub -v SCRIPT=lstm_melbournetemp.py,CONFIG=lstm_melbournetemp/lstm_config_adamw.yaml submit.sh
qsub -v SCRIPT=lstm_melbournetemp.py,CONFIG=lstm_melbournetemp/lstm_config_muon.yaml submit.sh
qsub -v SCRIPT=lstm_melbournetemp.py,CONFIG=lstm_melbournetemp/lstm_config_sgd.yaml submit.sh

qsub -v SCRIPT=ppo_lunar_lander.py,CONFIG=ppo_lunar_lander/ppo_config_adamw.yaml submit.sh
qsub -v SCRIPT=ppo_lunar_lander.py,CONFIG=ppo_lunar_lander/ppo_config_muon.yaml submit.sh
qsub -v SCRIPT=ppo_lunar_lander.py,CONFIG=ppo_lunar_lander/ppo_config_sgd.yaml submit.sh

qsub -v SCRIPT=sac_bipedal_walker.py,CONFIG=sac_bipedal_walker/sac_config_adamw.yaml submit.sh
qsub -v SCRIPT=sac_bipedal_walker.py,CONFIG=sac_bipedal_walker/sac_config_muon.yaml submit.sh
qsub -v SCRIPT=sac_bipedal_walker.py,CONFIG=sac_bipedal_walker/sac_config_sgd.yaml submit.sh
