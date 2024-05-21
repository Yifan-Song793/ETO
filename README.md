# Exploration-Based Trajectory Optimization for LLM Agents


<p align="center">
<img src=assets/main.png width=700/>
</p>

[**🌐 Homepage**](https://huggingface.co/spaces/agent-eto/Agent-ETO) | [**🤗 Dataset**](https://huggingface.co/datasets/agent-eto/eto-sft-trajectory) | [**📖 arXiv**](https://arxiv.org/abs/2403.02502)

Official repo for [Trial and Error: Exploration-Based Trajectory Optimization for LLM Agents](https://arxiv.org/abs/2403.02502) (ACL 2024 Main Conference)

Authors: [Yifan Song](https://github.com/Yifan-Song793), [Da Yin](https://wadeyin9712.github.io/), [Xiang Yue](https://xiangyue9607.github.io/), [Jie Huang](https://jeffhj.github.io/), [Sujian Li](http://123.56.88.210/), [Bill Yuchen Lin](https://yuchenlin.xyz/).

We introduce **ETO** (Exploration-based Trajectory Optimization), an agent learning framework inspired by "trial and error" process of human learning.
ETO allows an LLM agent to iteratively collect failure trajectories and updates its policy by learning from contrastive failure-success trajectory pairs.

**ETO** has following features:
- 🕹️ **Learning by Trial and Error**
  - 🎲 **Learning from Failure Trajectories.** Contrary to previous approaches that exclusively train on successful expert trajectories, ETO allows agents to learn from their exploration failures.
  - 🎭 **Contrastive Trajectory Optimization.** ETO applies [DPO](http://arxiv.org/abs/2305.18290) loss to perform policy learning from failure-success trajectory pairs.
  - 🌏 **Iterative Policy Learning.** ETO can be expanded to multiple rounds for further policy enhancement.
- 🎖️ **Superior Performance**
  - ⚔️ **Effectiveness on Three Datasets.** ETO significantly outperforms strong baselines, such as RFT, PPO, on [WebShop](https://webshop-pnlp.github.io/), [ScienceWorld](https://sciworld.apps.allenai.org/), and [ALFWorld](https://alfworld.github.io/).
  - 🦾 **Generalization on Unseen Scenarios.** ETO demonstrates an impressive performance improvement of 22% over SFT on the challenging out-of-distribution test set in ScienceWorld.
  - ⌛ **Task-Solving Efficiency.** ETO achieves higher rewards within fewer action steps on ScienceWorld.
  - 💡 **Potential in Extreme Scenarios.** ETO shows better performance in self-play scenarios where expert trajectories are not available.




## 🧩 Structure of This Project

There are three main folders in this project: `envs`, `eval_agent`, `fastchat`

`envs`: the interaction environment of WebShop and ScienceWorld. We transform the original [WebShop](https://github.com/princeton-nlp/WebShop) repo into a package.

`eval_agent`: the evaluation framework of agent tasks, which is inspired by [MINT](https://github.com/xingyaoww/mint-bench).

`fastchat`: training scripts for SFT and DPO, which is a modified version of [FastChat](https://github.com/lm-sys/FastChat).


## 🛠️ Setup

```bash
bash setup.sh
```

The setup script performs the following actions:

- Install Python dependencies for agent training, deployment, evaluation, and the environments for WebShop, ScienceWorld, ALFWorld
- Download data and search engine indices for WebShop
- Download game files for ALFWorld
- Download expert trajectories for behavioral cloning

You can also manually download the expert trajectories from [Google Drive](https://drive.google.com/file/d/1YbhbL8RhQGDWFv5y6k1qgwRqSyFFsao8/view?usp=sharing) or [Huggingface Datasets](https://huggingface.co/datasets/agent-eto/eto-sft-trajectory).

## 🚀 Quick Start

The bash script `run_eto.sh` implements the ETO pipeline. For example, you can run:
```bash
# Optional tasks: webshop, sciworld, alfworld
bash run_eto.sh webshop <EXP_NAME> <YOUR_MODEL_PATH> <YOUR_SAVE_PATH>
```

The script performs the pipeline of ETO:

1. SFT phase: using the expert trajectories to conduct SFT to get the base agent
2. Evaluate SFT agent
   1. Launch the FastChat controller
   2. Launch the FastChat model worker
   3. Run the evaluation
   4. Kill the model worker. The controller will be reused in the following steps
3. Launch multiple FastChat model workers and let the base agent to explore the environment in parallel
4. Build contrastive failure-success trajectory pairs
5. Conduct DPO training to learn from the mistakes
6. Evaluate DPO agent
7. Repeat 3-6 to iteratively update the policy

## 🎮 Evaluation

First, launch the controller of FastChat
```bash
python -m fastchat.serve.controller
```

Then, launch the model worker of FastChat
```bash
python -m fastchat.serve.model_worker --model-path <YOUR_MODEL_PATH> --port 21002 --worker-address http://localhost:21002
```

Finally, evaluate the agent
```bash
python -m eval_agent.main --agent_config fastchat --model_name <YOUR_MODEL_NAME> --exp_config <TASK_NAME> --split test --verbose
```


## ⚙️ How to Add a New Task

1. Implement your task loader in `eval_agent/tasks`. You should implement the `load_tasks` method which returns a task generator.
2. Implement the corresponding environment in `eval_agent/envs`. The environment should parse the action generated by the LLM agent, execute the action, and return the observation. The tool/API calling should also be implemented in the environment.
3. Write the instruction prompt and ICL examples in `eval_agent/prompt`. The default setting is 1-shot evaluation.
4. Write a new task config in `eval_agent/configs/task`. The config defines which task class and environment class to load, and the settings of the environment (e.g., max action steps).


## 📑 The Data Format for Training the Agent

### SFT data

```json
[
  {
    "id": "example_0",
    "conversations": [
      {
        "from": "human",
        "value": "Who are you?"
      },
      {
        "from": "gpt",
        "value": "I am Vicuna, a language model trained by researchers from Large Model Systems Organization (LMSYS)."
      },
      {
        "from": "human",
        "value": "Have a nice day!"
      },
      {
        "from": "gpt",
        "value": "You too!"
      }
    ]
  }
]
```

### DPO data

```json
[
    {
        "id": "identity_0",
        "prompt": [
            {
                "from": "human",
                "value": "Hello"
            },
            {
                "from": "gpt",
                "value": "Hi"
            },
            {
                "from": "human",
                "value": "Have a nice day!"
            }
        ],
        "chosen": [
            {
                "from": "gpt",
                "value": "OK!"
            },
            {
                "from": "human",
                "value": "How are you?"
            },
            {
                "from": "gpt",
                "value": "I'm fine"
            }
        ],
        "rejected": [
            {
                "from": "gpt",
                "value": "No, I'm bad"
            }
        ]
    }
]
```

## 📖 Citation

If you find this repo helpful, please cite out paper:

```
@article{song2024trial,
    author={Yifan Song and Da Yin and Xiang Yue and Jie Huang and Sujian Li and Bill Yuchen Lin},
    title={Trial and Error: Exploration-Based Trajectory Optimization for LLM Agents},
    year={2024},
    eprint={2403.02502},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```
