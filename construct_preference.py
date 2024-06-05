import os
import json
import argparse


instruction_part = {
    "webshop": [
        {
            "from": "human",
            "value": "You are web shopping.\nI will give you instructions about what to do.\nYou have to follow the instructions.\nEvery round I will give you an observation and a list of available actions, you have to respond an action based on the state and instruction.\nYou can use search action if search is available.\nYou can click one of the buttons in clickables.\nAn action should be of the following structure:\nsearch[keywords]\nclick[value]\nIf the action is not valid, perform nothing.\nKeywords in search are up to you, but the value in click must be a value in the list of available actions.\nRemember that your keywords in search should be carefully designed.\nYour response should use the following format:\n\nThought: I think ...\nAction: click[something]"
        },
        {
            "from": "gpt",
            "value": "OK"
        },
    ],
    "sciworld": [
        {
            "from": "human",
            "value": "You are a helpful assistant to do some scientific experiment in an environment.\nIn the environment, there are several rooms: kitchen, foundry, workshop, bathroom, outside, living room, bedroom, greenhouse, art studio, hallway\nYou should explore the environment and find the items you need to complete the experiment.\nYou can teleport to any room in one step.\nAll containers in the environment have already been opened, you can directly get items from the containers.\n\nThe available actions are:\n    open OBJ: open a container\n    close OBJ: close a container\n    activate OBJ: activate a device\n    deactivate OBJ: deactivate a device\n    connect OBJ to OBJ: connect electrical components\n    disconnect OBJ: disconnect electrical components\n    use OBJ [on OBJ]: use a device/item\n    look around: describe the current room\n    examine OBJ: describe an object in detail\n    look at OBJ: describe a container's contents\n    read OBJ: read a note or book\n    move OBJ to OBJ: move an object to a container\n    pick up OBJ: move an object to the inventory\n    pour OBJ into OBJ: pour a liquid into a container\n    mix OBJ: chemically mix a container\n    teleport to LOC: teleport to a specific room\n    focus on OBJ: signal intent on a task object\n    wait: task no action for 10 steps\n    wait1: task no action for a step"
        },
        {
            "from": "gpt",
            "value": "OK"
        },
    ],
    "alfworld": [
        {
            "from": "human",
            "value": "Interact with a household to solve a task. Imagine you are an intelligent agent in a household environment and your target is to perform actions to complete the task goal. At the beginning of your interactions, you will be given the detailed description of the current environment and your goal to accomplish. \nFor each of your turn, you will be given the observation of the last turn. You should first think about the current condition and plan for your future actions, and then output your action in this turn. Your output must strictly follow this format:\"Thought: your thoughts.\\nAction: your next action\".\n\nThe available actions are:\n1. go to {recep}\n2. task {obj} from {recep}\n3. put {obj} in/on {recep}\n4. open {recep}\n5. close {recep}\n6. toggle {obj} {recep}\n7. clean {obj} with {recep}\n8. heat {obj} with {recep}\n9. cool {obj} with {recep}\nwhere {obj} and {recep} correspond to objects and receptacles.\nAfter your each turn, the environment will give you immediate feedback based on which you plan your next few steps. if the envrionment output \"Nothing happened\", that means the previous action is invalid and you should try more options.\n\nYour response should use the following format:\n\nThought: <your thoughts>\nAction: <your next action>"
        },
        {
            "from": "gpt",
            "value": "OK"
        }
    ],
}

# The length of the instruction part and in-context examples. We do not include them in the preference dataset.
pred_traj_offset = {
    "webshop": 10,
    "sciworld": 16,
    "alfworld": 12,
}


def build_preference(args):
    golden_raw = json.load(open(args.golden_traj_path))
    golden_trajs = {}
    golden_rewards = {}
    for item in golden_raw:
        idx = str(item['id'])
        golden_trajs[idx] = item['conversations'][2:]   # 2 is the length of the instruction part
        if args.task == 'sciworld':
            golden_rewards[idx] = 1.0
        elif args.task == 'webshop':
            golden_rewards[idx] = item['reward']

    pred_trajs = {}
    pred_rewards = {}
    for folder in os.listdir("outputs"):
        if not folder.startswith(args.model):
            continue
        for file in os.listdir(os.path.join("outputs", folder, args.task)):
            if not file.endswith('.json'):
                continue
            idx = file.split('.')[0]
            pred_raw = json.load(open(os.path.join("outputs", folder, args.task, file)))
            pred_rewards[idx] = pred_raw['meta']['reward']
            pred_trajs[idx] = pred_raw['conversations'][pred_traj_offset[args.task]:]
            if pred_trajs[idx][-1]['from'] == 'human':
                pred_trajs[idx].pop()
            if args.task == 'sciworld':
                pred_trajs[idx][0]['value'] = pred_trajs[idx][0]['value'][len("Task Description:\n"):].strip()

    pm_data = []
    win, tie, lose = 0, 0, 0
    for key in pred_rewards:
        if pred_rewards[key] is None:
            continue
        if key not in golden_trajs or key not in pred_trajs or key not in golden_rewards:
            continue
        # assert key in golden_trajs and key in pred_trajs
        # assert key in golden_rewards
        if pred_rewards[key] < golden_rewards[key]:
            lose += 1
            # Some failed trajectories will be significantly longer than successful ones. Heuristically, we limit the length of failed trajectories to not exceed the successful ones by more than 5 rounds.
            pm_data.append({
                "id": key,
                "prompt": instruction_part[args.task] + golden_trajs[key][:1],
                "chosen": golden_trajs[key][1:],
                "rejected": pred_trajs[key][:len(golden_trajs[key]) + 5*2][1:],
            })
        elif pred_rewards[key] > golden_rewards[key]:
            win += 1
            pm_data.append({
                "id": key,
                "prompt": instruction_part[args.task] + pred_trajs[key][:1],
                "chosen": pred_trajs[key][1:],
                "rejected": golden_trajs[key][1:],
            })
        else:
            tie += 1
            continue
        if pm_data[-1]['chosen'][-1]['from'] != 'gpt' or pm_data[-1]['rejected'][-1]['from'] != 'gpt':
            pm_data.pop()

    print(f"win: {win}, tie: {tie}, lose: {lose}")
    print(len(pm_data))
    json.dump(pm_data, open(args.output_path, "w"), indent=4)



def build_preference_alfworld(args):
    golden_raw = json.load(open(args.golden_traj_path))
    golden_trajs = {}
    golden_rewards = {}
    for item in golden_raw:
        idx = item['game_file'].split('/')[-1]
        golden_trajs[idx] = item['conversations'][2:]   # 2 is the length of the instruction part
        if args.task == 'sciworld' or args.task == 'alfworld':
            golden_rewards[idx] = 1.0
        elif args.task == 'webshop':
            golden_rewards[idx] = item['reward']

    pred_trajs = {}
    pred_rewards = {}
    for folder in os.listdir("outputs"):
        if not folder.startswith(args.model):
            continue
        for file in os.listdir(os.path.join("outputs", folder, args.task)):
            if not file.endswith('.json'):
                continue
            pred_raw = json.load(open(os.path.join("outputs", folder, args.task, file)))
            idx = pred_raw['meta']['error'].split('/')[-2]
            pred_rewards[idx] = 1.0 if pred_raw['meta']['reward'] else 0.0
            pred_trajs[idx] = pred_raw['conversations'][pred_traj_offset[args.task]:]
            if pred_trajs[idx][-1]['from'] == 'human':
                pred_trajs[idx].pop()
            if args.task == 'sciworld':
                pred_trajs[idx][0]['value'] = pred_trajs[idx][0]['value'][len("Task Description:\n"):].strip()

    pm_data = []
    win, tie, lose = 0, 0, 0
    for key in pred_rewards:
        if key not in golden_trajs or key not in pred_trajs or key not in golden_rewards:
            continue
        # assert key in golden_trajs and key in pred_trajs
        # assert key in golden_rewards
        if pred_rewards[key] < golden_rewards[key]:
            lose += 1
            pm_data.append({
                "id": key,
                "prompt": instruction_part[args.task] + golden_trajs[key][:1],
                "chosen": golden_trajs[key][1:],
                "rejected": pred_trajs[key][1:],
            })
        elif pred_rewards[key] > golden_rewards[key]:
            win += 1
            pm_data.append({
                "id": key,
                "prompt": instruction_part[args.task] + pred_trajs[key][:1],
                "chosen": pred_trajs[key][1:],
                "rejected": golden_trajs[key][1:],
            })
        else:
            tie += 1
            continue
        if pm_data[-1]['chosen'][-1]['from'] != 'gpt' or pm_data[-1]['rejected'][-1]['from'] != 'gpt':
            pm_data.pop()

    print(f"win: {win}, tie: {tie}, lose: {lose}")
    print(len(pm_data))
    json.dump(pm_data, open(args.output_path, "w"), indent=4)



def main():
    parser = argparse.ArgumentParser("Construct trajectory preference dataset")
    parser.add_argument(
        "--model",
        type=str,
        help="model name",
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=['webshop', 'sciworld', 'alfworld'],
        help="task name",
    )
    parser.add_argument(
        "--golden_traj_path",
        type=str,
        default="data/webshop_sft.json",
        help="path of the golden trajectory",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        help="output path of the trajectory preference dataset",
    )
    args = parser.parse_args()

    if args.task == 'alfworld':
        build_preference_alfworld(args)
    else:
        build_preference(args)


if __name__ == "__main__":
    main()
