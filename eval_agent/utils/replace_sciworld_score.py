from scienceworld import ScienceWorldEnv


def step(self, inputStr:str):
    observation = self.server.step(inputStr)
    raw_score = self.server.getScore()
    score = int(round(100 * raw_score))        # Convert from 0-1 to 0-100
    isCompleted = self.server.getCompleted()
    numMoves = self.getNumMoves()

    # Calculate reward
    reward = score - self.lastStepScore         # Calculate reward (delta score) for this step
    self.lastStepScore = score                  # Store current score for reward calculation on the next step


    # If the number of moves exceeds the environment step limit, then set isCompleted to be true
    if (numMoves > self.envStepLimit):
        isCompleted = True

    # New: Handle this in the API rather than the agent -- if the score is less than zero, then set the isCompleted flag to true.
    if (score < 0):
        isCompleted = True

    # Mirror of Jericho API
    infos = {
        'moves': numMoves,
        'raw_score': raw_score,
        'score': score,
        'reward': reward,
        'look': self.look(),
        'inv': self.inventory(),
        'taskDesc': self.taskdescription(),
        'valid': self.getValidActionObjectCombinations(),
        'variationIdx': self.variationIdx,
        'taskName': self.taskName,
        'simplificationStr': self.simplificationStr,
    }

    return observation, reward, isCompleted, infos


def sciworld_monkey_patch():
    ScienceWorldEnv.step = step
    print("Monkey Patched ScienceWorldEnv.step")
