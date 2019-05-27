from enum import Enum
import numpy as np


"""
状態(セルの位置)＆行動(上下左右の行動を表現)の定義
"""
class State():

    def __init__(self, row=-1, column=-1):
        self.row = row
        self.column = column

    def __repr__(self):
        return "<State: [{}, {}]>".format(self.row, self.column)

    def clone(self):
        return State(self.row, self.column)

    def __hash__(self):
        return hash((self.row, self.column))

    def __eq__(self, other):
        return self.row == other.row and self.column == other.column


# Enum = enumerationの略
class Action(Enum):
    UP = 1
    DOWN = -1
    LEFT = 2
    RIGHT = -2


"""
環境の実装
"""
class Environment():

    def __init__(self, grid, move_prob=0.8):
        # grid is 2d-array. Its values are treated as an attribute.
        # Kinds of attribute is following.
        #  0: ordinary cell
        #  -1: damage cell (game end)
        #  1: reward cell (game end)
        #  9: block cell (can't locate agent)
        self.grid = grid
        self.agent_state = State()

        # Default reward is minus. Just like a poison swamp.
        # It means the agent has to reach the goal fast!
        # ゴール以外のセルでは移動のたびに報酬が-0.04
        self.default_reward = -0.04

        # Agent can move to a selected direction in move_prob.
        # It means the agent will move different direction
        # in (1 - move_prob).
        # (1 - move_prob)の確率で他の方向へ移動(真逆は確率0)
        self.move_prob = move_prob
        self.reset()

    @property
    def row_length(self):
        return len(self.grid)

    @property
    def column_length(self):
        return len(self.grid[0])

    @property
    def actions(self):
        return [Action.UP, Action.DOWN,
                Action.LEFT, Action.RIGHT]

    @property
    def states(self):
        states = []
        for row in range(self.row_length):
            for column in range(self.column_length):
                # Block cells are not included to the state.
                if self.grid[row][column] != 9:
                    states.append(State(row, column))
        return states

    """
    遷移関数
    """
    def transit_func(self, state, action):
        # 代入する値たち
        # state  → <State: [2, 0]>
        # action → Action.UP
        
        transition_probs = {}
        # こんな感じになる
        # {<State: [1, 0]>: 0.8, <State: [2, 0]>: 0.09999999999999998, <State: [2, 1]>: 0.09999999999999998}
        
        if not self.can_action_at(state):
            # Already on the terminal cell.
            return transition_probs

        # action.value = 指定した通し番号
        opposite_direction = Action(action.value * -1)

        for a in self.actions:
            prob = 0
            # 指定した行動
            if a == action:
                prob = self.move_prob
            # 真逆以外で指定した行動以外
            elif a != opposite_direction:
                prob = (1 - self.move_prob) / 2

            next_state = self._move(state, a)

            if next_state not in transition_probs:
                transition_probs[next_state] = prob
            else:
                transition_probs[next_state] += prob
                
        return transition_probs

    # 現在いる場所が移動を認められたセルか否か
    def can_action_at(self, state):
        if self.grid[state.row][state.column] == 0:
            return True
        else:
            return False

    def _move(self, state, action):
        if not self.can_action_at(state):
            raise Exception("Can't move from here!")

        next_state = state.clone()

        # Execute an action (move).
        if action == Action.UP:
            next_state.row -= 1
        elif action == Action.DOWN:
            next_state.row += 1
        elif action == Action.LEFT:
            next_state.column -= 1
        elif action == Action.RIGHT:
            next_state.column += 1

        # Check whether a state is out of the grid.
        # グリッド内にいるか否か
        # 状態変化せず現状維持
        if not (0 <= next_state.row < self.row_length):
            next_state = state
        if not (0 <= next_state.column < self.column_length):
            next_state = state

        # Check whether the agent bumped a block cell.
        # 移動できないセルに行こうとしているか否か
        # 状態変化せず現状維持
        if self.grid[next_state.row][next_state.column] == 9:
            next_state = state

        return next_state

    """
    報酬関数
    """
    def reward_func(self, state):
        # 即時報酬＆行動終了フラグを返す
        
        # 歩き回ってるだけだと報酬がマイナスに
        reward = self.default_reward
        done = False

        # Check an attribute of next state.
        attribute = self.grid[state.row][state.column]
        if attribute == 1:
            # Get reward! and the game ends.
            reward = 1
            done = True
        elif attribute == -1:
            # Get damage! and the game ends.
            reward = -1
            done = True

        return reward, done

    """
    その他便利な関数
    """
    def reset(self):
        # Locate the agent at lower left corner.
        self.agent_state = State(self.row_length - 1, 0)
        return self.agent_state

    def step(self, action):
        next_state, reward, done = self.transit(self.agent_state, action)
        if next_state is not None:
            self.agent_state = next_state

        return next_state, reward, done

    def transit(self, state, action):
        transition_probs = self.transit_func(state, action)
        
        # 終了判定(現在位置がゴールか否か)(使わない操作)
        if len(transition_probs) == 0:
            return None, None, True

        next_states = []
        probs = []
        for s in transition_probs:
            next_states.append(s)
            probs.append(transition_probs[s])

        next_state = np.random.choice(next_states, p=probs)
        reward, done = self.reward_func(next_state)
        return next_state, reward, done
