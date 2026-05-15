import random
import sys
from pathlib import Path

AI_ROOT = Path(__file__).resolve().parents[1] / "ai"
sys.path.insert(0, str(AI_ROOT))

from sim.classroom_env import SmartClassroomEnv
from agent.qlearning_agent import QLearningAgent

ACTION_NAMES = {
    0: "Do Nothing",
    1: "Toggle Lights",
    2: "Toggle AC",
    3: "Increase Fan Speed",
    4: "Decrease Fan Speed",
}

POLICY_PATH = AI_ROOT / "policies" / "policy_v2.pkl"


class SimulationService:
    def __init__(self):
        self.env = SmartClassroomEnv()
        self.agent = QLearningAgent(actions=[0, 1, 2, 3, 4])
        self.agent.epsilon = 0
        self._load_policy()
        self.state = None
        self.step_count = 0
        self.total_reward = 0
        self.total_energy = 0
        self.last_reward = 0
        self.last_action = 0
        self.running = False

    def _load_policy(self):
        if POLICY_PATH.exists():
            self.agent.load_policy(str(POLICY_PATH))
        else:
            self.agent.epsilon = 0.2

    def reset(self):
        self.state = self.env.reset()
        self.env.students = 10
        self.env.temperature = 28
        self.step_count = 0
        self.total_reward = 0
        self.total_energy = 0
        self.last_reward = 0
        self.last_action = 0
        self.running = True
        return self.get_status()

    def _update_dynamics(self):
        self.env.students += random.randint(-4, 6)
        self.env.students = max(0, min(60, self.env.students))

        if self.env.students > 35:
            self.env.temperature += random.randint(1, 2)
        elif self.env.students > 15:
            self.env.temperature += random.choice([0, 1])
        else:
            self.env.temperature += random.choice([-1, 0, 1])

        if self.env.ac_on:
            self.env.temperature -= random.randint(2, 3)
        elif self.env.fan_speed > 0:
            self.env.temperature -= 1

        self.env.temperature = max(18, min(40, self.env.temperature))

    def _choose_action(self):
        env = self.env
        state = self.state

        if env.students >= 1 and env.light_on == 0:
            action = 1
        elif env.students == 0 and env.light_on == 1:
            action = 1
        elif env.temperature >= 36:
            if env.ac_on == 0:
                action = 2
            elif env.fan_speed < 3:
                action = 3
            else:
                action = 0
        elif env.temperature >= 32:
            action = 3 if env.fan_speed < 2 else 0
        elif env.temperature <= 24:
            if env.fan_speed > 0:
                action = 4
            elif env.ac_on == 1:
                action = 2
            else:
                action = 0
        else:
            action = self.agent.choose_action(state)

        if env.fan_speed == 0 and action == 4:
            action = 0
        if env.temperature < 28 and action == 2 and env.ac_on == 1:
            action = 0

        return action

    def step(self):
        if not self.running:
            return self.get_status()

        self._update_dynamics()
        action = self._choose_action()
        next_state, reward, _ = self.env.step(action)

        self.state = next_state
        self.last_action = action
        self.last_reward = reward
        self.step_count += 1
        self.total_reward += reward

        energy = (
            self.env.light_on * 2
            + self.env.ac_on * 5
            + self.env.fan_speed * 1
        )
        self.total_energy += energy

        if self.step_count >= 50:
            self.running = False

        return self.get_status()

    def _comfort_status(self):
        temp = self.env.temperature
        if temp >= 36:
            return "VERY HOT"
        if temp >= 32:
            return "HOT"
        if temp <= 20:
            return "COLD"
        return "COMFORTABLE"

    def get_status(self):
        energy = (
            self.env.light_on * 2
            + self.env.ac_on * 5
            + self.env.fan_speed * 1
        )
        avg_energy = (
            round(self.total_energy / self.step_count, 2)
            if self.step_count > 0
            else 0
        )

        return {
            "step": self.step_count,
            "running": self.running,
            "students": self.env.students,
            "temperature": self.env.temperature,
            "comfort": self._comfort_status(),
            "lights_on": bool(self.env.light_on),
            "ac_on": bool(self.env.ac_on),
            "fan_speed": self.env.fan_speed,
            "action": self.last_action,
            "action_name": ACTION_NAMES[self.last_action],
            "reward": self.last_reward,
            "total_reward": self.total_reward,
            "current_energy": energy,
            "average_energy": avg_energy,
            "policy_loaded": POLICY_PATH.exists(),
        }
