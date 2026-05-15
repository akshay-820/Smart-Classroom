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

        self.agent = QLearningAgent(
            actions=[0, 1, 2, 3, 4]
        )

        self.agent.epsilon = 0

        self._load_policy()

        self.state = None

        self.step_count = 0

        self.total_reward = 0

        self.total_energy = 0

        self.last_reward = 0

        self.last_action = 0

        self.running = False

    # -----------------------------
    # LOAD POLICY
    # -----------------------------
    def _load_policy(self):

        if POLICY_PATH.exists():

            self.agent.load_policy(str(POLICY_PATH))

        else:
            self.agent.epsilon = 0.2

    # -----------------------------
    # RESET SIMULATION
    # -----------------------------
    def reset(self):

        self.state = self.env.reset()

        self.env.students = random.randint(5, 45)

        self.env.temperature = random.randint(24, 32)

        self.step_count = 0

        self.total_reward = 0

        self.total_energy = 0

        self.last_reward = 0

        self.last_action = 0

        self.running = True

        return self.get_status()

    # -----------------------------
    # DYNAMIC ENVIRONMENT
    # -----------------------------
    def _update_dynamics(self):

        # Student movement
        student_change = random.randint(-1, 3)

        self.env.students += student_change

        self.env.students = max(
            0,
            min(60, self.env.students)
        )

        # Natural room heating
        occupancy_heat = self.env.students * 0.02

        ambient_heat = random.uniform(0.2, 0.5)

        self.env.temperature += (
            occupancy_heat + ambient_heat
        )

        # Fan cooling
        if self.env.fan_speed > 0:

            self.env.temperature -= (
                self.env.fan_speed * 0.2
            )

        # AC cooling
        if self.env.ac_on:

            self.env.temperature -= (
                1.0 + (self.env.fan_speed * 0.25)
            )

        # Clamp realistic range
        self.env.temperature = round(
            max(20, min(40, self.env.temperature)),
            1
        )

    # -----------------------------
    # SMART CONTROLLER
    # -----------------------------
    def _choose_action(self):

        env = self.env

        state = self.state

        # -----------------------------
        # LIGHT CONTROL
        # -----------------------------
        if env.students > 0 and env.light_on == 0:
            return 1

        if env.students == 0 and env.light_on == 1:
            return 1

        # -----------------------------
        # VERY HOT
        # -----------------------------
        if env.temperature >= 36:

            # Turn ON AC
            if env.ac_on == 0:
                return 2

            # Increase fan speed
            if env.fan_speed < 3:
                return 3

        # -----------------------------
        # HOT
        # -----------------------------
        elif env.temperature >= 32:

            # Increase fan gradually
            if env.fan_speed < 2:
                return 3

            # AC only at higher heat
            if env.temperature >= 34 and env.ac_on == 0:
                return 2

        # -----------------------------
        # WARM
        # -----------------------------
        elif env.temperature >= 29:

            if env.fan_speed < 1:
                return 3

        # -----------------------------
        # COMFORTABLE
        # -----------------------------
        elif 24 <= env.temperature <= 28:

            # Turn OFF AC
            if env.ac_on == 1:
                return 2

            # Reduce fan slowly
            if env.fan_speed > 1:
                return 4

        # -----------------------------
        # COOL
        # -----------------------------
        elif env.temperature <= 23:

            if env.ac_on == 1:
                return 2

            if env.fan_speed > 0:
                return 4

        # -----------------------------
        # RL POLICY
        # -----------------------------
        action = self.agent.choose_action(state)

        # -----------------------------
        # SAFETY FILTERS
        # -----------------------------

        # Prevent useless AC usage
        if env.temperature < 32 and action == 2 and env.ac_on == 0:
            action = 0

        # Prevent overcooling
        if env.temperature <= 24 and action == 2:
            action = 0

        # Prevent invalid fan decrease
        if env.fan_speed == 0 and action == 4:
            action = 0

        return action

    # -----------------------------
    # STEP
    # -----------------------------
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

        # Energy usage
        energy = (
            self.env.light_on * 2
            + self.env.ac_on * 5
            + self.env.fan_speed * 1
        )

        self.total_energy += energy

        if self.step_count >= 50:
            self.running = False

        return self.get_status()

    # -----------------------------
    # COMFORT STATUS
    # -----------------------------
    def _comfort_status(self):

        temp = self.env.temperature

        if temp >= 37:
            return "VERY HOT"

        if temp >= 32:
            return "HOT"

        if temp <= 22:
            return "COLD"

        return "COMFORTABLE"

    # -----------------------------
    # STATUS
    # -----------------------------
    def get_status(self):

        energy = (
            self.env.light_on * 2
            + self.env.ac_on * 5
            + self.env.fan_speed * 1
        )

        avg_energy = (
            round(
                self.total_energy / self.step_count,
                2
            )
            if self.step_count > 0
            else 0
        )

        return {
            "step": self.step_count,
            "running": self.running,
            "students": self.env.students,
            "temperature": round(self.env.temperature, 1),
            "comfort": self._comfort_status(),
            "lights_on": bool(self.env.light_on),
            "ac_on": bool(self.env.ac_on),
            "fan_speed": self.env.fan_speed,
            "action": self.last_action,
            "action_name": ACTION_NAMES[self.last_action],
            "reward": round(self.last_reward, 2),
            "total_reward": round(self.total_reward, 2),
            "current_energy": energy,
            "average_energy": avg_energy,
            "policy_loaded": POLICY_PATH.exists(),
        }