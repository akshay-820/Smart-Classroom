import random

class SmartClassroomEnv:

    def __init__(self):

        self.max_students = 50

        self.reset()

    def reset(self):

        self.students = random.randint(0, 50)

        self.temperature = random.randint(20, 40)

        self.light_on = 0
        self.ac_on = 0
        self.fan_speed = 0

        return self.get_state()

    def get_state(self):

        return (
            self.students // 10,
            self.temperature // 5,
            self.light_on,
            self.ac_on,
            self.fan_speed
        )

    def step(self, action):

        """
        Actions:
        0 -> Do nothing
        1 -> Toggle lights
        2 -> Toggle AC
        3 -> Increase fan speed
        4 -> Decrease fan speed
        """

        if action == 1:
            self.light_on = 1 - self.light_on

        elif action == 2:
            self.ac_on = 1 - self.ac_on

        elif action == 3:
            self.fan_speed = min(3, self.fan_speed + 1)

        elif action == 4:
            self.fan_speed = max(0, self.fan_speed - 1)

        # Environment dynamics

        self.temperature += random.choice([-1, 0, 1])

        self.temperature = max(18, min(40, self.temperature))

        # Energy usage

        energy = (
            self.light_on * 2 +
            self.ac_on * 5 +
            self.fan_speed * 1
        )

        # Comfort score

        comfort = 0

        if self.students > 0:

            if self.light_on:
                comfort += 5

            if self.temperature > 30 and self.ac_on:
                comfort += 5

            elif self.temperature > 30 and self.fan_speed >= 2:
                comfort += 3

            elif 22 <= self.temperature <= 28:
                comfort += 4

        # Final reward

        reward = comfort - energy

        next_state = self.get_state()

        done = False

        return next_state, reward, done