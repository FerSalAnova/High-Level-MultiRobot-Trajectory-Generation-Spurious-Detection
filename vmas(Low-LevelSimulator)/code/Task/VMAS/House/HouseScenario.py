#  Copyright (c) 2022-2024.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.

import torch, os
import sys
from enum import Enum
from vmas import render_interactively
from vmas.simulator.core import Agent, Box, Landmark, Line, Sphere, World
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.utils import Color, ScenarioUtils

class CustomColor(Enum):
    RED = (0.75, 0.25, 0.25)
    GREEN = (0.25, 0.75, 0.25)
    BLUE = (0.25, 0.25, 0.75)
    LIGHT_GREEN = (0.45, 0.95, 0.45)
    WHITE = (0.75, 0.75, 0.75)
    GRAY = (0.25, 0.25, 0.25)
    BLACK = (0.15, 0.15, 0.15)
    ORANGE = (1.00, 0.50, 0)
    PINK = (0.97, 0.51, 0.75)
    PURPLE = (0.60, 0.31, 0.64)
    YELLOW = (0.87, 0.87, 0)
    LIGHT_RED = (1.00, 0.70, 0.70)
    LIGHT_BLUE = (0.60, 0.80, 1.00)
    LIGHT_YELLOW = (1.00, 1.00, 0.70)
    LIGHT_PURPLE = (0.85, 0.70, 1.00)
    LIGHT_ORANGE = (1.00, 0.85, 0.60)
    LIGHT_GRAY = (0.65, 0.65, 0.65)
    LIGHT_PINK = (1.00, 0.80, 0.90)
    LIGHT_BROWN = (0.85, 0.70, 0.55)
    MINT_GREEN = (0.70, 1.00, 0.85)
    PALE_GREEN = (0.75, 1.00, 0.75)
    PALE_BLUE = (0.75, 0.95, 1.00)
    CREAM = (1.00, 1.00, 0.90)

def get_color_from_string(color_str: str):
    try:
        return CustomColor[color_str.upper()].value
    except KeyError:
        raise ValueError(f"Unknown color name: {color_str}")

class Scenario(BaseScenario):
    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        # use two T-shaped corridors instead of original passages
        self.shared_reward = kwargs.pop("shared_reward", False)
        ScenarioUtils.check_kwargs_consumed(kwargs)

        self.passage_width = 0.2
        self.passage_length = 0.103

        self.shaping_factor = 100
        # agent setup unchanged

        self.n_agents = 4
        self.agent_radius = 0.03333
        self.agent_spacing = 0.1


        self.map_path = os.path.dirname(os.path.abspath(__file__)) + "/map.txt"
        self.wall_centers = []
        self.zone_centers = []
        self.map_txt = ""

        # world with semidim=1
        world = World(batch_dim, device, x_semidim=2, y_semidim=2)

        # add agents and goals as before
        for i in range(self.n_agents):
            agent = Agent(
                name=f"agent_{i}", shape=Sphere(self.agent_radius), u_multiplier=0.7, color=CustomColor.BLACK.value
            )
            world.add_agent(agent)

        # clear default passages
        self.create_mall_obstacles(world)

        return world  
      
    def add_wall(self, world, corner_1, corner_2, color=CustomColor.BLACK.value):
        self.map_txt = self.map_txt + str(corner_1) + ";" + str(corner_2)+"\n"
        p1 = torch.tensor(corner_1, device=world.device).flip(0)
        p2 = torch.tensor(corner_2, device=world.device).flip(0)
        center = ((p1+p2) / 2).flip(0)
        w = abs(p1[0] - p2[0])
        h = abs(p1[1] - p2[1])
        wall = Landmark(
            name=f"wall_{center}",
            collide=True,
            movable=False,
            shape=Box(length=h, width=w),
            color=color,
        )
        world.add_landmark(wall)

        return center
    
    def place_wall(self, center, env_index=None):
        for wall in self.world.landmarks:
            if wall.name == f"wall_{center}":
                if not env_index is None:
                    wall.set_pos(center.clone().detach(), batch_index=env_index)
                else:
                    for i in range(self.world.batch_dim):
                        wall.set_pos(center.clone().detach(), batch_index=i)
                return

    # def create_mall_obstacles(self, world):
    #     with open(self.map_path, "r") as f:
    #         for line in f:
    #             if line.strip():
    #                 aux = line.strip().split(",")
    #                 corner_1 = tuple(map(float, aux[:2]))
    #                 corner_2 = tuple(map(float, aux[2:]))
    #                 self.wall_centers.append(self.add_wall(world, corner_1, corner_2))

    def add_zone(self, world, corner_1, corner_2, color=CustomColor.GRAY.value):
        self.map_txt = self.map_txt + f"zone,{corner_1[0]},{corner_1[1]},{corner_2[0]},{corner_2[1]},{color.name.lower()}\n"
        
        p1 = torch.tensor(corner_1, device=world.device).flip(0)
        p2 = torch.tensor(corner_2, device=world.device).flip(0)
        center = ((p1 + p2) / 2).flip(0)
        w = abs(p1[0] - p2[0])
        h = abs(p1[1] - p2[1])

        zone = Landmark(
            name=f"zone_{center}",
            collide=False,
            movable=False,
            shape=Box(length=h, width=w),
            color=color.value,
        )
        world.add_landmark(zone)
        return center

    def place_zone(self, center, env_index=None):
        for landmark in self.world.landmarks:
            if landmark.name == f"zone_{center}":
                if env_index is not None:
                    landmark.set_pos(center.clone().detach(), batch_index=env_index)
                else:
                    for i in range(self.world.batch_dim):
                        landmark.set_pos(center.clone().detach(), batch_index=i)
                return


    def create_mall_obstacles(self, world):
        with open(self.map_path, "r") as f:
            for line in f:
                if line.strip():
                    parts = line.strip().split(",")
                    if parts[0] == "wall":
                        corner_1 = tuple(map(float, parts[1:3]))
                        corner_2 = tuple(map(float, parts[3:5]))
                        self.wall_centers.append(self.add_wall(world, corner_1, corner_2))
                    elif parts[0] == "zone":
                        corner_1 = tuple(map(float, parts[1:3]))
                        corner_2 = tuple(map(float, parts[3:5]))
                        color_str = parts[5].strip().upper()
                        try:
                            color = getattr(CustomColor, color_str)
                        except AttributeError:
                            print(f"[WARNING] Unknown color '{color_str}', defaulting to GRAY.")
                            color = CustomColor.GRAY
                        self.zone_centers.append(self.add_zone(world, corner_1, corner_2, color))


    def place_mall_obstacles(self, env_index = None):
        for center in self.wall_centers:
            self.place_wall(center, env_index)

    def place_mall_zones(self, env_index=None):
        for center in self.zone_centers:
            self.place_zone(center, env_index)

    def reset_world_at(self, env_index: int = None):
        self.place_mall_obstacles(env_index)
        self.place_mall_zones(env_index)
        central_agent_pos = torch.tensor(
            [
                [-1.2, -1.375 + (3 * self.agent_radius + self.agent_spacing)]
            ],
            device=self.world.device,
            dtype=torch.float32,
        )
        agents = self.world.agents
        for i, agent in enumerate(agents):
            if i == self.n_agents - 1:
                agent.set_pos(
                    central_agent_pos,
                    batch_index=env_index,
                )
            else:
                agent.set_pos(
                    central_agent_pos
                    + torch.tensor(
                        [
                            [
                                (
                                    0.0
                                    if i % 2
                                    else (
                                        self.agent_spacing
                                        if i == 0
                                        else -self.agent_spacing
                                    )
                                ),
                                (
                                    0.0
                                    if not i % 2
                                    else (
                                        self.agent_spacing
                                        if i == 1
                                        else -self.agent_spacing
                                    )
                                ),
                            ],
                        ],
                        device=self.world.device,
                    ),
                    batch_index=env_index,
                )


    def reward(self, agent: Agent):
        self.rew = torch.zeros(
            self.world.batch_dim,
            device=self.world.device,
            dtype=torch.float32,
        )

        if agent.collide:
            for a in self.world.agents:
                if a != agent:
                    self.rew[self.world.is_overlapping(a, agent)] -= 10
            for landmark in self.world.landmarks[self.n_agents :]:
                if landmark.collide:
                    self.rew[self.world.is_overlapping(agent, landmark)] -= 10

        return self.rew

    def observation(self, agent: Agent):
        # get positions of all entities in this agent's reference frame
        return torch.cat(
            [
                agent.state.pos,
                agent.state.vel,
            ],
            dim=-1,
        )

    def done(self):
        return torch.zeros(self.world.batch_dim, dtype=torch.bool, device=self.world.device)


if __name__ == "__main__":
    render_interactively(
        __file__, control_two_agents=True, n_passages=1, shared_reward=False
    )
