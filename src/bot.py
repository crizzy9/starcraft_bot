import random
import sc2
from sc2 import run_game, maps, Race, Difficulty, position
from sc2.player import Bot, Computer
from sc2.constants import UnitTypeId as uti
import cv2
import numpy as np


class StarBot(sc2.BotAI):

    ITERATIONS_PER_MINUTE = 165
    MAX_WORKERS = 65

    async def on_step(self, iteration):
        self.iteration = iteration
        await self.scout()
        await self.distribute_workers()
        await self.build_workers()
        await self.build_pylons()
        await self.build_assimilators()
        await self.expand()
        await self.offensive_force_buildings()
        await self.build_offensive_force()
        await self.extract_and_visualize()
        await self.attack()

    def random_location_variance(self, enemy_start_location):
        x = enemy_start_location[0]
        y = enemy_start_location[1]

        x += ((random.randrange(-20, 20)) / 100) * enemy_start_location[0]
        y += ((random.randrange(-20, 20)) / 100) * enemy_start_location[1]

        if x < 0:
            x = 0
        if y < 0:
            y = 0
        if x > self.game_info.map_size[0]:
            x = self.game_info.map_size[0]
        if y > self.game_info.map_size[1]:
            y = self.game_info.map_size[1]

        go_to = position.Point2(position.Pointlike((x, y)))
        return go_to

    async def scout(self):
        if len(self.units(uti.OBSERVER)) > 0:
            scout = self.units(uti.OBSERVER)[0]
            if scout.is_idle:
                enemy_location = self.enemy_start_locations[0]
                move_to = self.random_location_variance(enemy_location)
                print(move_to)
                await self.do(scout.move(move_to))
        else:
            for rf in self.units(uti.ROBOTICSFACILITY).ready.noqueue:
                if self.can_afford(uti.OBSERVER) and self.supply_left > 0:
                    await self.do(rf.train(uti.OBSERVER))

    async def extract_and_visualize(self):
        # to figure out the methods inside
        # print(dir(self))
        # print(self.game_info)
        game_data = np.zeros((self.game_info.map_size[1], self.game_info.map_size[0], 3), np.uint8)

        # UNIT: [SIZE, (BGR/RGB COLOR)]
        draw_dict = {
            uti.NEXUS: [15, (0, 255, 0)],
            uti.PYLON: [3, (20, 235, 0)],
            uti.PROBE: [1, (55, 200, 0)],
            uti.ASSIMILATOR: [2, (55, 200, 0)],
            uti.GATEWAY: [3, (200, 100, 0)],
            uti.CYBERNETICSCORE: [3, (150, 150, 0)],
            uti.STARGATE: [5, (255, 0, 0)],
            uti.VOIDRAY: [3, (255, 100, 0)],
            # uti.OBSERVER: [1, (255, 255, 255)]
        }

        for unit_type, attr in draw_dict.items():
            for unit in self.units(unit_type).ready:
                pos = unit.position
                cv2.circle(game_data, (int(pos[0]), int(pos[1])), attr[0], attr[1], -1)

        main_base_names = ["nexus", "commandcenter", "hatchery"]
        for enemy_building in self.known_enemy_structures:
            pos = enemy_building.position
            if enemy_building.name.lower() in main_base_names:
                cv2.circle(game_data, (int(pos[0]), int(pos[1])), 15, (0, 0, 255), -1)
            else:
                cv2.circle(game_data, (int(pos[0]), int(pos[1])), 5, (200, 50, 212), -1)

        for enemy_unit in self.known_enemy_units:

            if not enemy_unit.is_structure:
                worker_names = ["probe",
                                "scv",
                                "drone"]
                # if that unit is a PROBE, SCV, or DRONE... it's a worker
                pos = enemy_unit.position
                if enemy_unit.name.lower() in worker_names:
                    cv2.circle(game_data, (int(pos[0]), int(pos[1])), 1, (55, 0, 155), -1)
                else:
                    cv2.circle(game_data, (int(pos[0]), int(pos[1])), 3, (50, 0, 215), -1)

        # because want to draw observer last
        for obs in self.units(uti.OBSERVER).ready:
            pos = obs.position
            cv2.circle(game_data, (int(pos[0]), int(pos[1])), 1, (255, 255, 255), -1)

        # flip horizontally to make our final fix in visual representation
        flipped = cv2.flip(game_data, 0)
        resized = cv2.resize(flipped, dsize=None, fx=2, fy=2)
        cv2.imshow('Vision', resized)
        cv2.waitKey(1)

    async def build_workers(self):
        if self.units(uti.NEXUS).amount*16 > self.units(uti.PROBE).amount:
            if self.units(uti.PROBE).amount < self.MAX_WORKERS:
                for nexus in self.units(uti.NEXUS).ready.noqueue:
                    if self.can_afford(uti.PROBE):
                        await self.do(nexus.train(uti.PROBE))

    async def build_pylons(self):
        if self.supply_left < 5 and not self.already_pending(uti.PYLON):
            nexuses = self.units(uti.NEXUS).ready
            if nexuses.exists:
                if self.can_afford(uti.PYLON):
                    await self.build(uti.PYLON, near=nexuses.first)

    async def build_assimilators(self):
        for nexus in self.units(uti.NEXUS).ready:
            vespenes = self.state.vespene_geyser.closer_than(15.0, nexus)
            for vespene in vespenes:
                if not self.can_afford(uti.ASSIMILATOR):
                    break
                worker = self.select_build_worker(vespene.position)
                if worker is None:
                    break
                if not self.units(uti.ASSIMILATOR).closer_than(1.0, vespene).exists:
                    await self.do(worker.build(uti.ASSIMILATOR, vespene))

    async def expand(self):
        if self.units(uti.NEXUS).amount < 3 and self.can_afford(uti.NEXUS):
            await self.expand_now()

    async def offensive_force_buildings(self):
        if self.units(uti.PYLON).ready.exists:
            pylon = self.units(uti.PYLON).ready.random
            if self.units(uti.GATEWAY).ready.exists and not self.units(uti.CYBERNETICSCORE):
                if self.can_afford(uti.CYBERNETICSCORE) and not self.already_pending(uti.CYBERNETICSCORE):
                    await self.build(uti.CYBERNETICSCORE, near=pylon)

            elif self.units(uti.GATEWAY).amount < 1:
                if self.can_afford(uti.GATEWAY) and not self.already_pending(uti.GATEWAY):
                    await self.build(uti.GATEWAY, near=pylon)

            if self.units(uti.CYBERNETICSCORE).ready.exists:
                if self.units(uti.ROBOTICSFACILITY).amount < 1:
                    if self.can_afford(uti.ROBOTICSFACILITY) and not self.already_pending(uti.ROBOTICSFACILITY):
                        await self.build(uti.ROBOTICSFACILITY, near=pylon)
                if self.units(uti.STARGATE).amount < (self.iteration / self.ITERATIONS_PER_MINUTE):
                    if self.can_afford(uti.STARGATE) and not self.already_pending(uti.STARGATE):
                        await self.build(uti.STARGATE, near=pylon)

    async def build_offensive_force(self):
        for sg in self.units(uti.STARGATE).ready.noqueue:
            if self.can_afford(uti.VOIDRAY) and self.supply_left > 0:
                await self.do(sg.train(uti.VOIDRAY))

    def find_target(self, state):
        if len(self.known_enemy_units) > 0:
            return random.choice(self.known_enemy_units)
        elif len(self.known_enemy_structures) > 0:
            return random.choice(self.known_enemy_structures)
        else:
            return self.enemy_start_locations[0]

    async def attack(self):
        # {UNIT: [n to fight, n to defend]}
        aggressive_units = {
            uti.VOIDRAY: [8, 3]
        }

        for unit, count in aggressive_units.items():
            if self.units(unit).amount > count[0]:
                for s in self.units(unit).idle:
                    await self.do(s.attack(self.find_target(self.state)))
            elif self.units(unit).amount > count[1]:
                if len(self.known_enemy_units) > 0:
                    for s in self.units(unit).idle:
                        await self.do(s.attack(random.choice(self.known_enemy_units)))


if __name__ == '__main__':
    run_game(
        maps.get("AcidPlantLE"),
        [Bot(Race.Protoss, StarBot()), Computer(Race.Terran, Difficulty.Hard)],
        realtime=False
    )
