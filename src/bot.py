import random
import time
import os
import sc2
from sc2 import run_game, maps, Race, Difficulty, position, Result
from sc2.player import Bot, Computer
from sc2.constants import UnitTypeId as uti
import cv2
import numpy as np

# os.environ['SC2PATH'] = '/Applications/StarCraft\ II/'

HEADLESS = False

class StarBot(sc2.BotAI):


    def __init__(self):
        self.ITERATIONS_PER_MINUTE = 165
        self.MAX_WORKERS = 65
        self.do_something_after = 0
        self.train_data = []
        self.start_time = 0

    # changed the source code in main.py and bot_ai.py in sc2 to call on_end after game ends
    # files inside venv
    def on_end(self, game_result):
        print('--- on_end called ---')
        print(game_result)

        if game_result == Result.Victory:
            print('Saving data!')
            np.save("./../train_data/{}.npy".format(str(int(time.time()))), np.array(self.train_data))

    async def on_step(self, iteration):
        self.iteration = iteration
        print('iteration', self.iteration)
        print('game loop', self.state.game_loop)
        print('game time', self.state.game_loop * 0.725 * (1/16))
        if self.state.game_loop == 0:
            self.start_time = time.time()

        print('seconds', time.time() - self.start_time)

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

        x += ((random.randrange(-30, 30)) / 100) * enemy_start_location[0]
        y += ((random.randrange(-30, 30)) / 100) * enemy_start_location[1]

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

        line_max = 50
        barlength = lambda r: int(line_max*(1 if r > 1.0 else r))

        mineral_ratio = self.minerals / 1500
        vespene_ratio = self.vespene / 1500
        population_ratio = self.supply_left / self.supply_cap
        plausible_supply = self.supply_cap / 200.0
        military_weight = len(self.units(uti.VOIDRAY)) / self.supply_used

        ratios = {
            'fighters/supply_used': [(0, 19), (barlength(military_weight), 19), (250, 250, 200), 3],
            'total_supply/200': [(0, 15), (barlength(plausible_supply), 15), (220, 200, 200), 3],
            'supply_left/total_supply': [(0, 11), (barlength(population_ratio), 11), (150, 150, 150), 3],
            'gas/1500': [(0, 7), (barlength(vespene_ratio), 7), (210, 200, 0), 3],
            'minerals/1500': [(0, 3), (barlength(mineral_ratio), 3), (0, 255, 25), 3]
        }

        for r in ratios.values():
            cv2.line(game_data, r[0], r[1], r[2], r[3])

        # flip horizontally to make our final fix in visual representation
        self.flipped = cv2.flip(game_data, 0)
        if not HEADLESS:
            resized = cv2.resize(self.flipped, dsize=None, fx=2, fy=2)
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
        if len(self.units(uti.VOIDRAY).idle) > 0:
            choice = random.randrange(0,4)
            # 1. no attack
            # 2. attack_unit_closest_nexus
            # 3. attack_enemy_structures
            # 5. attack_enemy_start

            # other options it could have?

            # attack with random unit or in groups?

            # other things like enemy units closest to our units
            # later make choices on a per unit bases? (color them white?)

            target = False
            if self.iteration > self.do_something_after:
                if choice == 0:
                    # no attack
                    wait = random.randrange(20, self.ITERATIONS_PER_MINUTE)
                    self.do_something_after = self.iteration + wait
                elif choice == 1:
                    # attack_unit_closest_nexus
                    if len(self.known_enemy_units) > 0:
                        target = self.known_enemy_units.closest_to(random.choice(self.units(uti.NEXUS)))
                elif choice == 2:
                    # attack_enemy_structures
                    if len(self.known_enemy_structures) > 0:
                        target = random.choice(self.known_enemy_structures)
                elif choice == 3:
                    # attack_enemy_start
                    target = self.enemy_start_locations[0]

                if target:
                    for vr in self.units(uti.VOIDRAY).idle:
                        await self.do(vr.attack(target))

                y = np.zeros(4)
                y[choice] = 1
                print(y)
                self.train_data.append([y, self.flipped])
            print(len(self.train_data))


if __name__ == '__main__':
    run_game(
        maps.get("AcidPlantLE"),
        [Bot(Race.Protoss, StarBot()), Computer(Race.Terran, Difficulty.Medium)],
        realtime=False
    )
