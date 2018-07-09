import random
import sc2
from sc2 import run_game, maps, Race, Difficulty
from sc2.player import Bot, Computer
from sc2.constants import UnitTypeId as uti
from examples.protoss.cannon_rush import CannonRushBot

class StarBotV0(sc2.BotAI):

    ITERATIONS_PER_MINUTE = 165
    MAX_WORKERS = 65

    async def on_step(self, iteration):
        self.iteration = iteration
        await self.distribute_workers()
        await self.build_workers()
        await self.build_pylons()
        await self.build_assimilators()
        await self.expand()
        await self.offensive_force_buildings()
        await self.build_offensive_force()
        await self.attack()

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

    # how will this work out with all sites or all nexuses or are they individual to each site? (thats why its only creating one)
    async def offensive_force_buildings(self):
        if self.units(uti.PYLON).ready.exists:
            pylon = self.units(uti.PYLON).ready.random
            if self.units(uti.GATEWAY).ready.exists and not self.units(uti.CYBERNETICSCORE):
                # checks if already a cyberneticscore exists should check if near a nexus(should check and build near each nexus and pylon)
                if self.can_afford(uti.CYBERNETICSCORE) and not self.already_pending(uti.CYBERNETICSCORE):
                    await self.build(uti.CYBERNETICSCORE, near=pylon)

            elif self.units(uti.GATEWAY).amount < (self.iteration / self.ITERATIONS_PER_MINUTE / 2):
                if self.can_afford(uti.GATEWAY) and not self.already_pending(uti.GATEWAY):
                    await self.build(uti.GATEWAY, near=pylon)

            if self.units(uti.CYBERNETICSCORE).ready.exists:
                if self.units(uti.STARGATE).amount < (self.iteration / self.ITERATIONS_PER_MINUTE / 2):
                    if self.can_afford(uti.STARGATE) and not self.already_pending(uti.STARGATE):
                        await self.build(uti.STARGATE, near=pylon)

    async def build_offensive_force(self):
        for gw in self.units(uti.GATEWAY).ready.noqueue:
            if not self.units(uti.STALKER).amount > self.units(uti.VOIDRAY).amount:
                if self.can_afford(uti.STALKER) and self.supply_left > 0:
                    await self.do(gw.train(uti.STALKER))

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
            uti.STALKER: [15, 3],
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
        [Bot(Race.Protoss, StarBotV0()), Bot(Race.Protoss, CannonRushBot())],
        realtime=False
    )
