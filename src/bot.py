import sc2
from sc2 import run_game, maps, Race, Difficulty
from sc2.player import Bot, Computer
from sc2.constants import UnitTypeId as uti

class StarBot(sc2.BotAI):
    async def on_step(self, iteration):
        await self.distribute_workers()
        await self.build_workers()
        await self.build_pylons()
        await self.build_assimilators()
        await self.expand()

    async def build_workers(self):
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


run_game(maps.get("AcidPlantLE"), [Bot(Race.Protoss, StarBot()), Computer(Race.Terran, Difficulty.Easy)], realtime=False)
