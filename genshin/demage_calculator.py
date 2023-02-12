# encode: utf-8
'''
https://library.keqingmains.com/combat-mechanics/damage/damage-formula
'''


class TransformativeReactionMultiplier:
    Burning = 0.25
    Superconduct = 0.5
    Swirl = 0.6
    ElectroCharged = 1.2
    Shattered = 1.5
    Overloaded = 2
    Bloom = 2
    Burgeon = 3
    Hyperbloom = 3


class AmplifyingReactionMultiplier:
    # 增幅反应
    VaporizeWithHydro = 2
    VaporizeWithPyro = 1.5
    MeltWithPyro = 2
    MeltWithCyro = 1.5


class AmplifyingReaction:
    def __init__(self, multiplier):
        # assert multiplier in AmplifyingReactionMultiplier
        self.multiplier = multiplier

    def damage(self, em):
        res = 1 + (2.78 * em) / (1400 + em)
        res *= self.multiplier
        return res


class TransformativeReactions:
    # 剧变反应
    Burgeon = 3
    Hyperbloom = 3
    Overloaded = 2
    Bloom = 2
    Shattered = 1.5
    # ElectroCharged=1.2*???
    Swirl = 0.6
    SuperConduct = 0.5
    Burning = 0.25

    def __init__(self):
        self.multiplier = self.Hyperbloom
        self.level90 = 1446.85
        self.level80 = 1077.44

    def damage(self, em, level=90):
        res = (16 * em) / (2000 + em) + 1
        res *= self.multiplier * self.level90
        return res


# 怪物抗性
BaseResistance = 0.1
ResistanceReduction = 0.3
Resistance = BaseResistance - ResistanceReduction
if Resistance < 0:
    EnemyResMult = 1 - (Resistance / 2)
elif Resistance < 0.75:
    EnemyResMult = 1 - Resistance
else:
    EnemyResMult = 1 / (1 + 4 * Resistance)


# 伤害结果
tr = TransformativeReactions()
tr.multiplier = TransformativeReactionMultiplier.Hyperbloom
em = 939
print(f'90级 {em}精通，+ 草套，一颗种子伤害：', int(tr.damage(em) * EnemyResMult))

#
vaporizeDamage = AmplifyingReaction(AmplifyingReactionMultiplier.VaporizeWithPyro)
res = vaporizeDamage.damage(400)
print(f'{em}精通，火触发蒸发：', res * EnemyResMult)



import matplotlib.pyplot as plt
import numpy as np

# Data for plotting
em = np.arange(0.0, 1200.0, 1)
Hyperbloom = ((16 * em) / (2000 + em)) + 1
vapWithPyro = vaporizeDamage.damage(em)

fig, ax = plt.subplots()
ax.plot(em, Hyperbloom, c='g')
ax.plot(em, vapWithPyro, c='r')

ax.set(xlabel='em', ylabel='multiplier',
       title='transformative damage multiplier based on em')
ax.grid()

# fig.savefig("test.png")
plt.show()
