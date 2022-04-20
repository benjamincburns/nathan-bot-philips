from typing import List
from rlgym.utils.gamestates import PlayerData

def same_team(*args: List[PlayerData]):
  return len(args) == 0 or all(player.team_num == args[0].team_num for player in args)


def is_teammate(player, PlayerData, other: PlayerData):
  return player.car_id != other.car_id and player.team_num == other.team_num

def same_player(*args: List[PlayerData]):
  return len(args) == 1 or (
    len(args) > 1 and all(player.car_id == args[0].car_id for player in args)
  )
