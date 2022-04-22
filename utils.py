import os, re
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


def get_latest_model_path(logger):
    latest_model = None
    try:
        run_dir_match = re.compile(r"^%s_\d+\.\d+$" % logger.project)
        model_dir_match = re.compile(r"^%s_\d+$" % logger.project)
        run_dir_names = sorted([d for d in os.listdir("models") if run_dir_match.match(d)],
                               key=natural_keys,
                               reverse=True
                               )
        for run_dir_name in run_dir_names:
            run_dir_path = os.path.join("models", run_dir_name)
            model_dir_names = sorted([d for d in os.listdir(run_dir_path) if model_dir_match.match(d)],
                                     key=natural_keys,
                                     reverse=True
                                     )
            for model_dir_name in model_dir_names:
                model_dir_path = os.path.join(run_dir_path, model_dir_name)
                if "checkpoint.pt" in os.listdir(model_dir_path):
                    latest_model = os.path.join(
                        model_dir_path, "checkpoint.pt")
                    break
            if latest_model:
                break
    except IndexError:
        pass
    return latest_model

def atof(text):
    try:
        retval = float(text)
    except ValueError:
        retval = text
    return retval

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    float regex comes from https://stackoverflow.com/a/12643073/190597
    '''
    return [ atof(c) for c in re.split(r'[+-]?([0-9]+(?:[.][0-9]*)?|[.][0-9]+)', text) ]