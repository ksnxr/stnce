import logging
import os


def _configure_rank_miopen_cache():
    miopen_cache_base = os.environ.get("MY_MIOPEN_CACHE")
    if miopen_cache_base is None:
        return

    rank = (
        os.environ.get("RANK")
        or os.environ.get("LOCAL_RANK")
        or os.environ.get("SLURM_PROCID")
        or os.environ.get("SLURM_LOCALID")
        or "single"
    )
    rank_cache = os.path.join(miopen_cache_base, f"rank-{rank}")
    kernel_cache = os.path.join(rank_cache, "kernel_cache")
    user_db = os.path.join(rank_cache, "user_db")

    os.environ["MIOPEN_CUSTOM_CACHE_DIR"] = kernel_cache
    os.environ["MIOPEN_USER_DB_PATH"] = user_db
    os.makedirs(kernel_cache, exist_ok=True)
    os.makedirs(user_db, exist_ok=True)


_configure_rank_miopen_cache()

from absl import app
from absl import flags
from ml_collections.config_flags import config_flags


FLAGS = flags.FLAGS


def _is_main_process():
    process_index = os.environ.get("ACCELERATE_PROCESS_INDEX")
    if process_index is not None:
        return int(process_index) == 0
    local_rank = os.environ.get("LOCAL_RANK")
    if local_rank is not None:
        return int(local_rank) == 0
    return True


config_flags.DEFINE_config_file(
    "config", None, "training configuration", lock_config=True
)
flags.DEFINE_string("workdir", None, "work directory")
flags.DEFINE_enum("mode", None, ["train", "eval"], "running mode: train or eval")
flags.DEFINE_string("eval_folder", "eval", "eval folder")
flags.mark_flags_as_required(["config", "mode"])


def main(argv):

    workdir = FLAGS.workdir
    if workdir is None:
        if FLAGS.config.training.num_iters == 0:
            workdir = (
                f"../logs/data={FLAGS.config.data.name}_seed={FLAGS.config.seed}_sets"
            )
        else:
            estim_configs = ""
            workdir = f"../logs/data={FLAGS.config.data.name}_estimator={FLAGS.config.estim.name}_{estim_configs}lr={FLAGS.config.optim.lr}_seed={FLAGS.config.seed}"
    if FLAGS.mode == "train":
        os.makedirs(workdir, exist_ok=True)
        logger = logging.getLogger()
        if _is_main_process():
            file_stream = open(os.path.join(workdir, "stdout.txt"), "w")
            handler = logging.StreamHandler(file_stream)
            formatter = logging.Formatter(
                "%(levelname)s - %(filename)s - %(asctime)s - %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel("INFO")
        else:
            logger.setLevel("WARNING")
        if FLAGS.config.data.is_image:
            if FLAGS.config.data.name == "imagenet64":
                from imagenet_run_lib import train
            else:
                from image_run_lib import train
        else:
            from run_lib import train

        train(FLAGS.config, workdir)

    elif FLAGS.mode == "eval":
        if FLAGS.config.data.is_image:
            if FLAGS.config.data.name == "imagenet64":
                from imagenet_run_lib import evaluate
            else:
                from image_run_lib import evaluate
        else:
            from run_lib import evaluate

        evaluate(FLAGS.config, workdir, FLAGS.eval_folder)
    else:
        raise NotImplementedError


if __name__ == "__main__":
    app.run(main)
