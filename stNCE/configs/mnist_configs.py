import ml_collections
import torch


def get_config():
    config = ml_collections.ConfigDict()
    # training
    config.training = training = ml_collections.ConfigDict()
    training.batch_size = 512  # TODO
    training.num_iters = 100000
    training.save_checkpoint = True
    training.save_all_checkpoints = False
    training.log_freq = 1000
    training.setup = False
    training.num_workers = 6
    training.split_batches = True
    training.even_batches = False
    training.grad_norm = 1.0
    training.mixed_precision = "no"

    config.eval = eval = ml_collections.ConfigDict()
    eval.batch_size = 512
    eval.eval_freq = 2000
    eval.selected_metric = "val_mse"
    eval.checkpoint_name = "best_model.pt"
    eval.num_workers = 0
    eval.pin_memory = False
    eval.time_grid = [0.0, 0.25, 0.5, 0.75, 1.0]

    eval.ddplots = False
    eval.sample = True

    # data
    config.data = data = ml_collections.ConfigDict()
    data.name = "mnist"
    data.dim = 784
    data.is_image = True
    data.is_infinite = False
    data.shape = (1, 28, 28)
    data.use_dequantization = True

    # model
    config.model = model = ml_collections.ConfigDict()
    model.name = "precond_unet"
    model.in_channels = 1
    model.model_channels = 32
    model.out_channels = 1
    model.num_res_blocks = 2
    model.attention_resolutions = [7]
    model.dropout = 0.0
    model.channel_mult = [1, 2, 2]
    model.use_checkpoint = False
    model.num_heads = 1
    model.num_head_channels = -1
    model.num_heads_upsample = -1
    model.use_scale_shift_norm = True
    model.resblock_updown = False
    model.use_new_attention_order = True
    model.with_fourier_features = False
    model.input_projection = True

    model.out_dim = "1"

    model.use_ema = True
    model.ema_rate = 0.999

    model.a = 0.75

    model.output_param = "inner_product"
    model.learn_time_log_normalizer = True
    model.include_time_log_normalizer_in_output = True
    model.time_log_normalizer_hidden_dim = None
    model.time_log_normalizer_num_blocks = 2
    model.time_log_normalizer_anchor_at_zero = False
    model.time_log_normalizer_base_gaussian = False
    model.time_log_normalizer_out_bias = True

    model.fm_model_path = "none"

    # added to make pure DSM training work
    model.head_bias = True

    # optimization
    config.optim = optim = ml_collections.ConfigDict()
    optim.lr = 3e-4
    optim.optimizer = "Adam"

    config.scheduler = scheduler = ml_collections.ConfigDict()
    scheduler.name = "flat"  # "flat", "cosine", "warmup_cosine", or "step"
    scheduler.warmup_steps = 0
    scheduler.decay_start_step = 0
    scheduler.min_lr = 0.0
    scheduler.min_lr_ratio = 0.0
    scheduler.step_size = 10000
    scheduler.gamma = 0.1

    # estimation
    config.estim = estim = ml_collections.ConfigDict()
    estim.name = "tnce"
    estim.nce_std = 1.0
    estim.cnce_std = 1e-1
    estim.nce_nu = 1
    estim.cnce_nu = 1
    estim.gmm_num_limits = 1000  # maximum number of gmm components
    estim.gmm_sigma = 0.01  # std of each gmm component

    estim.num_noise_classes = float("inf")
    estim.num_cla_classes = 2
    estim.cnce_noise_type = "white-0.1"
    estim.stnce_noise_type = "diff"

    estim.score_type = "none"  # "none", "oracle", "model", or "self"
    estim.time_score_type = "none"  # "none", "oracle" or "fp"

    estim.reg_estimator = "none"
    estim.reg_main_prop = 0.5
    estim.main_weight = 1.0
    estim.reg_weight = 1.0
    estim.combine_type = "default"
    estim.mixture_loss = False
    estim.rne = False
    estim.rne_reference = False

    estim.time_std = 0.1

    estim.param = "energy"
    estim.prob_path = "ot"
    estim.prob_path_eps = 1e-4
    estim.ei_sde_transition_stats = "condot"  # "original" or "condot"

    estim.eps1 = 1e-4
    estim.eps2 = 1e-4
    estim.t_dist = "uniform"
    estim.resolve_collisions = False

    estim.ctsm_c = 1.0

    config.wandb = wandb = ml_collections.ConfigDict()
    wandb.use_wandb = False
    wandb.project_name = "stnce"
    wandb.run_name = "test"

    config.seed = 1
    config.end_seed = None
    config.device = (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )
    config.print_test = False

    return config
