import ml_collections
import torch


def get_config():
    config = ml_collections.ConfigDict()
    # training
    config.training = training = ml_collections.ConfigDict()
    training.batch_size = 250
    training.num_iters = 50000
    training.setup = False
    training.save_checkpoint = False

    config.eval = eval = ml_collections.ConfigDict()
    eval.batch_size = 500
    eval.eval_freq = 2000
    eval.selected_metric = "val_loss"
    eval.is_num_samples = 100000

    # data
    config.data = data = ml_collections.ConfigDict()
    data.name = "distant_gaussian"
    data.dim = 2
    data.is_image = False
    data.eps = 1e-5
    data.standardized = False
    data.num_train_data = 10000
    data.num_eval_data = 10000
    data.num_components = 10
    data.sigma = 0.5
    data.interp_alpha1 = 0.9
    data.interp_alpha2 = 1.0

    # model
    config.model = model = ml_collections.ConfigDict()
    model.name = "ebm_mlp"
    model.out_dim = "1"

    # only apply to mlp
    model.num_hidden_layers = 3
    model.num_hidden_units = 100
    model.embed_dim = 1

    model.use_ema = False
    model.ema_rate = 0.999

    model.fm_model_path = "none"

    # optimization
    config.optim = optim = ml_collections.ConfigDict()
    optim.lr = 1e-4
    optim.optimizer = "Adam"

    # estimation
    config.estim = estim = ml_collections.ConfigDict()
    estim.name = "tnce"
    estim.nce_std = 1.0
    estim.cnce_std = 1e-1
    estim.nce_nu = 1
    estim.cnce_nu = 1
    estim.gmm_num_limits = 1000  # maximum number of gmm components
    estim.gmm_sigma = 0.01  # std of each gmm component
    estim.param = "energy"  # "ratio" or "energy"

    estim.num_noise_classes = float("inf")
    estim.num_cla_classes = 2
    estim.cnce_noise_type = "white-0.1"
    estim.stnce_noise_type = "white-0.1"

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
    estim.normalize_time = False
    estim.reg_time_eps = 0.0

    estim.log_const_method = "importance"
    estim.prob_path = "ot"
    estim.prob_path_eps = 1e-4
    estim.ei_sde_transition_stats = "condot"  # "original" or "condot"

    estim.eps1 = 1e-4
    estim.eps2 = 1e-4
    estim.t_dist = "uniform"
    estim.resolve_collisions = False

    estim.ctsm_c = 1.0

    config.seed = 1
    config.end_seed = None
    config.device = (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )
    config.print_test = False

    return config
