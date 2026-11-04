"""Factories connecting configured estimators to noise builders and energy wrappers."""

import noises_lib
from utils import (
    DummyEBM,
    EBMFromMultRatio,
    EBMFromNCEEnergy,
    EBMFromSTNCEEnergy,
    EBMFromTRERatio,
    SimpleEBM,
)


def build_estimator_fns(
    config,
    prob_path,
    dim,
    data_shape,
    specified_log_const,
    force_fm_reg_estimator_none=False,
):
    """Build the shared noise and EBM factories used by run libraries."""
    estimator = config.estim.name
    reg_estimator = config.estim.reg_estimator

    main_build_noise_fn, get_ebm = _build_main_noise_and_ebm(
        config=config,
        prob_path=prob_path,
        dim=dim,
        data_shape=data_shape,
        specified_log_const=specified_log_const,
    )

    if (estimator == "fm") and force_fm_reg_estimator_none:
        reg_estimator = "none"

    reg_build_noise_fn = _build_reg_noise_fn(
        config=config,
        prob_path=prob_path,
        reg_estimator=reg_estimator,
    )

    build_noise_fn = _build_combined_noise_fn(
        config=config,
        prob_path=prob_path,
        estimator=estimator,
        reg_estimator=reg_estimator,
        main_build_noise_fn=main_build_noise_fn,
        reg_build_noise_fn=reg_build_noise_fn,
    )

    return (
        main_build_noise_fn,
        reg_build_noise_fn,
        build_noise_fn,
        get_ebm,
        reg_estimator,
    )


def _build_main_noise_and_ebm(
    config,
    prob_path,
    dim,
    data_shape,
    specified_log_const,
):
    estimator = config.estim.name
    reg_estimator = config.estim.reg_estimator
    num_noise_classes = config.estim.num_noise_classes

    if estimator == "nce":
        main_build_noise_fn = noises_lib.get_nce_noise(config=config)

        def get_ebm(model, **unused_kwargs):
            return SimpleEBM(model=model)

    elif estimator == "cnce":
        main_build_noise_fn = noises_lib.get_cnce_noise(config=config)

        def get_ebm(model, **unused_kwargs):
            return SimpleEBM(model=model)

    elif estimator == "stnce":
        main_build_noise_fn = noises_lib.get_stnce_noise(config=config)

        def get_ebm(model, **unused_kwargs):
            return SimpleEBM(model=model)

    elif estimator == "tnce":
        main_build_noise_fn = noises_lib.get_telescoping_noise(
            config=config, prob_path=prob_path
        )
        get_ebm = _build_tnce_ebm_factory(
            config=config,
            dim=dim,
            data_shape=data_shape,
            specified_log_const=specified_log_const,
            reg_estimator=reg_estimator,
        )

    elif estimator == "tcnce":
        main_build_noise_fn = noises_lib.get_telescoping_c_noise(
            config=config, prob_path=prob_path
        )

        def get_ebm(model, **unused_kwargs):
            return DummyEBM(
                model=model,
                num_noise_classes=config.estim.num_noise_classes,
                eps1=config.estim.eps1,
                eps2=config.estim.eps2,
            )

    elif estimator == "tstnce":
        main_build_noise_fn = noises_lib.get_telescoping_st_noise(
            config=config, prob_path=prob_path
        )

        def get_ebm(model, is_dataloader=None, accelerator=None, **unused_kwargs):
            return EBMFromSTNCEEnergy(
                model=model,
                num_noise_classes=num_noise_classes,
                dim=dim,
                data_shape=data_shape,
                log_const=specified_log_const,
                is_dataloader=is_dataloader,
                accelerator=accelerator,
                eps1=config.estim.eps1,
                eps2=config.estim.eps2,
            )

    elif estimator == "dsm":
        main_build_noise_fn = noises_lib.get_dsm_noise(
            config=config, prob_path=prob_path
        )

        def get_ebm(model, **unused_kwargs):
            return DummyEBM(
                model=model,
                num_noise_classes=num_noise_classes,
                eps1=config.estim.eps1,
                eps2=config.estim.eps2,
            )

    elif estimator == "fm":
        main_build_noise_fn = noises_lib.get_fm_noise(
            config=config, prob_path=prob_path
        )

        def get_ebm(model, **unused_kwargs):
            return DummyEBM(
                model=model,
                num_noise_classes=num_noise_classes,
                eps=config.estim.eps1,
                eps2=config.estim.eps2,
            )

    elif estimator == "ctsm":
        main_build_noise_fn = noises_lib.get_ctsm_noise(
            config=config,
            prob_path=prob_path,
        )
        get_ebm = _build_energy_ebm_factory(
            config=config,
            dim=dim,
            data_shape=data_shape,
            specified_log_const=specified_log_const,
            reg_estimator=reg_estimator,
        )

    else:
        raise NotImplementedError

    return main_build_noise_fn, get_ebm


def _build_tnce_ebm_factory(
    config,
    dim,
    data_shape,
    specified_log_const,
    reg_estimator,
):
    if config.estim.param == "ratio":
        if config.estim.num_cla_classes == 2:

            def get_ebm(model, **unused_kwargs):
                return EBMFromTRERatio(
                    model=model,
                    dim=dim,
                    num_noise_classes=config.estim.num_noise_classes,
                )

        else:

            def get_ebm(model, **unused_kwargs):
                return EBMFromMultRatio(
                    model=model,
                    dim=dim,
                    num_noise_classes=config.estim.num_noise_classes,
                )

        return get_ebm

    if config.estim.param == "energy":
        return _build_energy_ebm_factory(
            config=config,
            dim=dim,
            data_shape=data_shape,
            specified_log_const=specified_log_const,
            reg_estimator=reg_estimator,
        )

    raise NotImplementedError


def _build_energy_ebm_factory(
    config, dim, data_shape, specified_log_const, reg_estimator
):
    if reg_estimator == "none":

        def get_ebm(model, **unused_kwargs):
            return EBMFromNCEEnergy(
                model=model,
                num_noise_classes=config.estim.num_noise_classes,
                dim=dim,
                data_shape=data_shape,
                eps1=config.estim.eps1,
                eps2=config.estim.eps2,
            )

    else:

        def get_ebm(model, is_dataloader=None, accelerator=None, **unused_kwargs):
            return EBMFromSTNCEEnergy(
                model=model,
                num_noise_classes=config.estim.num_noise_classes,
                dim=dim,
                data_shape=data_shape,
                log_const=specified_log_const,
                is_dataloader=is_dataloader,
                accelerator=accelerator,
                eps1=config.estim.eps1,
                eps2=config.estim.eps2,
            )

    return get_ebm


def _build_reg_noise_fn(config, prob_path, reg_estimator):
    if reg_estimator == "tcnce":
        return noises_lib.get_telescoping_c_noise(config=config, prob_path=prob_path)
    if reg_estimator == "dsm":
        return noises_lib.get_dsm_noise(
            config=config,
            prob_path=prob_path,
        )
    if reg_estimator == "fm":
        return noises_lib.get_fm_noise(
            config=config,
            prob_path=prob_path,
        )
    return None


def _build_combined_noise_fn(
    config,
    prob_path,
    estimator,
    reg_estimator,
    main_build_noise_fn,
    reg_build_noise_fn,
):
    mixture_loss = getattr(config.estim, "mixture_loss", False)
    if mixture_loss:
        if (estimator != "tnce") or (reg_estimator != "tcnce"):
            raise ValueError(
                "mixture_loss=True is only supported for "
                "config.estim.name='tnce' and config.estim.reg_estimator='tcnce'."
            )
        if config.estim.combine_type not in ["default", "reuse"]:
            raise ValueError(
                "mixture_loss=True requires config.estim.combine_type to be "
                "'default' or 'reuse'."
            )

    if (
        (config.estim.combine_type == "reuse")
        and (reg_estimator == "none")
        and (estimator in ["tnce", "tcnce", "tstnce"])
    ):
        return noises_lib.get_reuse_main_noise(
            config=config,
            prob_path=prob_path,
            estimator=estimator,
        )

    if reg_build_noise_fn is None:

        def build_noise_fn(original_data, train=False, model_energy_fn=None):
            return main_build_noise_fn(
                original_data=original_data,
                train=train,
                model_energy_fn=model_energy_fn,
            )

        return build_noise_fn

    if (config.estim.combine_type == "reuse") and (not mixture_loss):
        assert estimator in ["tnce", "tstnce"]
        assert reg_estimator in ["dsm", "fm"]
        if estimator == "tstnce":
            assert config.estim.stnce_noise_type in [
                "scale",
                "diff",
                "recovery-diff",
            ]

    return noises_lib.get_combine_noise(
        config=config,
        prob_path=prob_path,
        main_estimator=estimator,
        reg_estimator=reg_estimator,
        main_build_noise_fn=main_build_noise_fn,
        reg_build_noise_fn=reg_build_noise_fn,
    )
