"""Training objectives consuming the batches constructed by noises_lib."""

import math
import torch
import torch.optim as optim


def get_optimizer(config, params):
    if config.optim.optimizer == "Adam":
        optimizer = optim.Adam(params, lr=config.optim.lr)
    elif config.optim.optimizer == "LBFGS":
        optimizer = optim.LBFGS(params, lr=config.optim.lr)
    else:
        raise NotImplementedError

    return optimizer


def get_nce_loss_fn(nce_nu):
    log_nce_nu = math.log(nce_nu)

    def loss_fn(model_energy_fn, batch):
        data, noise, log_noise_fn = batch

        all_inputs = torch.concat([data, noise], dim=0)

        logits = -model_energy_fn(all_inputs) - log_noise_fn(all_inputs) - log_nce_nu
        logits_d = logits[: data.shape[0], :]
        logits_n = logits[data.shape[0] :, :]

        loss_d = -torch.mean(torch.nn.functional.logsigmoid(logits_d))
        loss_n = -nce_nu * torch.mean(torch.nn.functional.logsigmoid(-logits_n))
        return loss_d + loss_n

    return loss_fn


def get_cnce_loss_fn(cnce_nu):
    def loss_fn(model_energy_fn, batch):
        # cond: log p(conditional data | data) - log p(data | conditional data))
        data, conditional_data, cond = batch

        outputs = -model_energy_fn(torch.concat([data, conditional_data], dim=0))

        num_data = data.shape[0]
        log_model_data = torch.repeat_interleave(
            outputs[:num_data, :], repeats=cnce_nu, dim=0
        )
        log_model_conditional_datas = outputs[num_data:, :]

        loss = -2.0 * torch.mean(
            torch.nn.functional.logsigmoid(
                log_model_data - log_model_conditional_datas + cond
            )
        )
        return loss

    return loss_fn


def get_stnce_loss_fn(nce_nu, cnce_nu):

    log_nce_nu = math.log(nce_nu)

    def loss_fn(model_energy_fn, batch):
        # cond: log p(noise | data) - log p(data | noise)
        (
            data,
            conditional_noise,
            data_cond,
            noise,
            conditional_data,
            noise_cond,
            log_noise_fn,
        ) = batch

        # get all outputs
        num_data = data.shape[0]
        model_outputs = -model_energy_fn(torch.concat([data, conditional_data], dim=0))

        num_conditional_noise = conditional_noise.shape[0]
        noise_outputs = log_noise_fn(torch.concat([conditional_noise, noise], dim=0))

        # data
        d_logits = (
            torch.repeat_interleave(model_outputs[:num_data], repeats=cnce_nu, dim=0)
            - noise_outputs[:num_conditional_noise]
            + data_cond
            - log_nce_nu
        )

        # noise
        n_logits = (
            model_outputs[num_data:]
            - torch.repeat_interleave(
                noise_outputs[num_conditional_noise:], repeats=cnce_nu, dim=0
            )
            + noise_cond
            - log_nce_nu
        )

        loss_d = -torch.mean(torch.nn.functional.logsigmoid(d_logits))
        loss_n = -nce_nu * torch.mean(torch.nn.functional.logsigmoid(-n_logits))
        return loss_d + loss_n

    return loss_fn


def get_tnce_loss_fn(param="ratio", num_noise_classes=1, num_cla_classes=2):
    if param == "energy":
        if num_cla_classes == 2:

            def loss_fn(energy_fn, batch, energies=None):
                # xs: (B, D), times: (B*(K+1)), every first entry of times is gt
                if energies is None:
                    xs, ts, taus = batch
                    all_logits = -energy_fn(
                        xs.repeat_interleave(2, dim=0),
                        torch.stack([ts, taus], dim=1).view(-1),
                    ).view((xs.shape[0], 2))  # (B, K+1)
                else:
                    energy_xs, energy_ys = energies
                    all_logits = -torch.stack(
                        [energy_xs.view(-1), energy_ys.view(-1)], dim=1
                    )  # (B, 2)

                return -2.0 * torch.mean(
                    torch.nn.functional.logsigmoid(all_logits[:, 0] - all_logits[:, 1])
                )

        else:

            def loss_fn(energy_fn, batch):
                # xs: (B, D), times: (B*(K+1)), every first entry of times is gt
                xs, times = batch

                all_logits = -energy_fn(
                    xs.repeat_interleave(num_cla_classes, dim=0), times
                ).view((xs.shape[0], num_cla_classes))  # (B, K+1)

                loss = -num_cla_classes * torch.mean(
                    torch.nn.functional.log_softmax(all_logits, dim=-1)[..., 0]
                )
                return loss

    elif param == "ratio":
        if num_cla_classes == 2:

            def loss_fn(log_ratio_fn, batch):
                xs, ys, times = batch

                all_xs = torch.cat([xs, ys], dim=0)
                all_log_ratios = log_ratio_fn(all_xs)
                log_ratio_x, log_ratio_y = torch.split(
                    all_log_ratios, [xs.shape[0], ys.shape[0]], dim=0
                )

                log_ratio_x = log_ratio_x.gather(1, times.unsqueeze(dim=-1)).squeeze(
                    dim=-1
                )
                log_ratio_y = log_ratio_y.gather(1, times.unsqueeze(dim=-1)).squeeze(
                    dim=-1
                )

                return -torch.mean(
                    torch.nn.functional.logsigmoid(log_ratio_x)
                ) - torch.mean(torch.nn.functional.logsigmoid(-log_ratio_y))

        else:

            def loss_fn(log_ratio_fn, batch):
                # xs: (B, D), times: (B)
                xs, times = batch

                all_logits = log_ratio_fn(xs)  # (B, K)
                loss = torch.nn.functional.cross_entropy(all_logits, times)

                return loss

    return loss_fn


def get_tcnce_loss_fn(param="ratio", num_noise_classes=1, num_cla_classes=2):
    if param == "energy":

        def loss_fn(energy_fn, batch, energies=None):
            if energies is None:
                xs, ts, ys, taus, conds = batch

                all_energies = energy_fn(
                    torch.concat([xs, ys], dim=0), torch.concat([ts, taus], dim=0)
                )
                energy_xs, energy_ys = torch.split(
                    all_energies, [xs.shape[0], ys.shape[0]], dim=0
                )
            else:
                energy_xs, energy_ys = energies
                conds = batch

            conds = conds.view((-1, 1))

            return -2.0 * torch.mean(
                torch.nn.functional.logsigmoid(-energy_xs + energy_ys + conds)
            )

    elif param == "ratio":

        def loss_fn(log_ratio_fn, batch):
            xs, ys, times = batch

            all_xs = torch.cat([xs, ys], dim=0)
            all_log_ratios = log_ratio_fn(all_xs, times).view((-1))
            log_ratio_x, log_ratio_y = torch.split(
                all_log_ratios, [xs.shape[0], ys.shape[0]], dim=0
            )

            return -2.0 * torch.mean(
                torch.nn.functional.logsigmoid(log_ratio_x - log_ratio_y)
            )

    return loss_fn


def get_tstnce_loss_fn(param="energy", rne=False):
    if param == "energy":
        # discrete

        # continuous

        def loss_fn(energy_fn, batch, energies=None):
            if energies is None:
                if len(batch) == 6:
                    xs, ts, ys, taus, rev_xs, conds = batch
                    num_total = xs.shape[0]
                    all_energies = energy_fn(
                        torch.cat([xs, ys, rev_xs], dim=0),
                        torch.cat([ts, taus, ts], dim=0),
                    )
                    energy_xs_ts = all_energies[:num_total]
                    energy_ys_taus = all_energies[num_total : 2 * num_total]
                    energy_rev_xs_ts = all_energies[2 * num_total :]
                    energy_xs = torch.cat([energy_xs_ts, energy_ys_taus], dim=0)
                    energy_ys = torch.cat([energy_ys_taus, energy_rev_xs_ts], dim=0)
                else:
                    xs, ts, ys, taus, conds = batch
                    all_energies = energy_fn(
                        torch.cat([xs, ys], dim=0), torch.cat([ts, taus], dim=0)
                    )
                    energy_xs, energy_ys = torch.split(
                        all_energies, [xs.shape[0], ys.shape[0]], dim=0
                    )
            else:
                energy_xs, energy_ys = energies
                conds = batch[0]

            if rne:
                return torch.mean(torch.square(-energy_xs + energy_ys + conds))
            return -2.0 * torch.mean(
                torch.nn.functional.logsigmoid(-energy_xs + energy_ys + conds)
            )

    return loss_fn


def get_dsm_loss_fn(param="energy"):
    if param == "energy":

        @torch.enable_grad()
        def loss_fn(energy_fn, batch, energies=None):
            if energies is None:
                xs, ts, targets, weights = batch
                xs.requires_grad_(True)
                energy_xs = energy_fn(xs, ts)
            else:
                energy_xs, xs = energies
                targets, weights = batch  # we still need xs for grad

            predictions = torch.autograd.grad(energy_xs.sum(), xs, create_graph=True)[0]

            return torch.mean(torch.square(weights * predictions - targets))

    else:
        raise NotImplementedError

    return loss_fn


def get_ctsm_loss_fn(param="energy"):
    if param == "energy":

        @torch.enable_grad()
        def loss_fn(energy_fn, batch, energies=None):
            if energies is None:
                xs, ts, targets, weights = batch
                ts.requires_grad_(True)
                energy_xs = energy_fn(xs, ts)
            else:
                energy_xs, ts = energies
                targets, weights = batch  # we still need ts for grad

            predictions = torch.autograd.grad(energy_xs.sum(), ts, create_graph=True)[0]

            return torch.mean(torch.square(weights * predictions - targets))

    else:
        raise NotImplementedError

    return loss_fn


def get_fm_loss_fn(prob_path, param="energy"):

    if param == "energy":

        @torch.enable_grad()
        def loss_fn(energy_fn, batch, energies=None):
            if energies is None:
                xs, ts, vs = batch
                xs.requires_grad_(True)
                energy_xs = energy_fn(xs, ts)
            else:
                energy_xs, xs = energies
                ts, vs = batch  # we still need xs for grad

            model_scores = torch.autograd.grad(energy_xs.sum(), xs, create_graph=True)[
                0
            ]

            model_vs = prob_path.v_from_score(-model_scores, xs, ts)

            loss = torch.mean(torch.square(model_vs - vs))
            return loss

    elif param == "velocity":

        def loss_fn(model, batch):
            xs, ts, vs = batch

            loss = torch.mean(torch.square(model(xs, ts) - vs))
            return loss

    return loss_fn


def get_combined_loss_fn(
    loss1_fn,
    loss2_fn,
    main_weight=1.0,
    reg_weight=1.0,
    estimator=None,
    reg_estimator=None,
    combine_type=None,
    mixture_loss=False,
):
    if mixture_loss:
        if (estimator != "tnce") or (reg_estimator != "tcnce"):
            raise ValueError(
                "mixture_loss=True is only supported for "
                "estimator='tnce' with reg_estimator='tcnce'."
            )
        if combine_type not in ["default", "reuse"]:
            raise ValueError(
                "mixture_loss=True requires combine_type to be 'default' or 'reuse'."
            )

        def loss_fn(model_energy_fn, batches):
            batch1, batch2, main_prop = batches
            if batch1 is None and batch2 is None:
                raise ValueError("mixture_loss received no TNCE or TCNCE batch.")
            if batch1 is None:
                return loss2_fn(model_energy_fn, batch2)
            if batch2 is None:
                return loss1_fn(model_energy_fn, batch1)

            loss1 = loss1_fn(model_energy_fn, batch1)
            loss2 = loss2_fn(model_energy_fn, batch2)
            return main_prop * loss1 + (1.0 - main_prop) * loss2

        return loss_fn

    if combine_type == "separate":
        # no dual sm here

        def loss_fn(model_energy_fn, batches):
            batch1, batch2, reg_main_prop = batches
            loss1 = loss1_fn(model_energy_fn, batch1)
            loss2 = loss2_fn(model_energy_fn, batch2)
            return (
                main_weight * reg_main_prop * loss1
                + reg_weight * (1.0 - reg_main_prop) * loss2
            )

    else:
        if reg_estimator in ["dsm", "fm"]:
            if estimator != "ctsm":
                if combine_type == "reuse":

                    @torch.enable_grad()
                    def loss_fn(model_energy_fn, batches):
                        batch1, batch2, reg_main_prop = batches
                        loss1 = loss1_fn(
                            model_energy_fn, batch1[2:], energies=batch1[:2]
                        )

                        if reg_estimator == "dsm":
                            targets, weights, predictions = batch2
                            loss2 = torch.mean(
                                torch.square(weights * predictions - targets)
                            )
                        elif reg_estimator == "fm":
                            targets, predictions = batch2
                            loss2 = torch.mean(torch.square(predictions - targets))

                        return (
                            loss1,
                            loss2,
                            main_weight * reg_main_prop * loss1
                            + reg_weight * (1.0 - reg_main_prop) * loss2,
                        )

                elif combine_type == "default":

                    @torch.enable_grad()
                    def loss_fn(model_energy_fn, batches):
                        batch1, batch2, xs, ts, reg_main_prop = batches
                        xs.requires_grad_(True)
                        ys = batch1[0]
                        taus = batch1[1]
                        all_xs = torch.cat([xs, ys], dim=0)
                        all_ts = torch.cat([ts, taus], dim=0)
                        all_energies = model_energy_fn(all_xs, all_ts)
                        energy_xs, energy_ys = torch.split(
                            all_energies, [xs.shape[0], ys.shape[0]], dim=0
                        )
                        loss1 = loss1_fn(
                            model_energy_fn, batch1[2:], energies=[energy_xs, energy_ys]
                        )
                        loss2 = loss2_fn(
                            model_energy_fn, batch2, energies=[energy_xs, xs]
                        )
                        return (
                            loss1,
                            loss2,
                            (
                                main_weight * reg_main_prop * loss1
                                + reg_weight * (1.0 - reg_main_prop) * loss2
                            ),
                        )

            else:
                # dual SM / FM

                @torch.enable_grad()
                def loss_fn(model_energy_fn, batches):
                    batch1, batch2, reg_main_prop = batches

                    ctsm_targets, ctsm_weights, ctsm_predictions = batch1
                    loss1 = torch.mean(
                        torch.square(ctsm_weights * ctsm_predictions - ctsm_targets)
                    )
                    if reg_estimator == "dsm":
                        dsm_targets, dsm_weights, dsm_predictions = batch2
                        loss2 = torch.mean(
                            torch.square(dsm_weights * dsm_predictions - dsm_targets)
                        )
                    elif reg_estimator == "fm":
                        fm_targets, fm_predictions = batch2
                        loss2 = torch.mean(torch.square(fm_predictions - fm_targets))

                    return (
                        loss1,
                        loss2,
                        main_weight * reg_main_prop * loss1
                        + reg_weight * (1.0 - reg_main_prop) * loss2,
                    )

        else:
            raise NotImplementedError

    return loss_fn


def get_loss_fn(
    estimator,
    reg_estimator,
    num_noise_classes,  # for CNCE
    num_cla_classes=2,
    param="ratio",  # for TNCE and TCNCE
    nce_nu=1,  # for NCE
    cnce_nu=1,  # for CNCE
    main_weight=1.0,
    reg_weight=1.0,
    combine_type="default",
    mixture_loss=False,
    rne=False,
    prob_path=None,
):
    def get_loss_from_estimator(estimator):
        if estimator == "nce":
            return get_nce_loss_fn(nce_nu)
        elif estimator == "cnce":
            return get_cnce_loss_fn(cnce_nu=cnce_nu)
        elif estimator == "stnce":
            return get_stnce_loss_fn(nce_nu=nce_nu, cnce_nu=cnce_nu)
        elif estimator == "tnce":
            return get_tnce_loss_fn(
                param=param,
                num_noise_classes=num_noise_classes,
                num_cla_classes=num_cla_classes,
            )
        elif estimator == "tcnce":
            return get_tcnce_loss_fn(param=param, num_noise_classes=num_noise_classes)
        elif estimator == "tstnce":
            return get_tstnce_loss_fn(param=param, rne=rne)
        elif estimator == "dsm":
            return get_dsm_loss_fn(param=param)
        elif estimator == "ctsm":
            return get_ctsm_loss_fn(param=param)
        elif estimator == "fm":
            return get_fm_loss_fn(prob_path=prob_path, param=param)

    if reg_estimator == "none":
        loss_fn = get_loss_from_estimator(estimator)
    else:
        loss1_fn = get_loss_from_estimator(estimator)
        loss2_fn = get_loss_from_estimator(reg_estimator)

        loss_fn = get_combined_loss_fn(
            loss1_fn,
            loss2_fn,
            main_weight=main_weight,
            reg_weight=reg_weight,
            estimator=estimator,
            reg_estimator=reg_estimator,
            combine_type=combine_type,
            mixture_loss=mixture_loss,
        )

    return loss_fn


def get_step_fn(
    estimator,
    num_noise_classes,
    num_cla_classes,
    train,
    reg_estimator=None,
    nce_nu=1,
    cnce_nu=1,
    param="ratio",
    use_ema=True,
    main_weight=1.0,
    reg_weight=1.0,
    combine_type="default",
    mixture_loss=False,
    rne=False,
    grad_norm=-1.0,
    accelerator=None,
    prob_path=None,
):

    loss_fn = get_loss_fn(
        estimator=estimator,
        reg_estimator=reg_estimator,
        num_noise_classes=num_noise_classes,
        num_cla_classes=num_cla_classes,
        param=param,
        nce_nu=nce_nu,
        cnce_nu=cnce_nu,
        main_weight=main_weight,
        reg_weight=reg_weight,
        combine_type=combine_type,
        mixture_loss=mixture_loss,
        rne=rne,
        prob_path=prob_path,
    )

    if train:

        def step_fn(state, batch, build_noise_fn):
            model = state["model"]
            optimizer = state["optimizer"]
            optimizer.zero_grad()

            # build noise batch
            batch = build_noise_fn(
                original_data=batch, train=True, model_energy_fn=model
            )

            losses = loss_fn(model, batch)
            if type(losses) == tuple:
                loss = losses[-1]
            else:
                loss = losses
            if accelerator is None:
                loss.backward()
            else:
                accelerator.backward(loss)
            if grad_norm > 0:
                if accelerator is None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_norm)
                elif accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), grad_norm)
            # optimize_fn
            optimizer.step()
            if use_ema:
                state["ema"].update(model.parameters())
            return losses

    else:

        def step_fn(state, batch, build_noise_fn):
            model = state["model"]
            with torch.no_grad():
                if use_ema:
                    ema = state["ema"]
                    ema.store(model.parameters())
                    ema.copy_to(model.parameters())

                # build noise batch
                batch = build_noise_fn(
                    original_data=batch, train=False, model_energy_fn=model
                )

                losses = loss_fn(model, batch)
                if use_ema:
                    ema.restore(model.parameters())
            return losses

    return step_fn
