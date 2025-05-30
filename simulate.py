import hydra
import torch
from tqdm import tqdm
import os,math

from sequence_single_index import (
    PlainFeedForward,
    TiedFeedForward,
    OverlappedInitialCondition,
    RandomInitialCondition,
    SequenceSingleIndex,
    NormalizedSGD,
    GaussianTeacherDataset,
    NoPositionalEncoding,
    OppositePairedPositionalEncoding,
    NoAttentionReduction,
    ProjectionAttentionReduction,
    TraceAttentionReduction,
    SingleIndexAttention,
    PositionalSemanticTransitionModel,
    FrobeniusNormLoss,
)

from hydra_extension import cfg_to_string
import simulation_conf

@hydra.main(config_path="experiments")
def main(cfg):
    # Using single thread is much faster with online SGD
    torch.set_num_threads(1)
    
    uid = cfg_to_string(cfg)
    filename = hydra.utils.to_absolute_path(cfg.machine.computation_storage_path+'/'+uid+'.pt')

    if cfg.machine.skip_existing and os.path.exists(filename):
        print(f"Skipping {uid}")
        return
    print(f"Computing {uid}")
    
    # read d and L from the configuration file
    d = cfg.run.d
    L = cfg.run.L
    
    # Set default device to cuda, mps, or cpu in this order of preference
    device = torch.device(cfg.machine.device)

    match cfg.run.ic_type:
        case "overlapped":
            initial_condition = OverlappedInitialCondition(
                d=d,
                seed=cfg.run.get('ic_seed', None),
                random_sign=cfg.run.get('ic_randomsign', False),
                overlap=cfg.run.get('ic_overlap', None),
            )
        case "random":
            initial_condition = RandomInitialCondition(d=d, seed=cfg.run.get('ic_seed', None), normalized=cfg.run.get('ic_normalized', True))
        case _:
            raise ValueError(f"Unknown initial condition type {cfg.run.ic_type}")

    # Retrocompatibility with old config files
    try:
        teacher_type = cfg.run.teacher.type
    except AttributeError:
        teacher_type = "ssi"
    #
    try:
        teacher_function = cfg.run.teacher.function
    except AttributeError:
        try:
            teacher_function = cfg.run.teacher_function
        except AttributeError:
            pass
    ##
    match teacher_type:
        case "ssi":
            teacher = SequenceSingleIndex(
                hidden_direction=initial_condition.teacher(),
                function=simulation_conf.ssi_function[teacher_function],
                require_grad=False,
            ).to(device)
        case "positional_semantic_model":
            teacher = PositionalSemanticTransitionModel(
                hidden_direction=initial_condition.teacher(),
                omega=cfg.run.teacher.omega,
                A=torch.softmax(torch.tensor(cfg.run.teacher.pre_softmax_A, dtype=torch.float32), dim=-1),
            ).to(device)
        case _:
            raise ValueError(f"Unknown teacher type {teacher_type}")

    # Retrocompatibility with old config files
    try:
        student_type = cfg.run.student.type
    except:
        student_type = cfg.run.student_type
    #
    try:
        student_function = cfg.run.student.function
    except:
        try:
            student_function = cfg.run.student_function
        except AttributeError:
            pass 
    #
    try:
        student_activation = cfg.run.student.activation
    except AttributeError:
        try:
            student_activation = cfg.run.student_activation
        except AttributeError:
            pass
    ####
    match student_type:
        case "ssi":
            student = SequenceSingleIndex(
                hidden_direction=initial_condition.student(),
                function=simulation_conf.ssi_function[student_function],
                require_grad=True
            ).to(device)
        case "plain_network":
            student = PlainFeedForward(
                hidden_direction=initial_condition.student(L),
                width=1,
                activation=simulation_conf.activation[student_activation],
                require_grad=True
            ).to(device)
        case "tied_network":
            student = TiedFeedForward(
                hidden_direction=initial_condition.student(),
                width=1,
                activation=simulation_conf.activation[student_activation],
                require_grad=True
            ).to(device)
        case "attention":
            student_initial_weight = initial_condition.student(1)

            # Set up the positional encoding
            match cfg.run.student.positional_encoding.type:
                case "none":
                    pe = NoPositionalEncoding()
                case "opposite":
                    pe = OppositePairedPositionalEncoding(
                        d=cfg.run.d,
                        seed=cfg.run.student.positional_encoding.get('seed', None),
                        orthogonal_to=torch.stack((student_initial_weight, initial_condition.teacher()))
                    )
                case _:
                    raise ValueError(f"Unknown positional encoding {cfg.run.student.positional_encoding}")
                
            # Set up the reduction
            match cfg.run.student.reduction.type:
                case "none":
                    reduction = NoAttentionReduction()
                case "projection":
                    # Set up the projections
                    match cfg.run.student.reduction.choice:
                        case "random":
                            a_generator = torch.Generator()
                            a_generator.manual_seed(cfg.run.student.a.seed)
                            a_left = torch.randn(size=(cfg.run.L,), generator=a_generator)
                            a_right = torch.randn(size=(cfg.run.L,), generator=a_generator)
                        case "uniform":
                            a_left = torch.ones(cfg.run.L)
                            a_right = torch.ones(cfg.run.L)
                        case _:
                            raise ValueError(f"Unknown a type {cfg.run.student.a.type}")
                    a_norm = cfg.run.student.a.get('norm', 1.)
                    a_left /= torch.norm(a_left) / a_norm
                    a_right /= torch.norm(a_right) / a_norm
                    reduction = ProjectionAttentionReduction(
                        a_left=a_left,
                        a_right=a_right,
                    )
                case "trace":
                    reduction = TraceAttentionReduction()
                case _:
                    raise ValueError(f"Unknown reduction type {cfg.run.student.reduction.type}")
            student = SingleIndexAttention(
                hidden_direction=student_initial_weight,
                require_grad=True,
                positional_encoding=pe,
                reduction=reduction,
            ) 
        case _:
            raise ValueError(f"Unknown student type {student_type}")
        
    measure_every = cfg.run.T // cfg.run.number_of_measures

    learning_rate = (
        cfg.run.learning_rate_coefficient *
        simulation_conf.scaling_functions[cfg.run.learning_rate_scaling_with_L](cfg.run.L)
    )

    # Retrocompatibility with old config files
    try:
        loss_type = cfg.run.loss
    except AttributeError:
        loss_type = "square"
    #
    match loss_type:
        case "square":
            loss = torch.nn.MSELoss()
        case "frobenius":
            loss = FrobeniusNormLoss()
        case _:
            raise ValueError(f"Unknown loss type {loss_type}")
        
    optimizer = NormalizedSGD(student.parameters(), lr=learning_rate)

    
    metrics = [
        simulation_conf.metrics[metric_name](
            student=student,
            teacher=teacher,
            L=L,
            **cfg.run.get(f'metric_kwargs.{metric_name}', {}),
        ) for metric_name in cfg.run.metrics
    ]

    dataset = GaussianTeacherDataset(
        teacher,
        L=L,
        seed = cfg.run.data_seed,
        number_of_samples=cfg.run.T
    )

    for i, xy in tqdm(enumerate(dataset), total=cfg.run.T):
        # Store metrics
        if i % measure_every == 0:
            for metric in metrics:
                metric(step=i)

        # Gradient descent step
        x, y = xy
        optimizer.zero_grad()
        l = loss(y, student(x))
        l.backward()
        optimizer.step()

    # store the results in a single file
    torch.save({
        metric.name: metric.data for metric in metrics
    }, filename)

if __name__ == "__main__":
    main()


