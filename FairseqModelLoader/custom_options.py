import argparse
from pathlib import Path
from typing import Callable, List, Optional, Union

import torch
from fairseq import utils
from fairseq.data.indexed_dataset import get_available_dataset_impl
from fairseq.dataclass.configs import (
    CheckpointConfig,
    CommonConfig,
    CommonEvalConfig,
    DatasetConfig,
    DistributedTrainingConfig,
    EvalLMConfig,
    GenerationConfig,
    InteractiveConfig,
    OptimizationConfig,
    EMAConfig,
)
from fairseq.dataclass.utils import gen_parser_from_dataclass

# this import is for backward compatibility
from fairseq.utils import csv_str_list, eval_bool, eval_str_dict, eval_str_list  # noqa

#error methodのオーバーライド
class ThrowingArgumentParser(argparse.ArgumentParser):
    def error(self, message):
        pass

def get_parser(desc, default_task="translation"):
    # Before creating the true parser, we need to import optional user module
    # in order to eagerly import custom tasks, optimizers, architectures, etc.
    usr_parser = ThrowingArgumentParser(add_help=False, allow_abbrev=False)
    usr_parser.add_argument("--user-dir", default=None)
    usr_args, _ = usr_parser.parse_known_args()
    utils.import_user_module(usr_args)

    parser = ThrowingArgumentParser(allow_abbrev=False)
    gen_parser_from_dataclass(parser, CommonConfig())

    from fairseq.registry import REGISTRIES

    for registry_name, REGISTRY in REGISTRIES.items():
        parser.add_argument(
            "--" + registry_name.replace("_", "-"),
            default=REGISTRY["default"],
            choices=REGISTRY["registry"].keys(),
        )

    # Task definitions can be found under fairseq/tasks/
    from fairseq.tasks import TASK_REGISTRY

    parser.add_argument(
        "--task",
        metavar="TASK",
        default=default_task,
        choices=TASK_REGISTRY.keys(),
        help="task",
    )
    # fmt: on
    return parser


def get_training_parser(default_task="translation"):
    parser = get_parser("Trainer", default_task)
    add_dataset_args(parser, train=True)
    add_distributed_training_args(parser)
    add_model_args(parser)
    add_optimization_args(parser)
    add_checkpoint_args(parser)
    add_ema_args(parser)
    return parser

def add_dataset_args(parser, train=False, gen=False):
    group = parser.add_argument_group("dataset_data_loading")
    gen_parser_from_dataclass(group, DatasetConfig())
    # fmt: on
    return group

def add_distributed_training_args(parser, default_world_size=None):
    group = parser.add_argument_group("distributed_training")
    if default_world_size is None:
        default_world_size = max(1, torch.cuda.device_count())
    gen_parser_from_dataclass(
        group, DistributedTrainingConfig(distributed_world_size=default_world_size)
    )
    return group


def add_optimization_args(parser):
    group = parser.add_argument_group("optimization")
    # fmt: off
    gen_parser_from_dataclass(group, OptimizationConfig())
    # fmt: on
    return group


def add_checkpoint_args(parser):
    group = parser.add_argument_group("checkpoint")
    # fmt: off
    gen_parser_from_dataclass(group, CheckpointConfig())
    # fmt: on
    return group

def add_model_args(parser):
    group = parser.add_argument_group("Model configuration")
    # fmt: off

    # Model definitions can be found under fairseq/models/
    #
    # The model architecture can be specified in several ways.
    # In increasing order of priority:
    # 1) model defaults (lowest priority)
    # 2) --arch argument
    # 3) --encoder/decoder-* arguments (highest priority)
    from fairseq.models import ARCH_MODEL_REGISTRY
    group.add_argument('--arch', '-a', metavar='ARCH',
                       choices=ARCH_MODEL_REGISTRY.keys(),
                       help='model architecture')
    # fmt: on
    return group


def get_args(
    data: Union[str, Path],
    task: str = "translation",
    arch: str = "transformer",
    **overrides
):
    parser = get_training_parser(task)
    args = parse_args_and_arch(parser, [str(data), "--task", task, "--arch", arch])

    for k, v in overrides.items():
        setattr(args, k, v)

    return args


def add_ema_args(parser):
    group = parser.add_argument_group("EMA configuration")
    gen_parser_from_dataclass(group, EMAConfig())






def parse_args_and_arch(
    parser: argparse.ArgumentParser,
    input_args: List[str] = None,
    parse_known: bool = False,
    suppress_defaults: bool = False,
    modify_parser: Optional[Callable[[argparse.ArgumentParser], None]] = None,
):
    """
    Args:
        parser (ArgumentParser): the parser
        input_args (List[str]): strings to parse, defaults to sys.argv
        parse_known (bool): only parse known arguments, similar to
            `ArgumentParser.parse_known_args`
        suppress_defaults (bool): parse while ignoring all default values
        modify_parser (Optional[Callable[[ArgumentParser], None]]):
            function to modify the parser, e.g., to set default values
    """

    if suppress_defaults:
        # Parse args without any default values. This requires us to parse
        # twice, once to identify all the necessary task/model args, and a second
        # time with all defaults set to None.
        args = parse_args_and_arch(
            parser,
            input_args=input_args,
            parse_known=parse_known,
            suppress_defaults=False,
        )
        suppressed_parser = ThrowingArgumentParser(add_help=False, parents=[parser])
        suppressed_parser.set_defaults(**{k: None for k, v in vars(args).items()})
        args = suppressed_parser.parse_args(input_args)
        return argparse.Namespace(
            **{k: v for k, v in vars(args).items() if v is not None}
        )

    from fairseq.models import ARCH_MODEL_REGISTRY, ARCH_CONFIG_REGISTRY, MODEL_REGISTRY

    # Before creating the true parser, we need to import optional user module
    # in order to eagerly import custom tasks, optimizers, architectures, etc.
    usr_parser = ThrowingArgumentParser(add_help=False, allow_abbrev=False)
    usr_parser.add_argument("--user-dir", default=None)
    usr_args, _ = usr_parser.parse_known_args(input_args)
    utils.import_user_module(usr_args)

    if modify_parser is not None:
        modify_parser(parser)

    # The parser doesn't know about model/criterion/optimizer-specific args, so
    # we parse twice. First we parse the model/criterion/optimizer, then we
    # parse a second time after adding the *-specific arguments.
    # If input_args is given, we will parse those args instead of sys.argv.
    args, _ = parser.parse_known_args(input_args)

    # Add model-specific args to parser.
    if hasattr(args, "arch"):
        model_specific_group = parser.add_argument_group(
            "Model-specific configuration",
            # Only include attributes which are explicitly given as command-line
            # arguments or which have default values.
            argument_default=argparse.SUPPRESS,
        )
        if args.arch in ARCH_MODEL_REGISTRY:
            ARCH_MODEL_REGISTRY[args.arch].add_args(model_specific_group)
        elif args.arch in MODEL_REGISTRY:
            MODEL_REGISTRY[args.arch].add_args(model_specific_group)
        else:
            pass
            #raise RuntimeError()

    if hasattr(args, "task"):
        from fairseq.tasks import TASK_REGISTRY

        TASK_REGISTRY[args.task].add_args(parser)
    if getattr(args, "use_bmuf", False):
        # hack to support extra args for block distributed data parallelism
        from fairseq.optim.bmuf import FairseqBMUF

        FairseqBMUF.add_args(parser)

    # Add *-specific args to parser.
    from fairseq.registry import REGISTRIES

    for registry_name, REGISTRY in REGISTRIES.items():
        choice = getattr(args, registry_name, None)
        if choice is not None:
            cls = REGISTRY["registry"][choice]
            if hasattr(cls, "add_args"):
                cls.add_args(parser)
            elif hasattr(cls, "__dataclass"):
                gen_parser_from_dataclass(parser, cls.__dataclass())

    # Modify the parser a second time, since defaults may have been reset
    if modify_parser is not None:
        modify_parser(parser)

    # Parse a second time.
    parser.set_defaults(update_epoch_batch_itr=1)
    if parse_known:
        args, extra = parser.parse_known_args(input_args)
    else:
        args = parser.parse_args(input_args)
        extra = None
    # Post-process args.
    if (
        hasattr(args, "batch_size_valid") and args.batch_size_valid is None
    ) or not hasattr(args, "batch_size_valid"):
        args.batch_size_valid = args.batch_size
    if hasattr(args, "max_tokens_valid") and args.max_tokens_valid is None:
        args.max_tokens_valid = args.max_tokens
    if getattr(args, "memory_efficient_fp16", False):
        args.fp16 = True
    if getattr(args, "memory_efficient_bf16", False):
        args.bf16 = True
    args.tpu = getattr(args, "tpu", False)
    args.bf16 = getattr(args, "bf16", False)
    if args.bf16:
        args.tpu = True
    if args.tpu and args.fp16:
        raise ValueError("Cannot combine --fp16 and --tpu, use --bf16 on TPUs")

    if getattr(args, "seed", None) is None:
        args.seed = 1  # default seed for training
        args.no_seed_provided = True
    else:
        args.no_seed_provided = False

    if getattr(args, "update_epoch_batch_itr", None) is None:
        if hasattr(args, "grouped_shuffling"):
            args.update_epoch_batch_itr = args.grouped_shuffling
        else:
            args.grouped_shuffling = False
            args.update_epoch_batch_itr = False

    # Apply architecture configuration.
    if hasattr(args, "arch") and args.arch in ARCH_CONFIG_REGISTRY:
        ARCH_CONFIG_REGISTRY[args.arch](args)

    if parse_known:
        return args, extra
    else:
        return args