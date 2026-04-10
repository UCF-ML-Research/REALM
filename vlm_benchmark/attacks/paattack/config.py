"""PA-Attack CLI configuration."""


def resolve_cli_kwargs(attack_name, args, context):
    """Build kwargs for AttackRegistry.create() from CLI args."""
    kwargs = {"device": args.device}
    if getattr(args, "epsilon", None) is not None:
        kwargs["epsilon"] = args.epsilon
    if getattr(args, "steps", None) is not None:
        kwargs["max_iterations"] = args.steps
    return kwargs
