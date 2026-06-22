def import_extensions():
    import internutopia_extension.controllers  # noqa: F401
    import internutopia_extension.metrics  # noqa: F401
    import internutopia_extension.objects  # noqa: F401
    import internutopia_extension.robots  # noqa: F401
    import internutopia_extension.sensors  # noqa: F401
    import internutopia_extension.tasks  # noqa: F401

    from . import controllers, robots, sensors, tasks  # noqa: F401
    from .metrics.vln_pe_metrics import VLNPEMetrics
    from .tasks.vln_eval_task import VLNEvalTask  # noqa: F401
