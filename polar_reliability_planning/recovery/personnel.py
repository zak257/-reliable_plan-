"""A crew occupies one unit until its startup demand completes."""


def may_dispatch(access_safe: bool) -> bool:
    return bool(access_safe)


def advance_preparation(done: float, required: float, onsite: bool,
                        access_safe: bool, mode: str) -> float:
    if mode not in ("arrival_only", "all_work_safe"):
        raise ValueError("Unknown personnel mode")
    if onsite and (access_safe or mode == "arrival_only"):
        return min(required, done + 1.0)
    return done
