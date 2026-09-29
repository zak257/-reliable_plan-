"""Restricted incumbent improvement; never transfer restrictions to a free MILP."""
from gurobipy import GRB


def is_battery_mode(variable):
    return '_battery_ch_on[' in variable.VarName or '_battery_dis_on[' in variable.VarName


def saved_binary(variable, values):
    if variable.LB == variable.UB:
        return variable.LB
    if variable.VarName not in values:
        raise ValueError(f'No saved policy value for {variable.VarName}')
    value = values[variable.VarName]
    if abs(value-round(value)) > 1e-6 or not variable.LB-1e-8 <= value <= variable.UB+1e-8:
        raise ValueError(f'Invalid saved binary {variable.VarName}: {value}')
    return float(round(value))


def fix_incumbent_commitment(planner, values):
    """Retain diesel/observation policy but release capacity, storage and tasks.

    The caller builds with only the diesel count fixed. Storage mode indicators
    remain in the model, allowing their bounds to be restored after a quick
    capacity-polishing phase. Other fixed indicators are expanded exactly.
    """
    if set(planner.fixed_modules or {}) != {'diesel_units'}:
        raise ValueError('Restricted improvement must fix only the diesel count')
    m = planner.model
    count = 0
    for variable in m.getVars():
        if variable.VType == GRB.BINARY and not is_battery_mode(variable):
            value = saved_binary(variable, values)
            variable.LB = variable.UB = value
            count += 1
    m.update()
    removed = []
    active = 0
    for indicator in m.getGenConstrs():
        if indicator.GenConstrType != GRB.GENCONSTR_INDICATOR:
            raise ValueError('Unexpected non-indicator general constraint')
        control, value, expression, sense, rhs = m.getGenConstrIndicator(indicator)
        if control.LB != control.UB:
            continue
        if int(round(control.LB)) == value:
            m.addLConstr(expression, sense, rhs)
            active += 1
        removed.append(indicator)
    m.remove(removed)
    m.update()
    return dict(fixed_policy_binaries=count, expanded_active_indicators=active,
        removed_fixed_indicators=len(removed), remaining_indicators=m.NumGenConstrs,
        scope='temporary incumbent improvement only; free model rebuilt separately')


def temporarily_fix_battery_modes(planner, values):
    bounds = []
    for variable in planner.model.getVars():
        if variable.VType == GRB.BINARY and is_battery_mode(variable):
            bounds.append((variable, variable.LB, variable.UB))
            value = saved_binary(variable, values)
            variable.LB = variable.UB = value
    planner.model.update()
    return bounds


def restore_bounds(model, bounds):
    for variable, lower, upper in bounds:
        variable.LB, variable.UB = lower, upper
    model.update()
