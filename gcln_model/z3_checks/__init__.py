# from .check_index import get_checks
import z3_checks.cohendiv_z3
import z3_checks.divbin_z3
import z3_checks.mannadiv_z3
import z3_checks.sqrt1_z3
import z3_checks.dijkstra_z3
import z3_checks.cohencu_z3
import z3_checks.egcd_z3
import z3_checks.fermat1_z3
import z3_checks.fermat2_z3
import z3_checks.freire1_z3
import z3_checks.freire2_z3
import z3_checks.geo1_z3
import z3_checks.geo2_z3
import z3_checks.geo3_z3
import z3_checks.hard_z3
import z3_checks.prod4br_z3
import z3_checks.prodbin_z3
import z3_checks.ps2_z3
import z3_checks.ps3_z3
import z3_checks.ps4_z3
import z3_checks.ps5_z3
import z3_checks.ps6_z3
import z3_checks.lcm2_z3
import z3_checks.core
from z3_checks.core import *
import z3


def get_z3_checks(name, loop_idx, z3_vars, z3_var2s):
    if name == 'cohencu':
        return cohencu_z3.get_checks(z3_vars, z3_var2s, loop_idx)

    if name == 'cohendiv':
        return cohendiv_z3.get_checks(z3_vars, z3_var2s, loop_idx)

    if name == 'dijkstra':
        return dijkstra_z3.get_checks(z3_vars, z3_var2s, loop_idx)

    if name == 'divbin':
        return divbin_z3.get_checks(z3_vars, z3_var2s, loop_idx)

    if name == 'egcd':
        return egcd_z3.get_checks(z3_vars, z3_var2s, loop_idx)

    if name == 'fermat1':
        return fermat1_z3.get_checks(z3_vars, z3_var2s, loop_idx)

    if name == 'fermat2':
        return fermat2_z3.get_checks(z3_vars, z3_var2s, loop_idx)

    if name == 'freire1':
        return freire1_z3.get_checks(z3_vars, z3_var2s, loop_idx)

    if name == 'freire2':
        return freire2_z3.get_checks(z3_vars, z3_var2s, loop_idx)

    if name == 'geo1':
        return geo1_z3.get_checks(z3_vars, z3_var2s, loop_idx)

    if name == 'geo2':
        return geo2_z3.get_checks(z3_vars, z3_var2s, loop_idx)

    if name == 'geo3':
        return geo3_z3.get_checks(z3_vars, z3_var2s, loop_idx)

    if name == 'hard':
        return hard_z3.get_checks(z3_vars, z3_var2s, loop_idx)

    if name == 'mannadiv':
        return mannadiv_z3.get_checks(z3_vars, z3_var2s, loop_idx)

    if name == 'prod4br':
        return prod4br_z3.get_checks(z3_vars, z3_var2s, loop_idx)

    if name == 'prodbin':
        return prodbin_z3.get_checks(z3_vars, z3_var2s, loop_idx)

    if name == 'ps2':
        return ps2_z3.get_checks(z3_vars, z3_var2s, loop_idx)

    if name == 'ps3':
        return ps3_z3.get_checks(z3_vars, z3_var2s, loop_idx)

    if name == 'ps4':
        return ps4_z3.get_checks(z3_vars, z3_var2s, loop_idx)

    if name == 'ps5':
        return ps5_z3.get_checks(z3_vars, z3_var2s, loop_idx)

    if name == 'ps6':
        return ps6_z3.get_checks(z3_vars, z3_var2s, loop_idx)

    if name == 'sqrt1':
        return sqrt1_z3.get_checks(z3_vars, z3_var2s, loop_idx)

    if name == 'knuth':
        return z3.And(False),z3.And(False),z3.And(False), z3.And(False),(), ()

    if name == 'lcm2':
        return lcm2_z3.get_checks(z3_vars, z3_var2s, loop_idx)


def full_z3_validation(name, loop_idx, z3_vars, invariant):
    if name == 'cohencu':
        return cohencu_z3.full_check(z3_vars, loop_idx, invariant)

    if name == 'cohendiv':
        return cohendiv_z3.full_check(z3_vars, loop_idx, invariant)

    if name == 'dijkstra':
        return dijkstra_z3.full_check(z3_vars, loop_idx, invariant)

    if name == 'divbin':
        return divbin_z3.full_check(z3_vars, loop_idx, invariant)

    if name == 'egcd':
        return egcd_z3.full_check(z3_vars, loop_idx, invariant)

    if name == 'fermat1':
        return fermat1_z3.full_check(z3_vars, loop_idx, invariant)

    if name == 'fermat2':
        return fermat2_z3.full_check(z3_vars, loop_idx, invariant)

    if name == 'freire1':
        return freire1_z3.full_check(z3_vars, loop_idx, invariant)

    if name == 'freire2':
        return freire2_z3.full_check(z3_vars, loop_idx, invariant)

    if name == 'geo1':
        return geo1_z3.full_check(z3_vars, loop_idx, invariant)

    if name == 'geo2':
        return geo2_z3.full_check(z3_vars, loop_idx, invariant)

    if name == 'geo3':
        return geo3_z3.full_check(z3_vars, loop_idx, invariant)

    if name == 'hard':
        return hard_z3.full_check(z3_vars, loop_idx, invariant)

    if name == 'mannadiv':
        return mannadiv_z3.full_check(z3_vars, loop_idx, invariant)

    if name == 'prod4br':
        return prod4br_z3.full_check(z3_vars, loop_idx, invariant)

    if name == 'prodbin':
        return prodbin_z3.full_check(z3_vars, loop_idx, invariant)

    if name == 'ps2':
        return ps2_z3.full_check(z3_vars, loop_idx, invariant)

    if name == 'ps3':
        return ps3_z3.full_check(z3_vars, loop_idx, invariant)

    if name == 'ps4':
        return ps4_z3.full_check(z3_vars, loop_idx, invariant)

    if name == 'ps5':
        return ps5_z3.full_check(z3_vars, loop_idx, invariant)

    if name == 'ps6':
        return ps6_z3.full_check(z3_vars, loop_idx, invariant)

    if name == 'sqrt1':
        return sqrt1_z3.full_check(z3_vars, loop_idx, invariant)
