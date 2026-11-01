"""CPU-only checks for the bounded Nsight operator plan."""
import pipeline_stage_profile_sweep as sweep


def test_command_uses_project_venv_and_one_selected_launch():
    command = sweep.command('/usr/local/cuda-13.3/bin/ncu', 3)
    assert command[-4:] == [str(sweep.ROOT / '.venv/bin/python'),
                            'stage_reduction_experiment.py', '--profile-one', '3']
    assert command[command.index('--launch-count') + 1] == '1'
    assert command[command.index('--cache-control') + 1] == 'none'
    assert command[command.index('--clock-control') + 1] == 'none'
    assert command[command.index('--replay-mode') + 1] == 'kernel'
    assert command[command.index('--metrics') + 1] == ','.join(sweep.METRICS)
    assert sweep.ORDER == (3, 2, 2, 3)


def test_current_metric_query_has_exact_names_and_units():
    units, hashes = sweep.query_units()
    assert set(units) == set(sweep.METRICS)
    assert units['sm__warps_active.avg.pct_of_peak_sustained_active']['unit'] == '%'
    assert units['smsp__warps_issue_stalled_long_scoreboard.avg.per_cycle_active']['unit'] == 'warp'
    assert units['dram__bytes.sum.per_second']['unit'] == 'byte/s'
    assert units['profiler__replayer_passes']['unit'] == 'pass'
    assert all(len(value) == 64 for value in hashes.values())
