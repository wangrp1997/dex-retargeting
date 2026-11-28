"""Algorithm labels shared by tracking, worker telemetry and hardware logs."""


def botyard_version(opposition):
    if opposition == 'v4_2':
        return '4.2'
    if opposition == 'inward':
        return '4.3'
    raise ValueError(f'Unknown Botyard opposition mode: {opposition}')
