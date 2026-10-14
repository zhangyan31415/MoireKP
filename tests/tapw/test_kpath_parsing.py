import numpy as np
from tapw.io.kpath import parse_kpath_labels, read_kpath


def test_shared_parser_merges_disconnected_segment_labels():
    lines = ['K path\n', '2\n', 'Line-mode\n', 'Reciprocal\n',
             '0 0 0 G\n', '.5 0 0 M\n', '0 .5 0 K\n', '0 0 0 G\n']
    labels, ticks = parse_kpath_labels(lines)
    assert labels == [r'${\Gamma}$', r'${\mathrm{M}}$|${\mathrm{K}}$', r'${\Gamma}$']
    assert ticks == [0, 2, 4]


def test_shared_reader_uses_path_distance_column(tmp_path):
    source = tmp_path/'KPATH.in'
    source.write_text('K path\n2\nLine-mode\nReciprocal\n0 0 0 G\n.5 0 0 M\n')
    output = tmp_path/'KPATH.out'
    np.savetxt(output, [[0, 0, 0, 0], [.25, 0, 0, .75], [.5, 0, 0, 1.5]])
    labels, coordinates, ticks = read_kpath(source, output)
    assert labels == [r'${\Gamma}$', r'${\mathrm{M}}$']
    assert ticks == [0, 2]
    np.testing.assert_array_equal(coordinates, [0, .75, 1.5])
