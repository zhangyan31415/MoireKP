#!/usr/bin/env python3
"""Draw the appendix TAPW basis figure from the supplied portable reference table."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle, Polygon, Rectangle
from scipy.spatial import Voronoi
import numpy as np
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'verification'/'appendix_figure'
OUT.mkdir(parents=True,exist_ok=True)
RED,BLUE,INK='#C43B3B','#2E68A6','#202327'
plt.rcParams.update({'font.family':'DejaVu Sans','mathtext.fontset':'dejavusans','font.size':9,'pdf.fonttype':42,'ps.fonttype':42})
def ws_cell(basis):
    indices = np.array([(i, j) for i in range(-3, 4) for j in range(-3, 4)])
    points = indices @ basis
    vor = Voronoi(points)
    origin = np.where(np.all(indices == 0, axis=1))[0][0]
    region = vor.regions[vor.point_region[origin]]
    assert -1 not in region
    vertices = vor.vertices[region]
    return vertices[np.argsort(np.arctan2(vertices[:, 1], vertices[:, 0]))]

def radial_shells(points, center, tolerance=2e-05):
    distances = np.linalg.norm(points - center, axis=1)
    radii, shell = ([], np.empty(len(points), dtype=int))
    for i in np.argsort(distances):
        if not radii or abs(distances[i] - radii[-1]) > tolerance:
            radii.append(float(distances[i]))
        shell[i] = len(radii)
    return (np.array(radii), shell)

def tiling(ax, zoom=False):
    centers = []
    if zoom:
        idx0 = np.rint(k_h @ np.linalg.inv(b)).astype(int)
        centers = [(np.array([i, j]) + idx0) @ b - k_h for i in range(-5, 6) for j in range(-5, 6)]
    else:
        centers = [np.array([i, j]) @ b for i in range(-42, 43) for j in range(-42, 43) if np.linalg.norm(np.array([i, j]) @ b) < 36]
    lines = []
    for center in centers:
        p = moire + center
        lines.extend([(p[i], p[(i + 1) % len(p)]) for i in range(len(p))])
    ax.add_collection(LineCollection(lines, colors='#D5DADF', linewidths=0.32 if zoom else 0.17, alpha=0.8 if zoom else 0.5, zorder=0))

def overview(ax):
    tiling(ax)
    for n, color in [(57, BLUE), (56, RED)]:
        ax.add_patch(Polygon(ws_cell(n * b), closed=True, fill=False, edgecolor=color, lw=1.3, zorder=3))
    ax.add_patch(Polygon(moire, closed=True, facecolor='#727C86', edgecolor=INK, lw=0.7, zorder=5))
    ax.scatter(g_g[:, 0], g_g[:, 1], s=10, facecolors='none', edgecolors=BLUE, lw=0.6, zorder=6)
    ax.scatter(g_h[:, 0], g_h[:, 1], s=3.5, color=RED, zorder=7)
    ax.scatter([0], [0], s=9, color=INK, zorder=7)
    ax.text(-1, -2.8, '$\\Gamma$', ha='right')
    ax.annotate('moire BZ', xy=(0, 0.2), xytext=(-17, -7), fontsize=9, arrowprops={'arrowstyle': '-', 'lw': 0.7, 'color': INK})
    ax.text(-31, 19, 'graphene BZ', color=BLUE, fontsize=9)
    ax.text(-31, 15, 'hBN BZ', color=RED, fontsize=9)
    center = (k_h + k_g) / 2
    ax.add_patch(Rectangle(center - np.array([4.3, 4.3]), 8.6, 8.6, fill=False, edgecolor=INK, lw=0.9, zorder=8))
    ax.annotate('selected valley\n(enlarged in b)', xy=center + np.array([4.3, 0]), xytext=(31, 23), fontsize=8, ha='center', va='top', arrowprops={'arrowstyle': '-', 'lw': 0.7})
    ax.text(0, -35, '$\\theta=0^\\circ$; lattice-mismatch moire', ha='center', fontsize=9)
    ax.set(xlim=(-38, 38), ylim=(-37, 37))
    ax.set_aspect('equal')
    ax.axis('off')

def detail(ax, material='hbn'):
    tiling(ax, zoom=True)
    center = k_h if material == 'hbn' else k_g
    points = g_h if material == 'hbn' else g_g
    rr = radii if material == 'hbn' else gradii
    ss = shells if material == 'hbn' else gshells
    color = RED if material == 'hbn' else BLUE
    shifted = points - k_h
    for r in rr:
        if r < 1e-05:
            continue
        ax.add_patch(Circle(center - k_h, r, fill=False, ls=(0, (3, 3)), lw=0.65, edgecolor='#7B838A', zorder=1))
    ax.scatter(shifted[:, 0], shifted[:, 1], s=35, color=color, edgecolors='white', linewidths=0.45, zorder=5)
    ax.scatter([0], [0], marker='D', s=30, color=RED, zorder=7)
    ax.annotate('$\\mathbf{K}_{\\rm hBN}$', xy=(0, 0), xytext=(-1.15, -0.36), fontsize=10, color=RED, arrowprops={'arrowstyle': '-', 'color': RED, 'lw': 0.6}, zorder=8)
    delta = k_g - k_h
    ax.scatter(*delta, marker='D', s=30, color=BLUE, zorder=7)
    ax.annotate('$\\mathbf{K}_{\\rm Gr}$', xy=delta, xytext=(1.0, 0.36), fontsize=10, color=BLUE, arrowprops={'arrowstyle': '-', 'color': BLUE, 'lw': 0.6}, zorder=8)
    for i, (r, angle) in enumerate(zip(rr, [95, 105, 115, 125, 135, 145]), 1):
        if r < 1e-05:
            continue
        pos = center - k_h + r * np.array([np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))])
        ax.text(*pos, str(i), ha='center', va='center', fontsize=8, color=INK, zorder=9, bbox={'facecolor': 'white', 'edgecolor': 'none', 'pad': 0.8})
    idx = np.flatnonzero(ss == 4)[0]
    start = shifted[idx]
    ax.annotate('', xy=center - k_h, xytext=start, arrowprops={'arrowstyle': '->', 'color': INK, 'lw': 1.05}, zorder=6)
    equation = '$\\mathbf{Q}=\\mathbf{K}_{\\rm hBN}-\\widetilde{\\mathbf{G}}$' if material == 'hbn' else '$\\mathbf{Q}=\\mathbf{K}_{\\rm Gr}-\\widetilde{\\mathbf{G}}$'
    ax.text(-2.65, 2.7, equation, fontsize=8, zorder=10, bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': 0.85, 'pad': 1})
    ax.set(xlim=(-3.05, 3.05), ylim=(-3.05, 3.05), aspect='equal')
    ax.set_xlabel('$(\\widetilde{G}_x-K_{{\\rm hBN},x})/|\\mathbf{b}_{M,1}|$')
    ax.set_ylabel('$(\\widetilde{G}_y-K_{{\\rm hBN},y})/|\\mathbf{b}_{M,1}|$')
    ax.set_xticks(np.arange(-3, 4))
    ax.set_yticks(np.arange(-3, 4))
    for spine in ax.spines.values():
        spine.set_color('#8A939B')
        spine.set_linewidth(0.65)
    ax.tick_params(length=3, width=0.6, labelsize=8)

data=np.genfromtxt(ROOT/'basis_reference'/'retained_vectors.csv',delimiter=',',names=True,dtype=None,encoding='utf-8')
basis=np.array(json.loads((ROOT/'basis_reference'/'geometry.json').read_text())['reciprocal_basis_Ainv'])
scale=float(np.linalg.norm(basis[0]));b=basis/scale
g_h=np.column_stack([data[data['material']=='hBN'][n] for n in ['Gx_Ainv','Gy_Ainv']])/scale
g_g=np.column_stack([data[data['material']=='graphene'][n] for n in ['Gx_Ainv','Gy_Ainv']])/scale
k_h,k_g=g_h.mean(axis=0),g_g.mean(axis=0)
radii,shells=radial_shells(g_h,k_h);gradii,gshells=radial_shells(g_g,k_g)
assert len(g_h)==27 and len(g_g)==37 and len(radii)==len(gradii)==6
moire=ws_cell(b)
ns=globals()
def style_panel(ax, index, compact):
    factor = 0.7 if compact else 1.0
    if index == 0:
        ns['overview'](ax)
        for patch in ax.patches:
            if not patch.get_fill():
                patch.set_linewidth(0.5 if compact else 0.7)
        for text in ax.texts:
            if 'selected valley' in text.get_text():
                text.set_text('selected valley\n(panels b,c)')
            if 'moire' in text.get_text():
                text.set_text(text.get_text().replace('moire', 'moiré'))
    else:
        ns['detail'](ax, 'hbn' if index == 1 else 'gr')
        ax.set(xlim=(-3.2, 3.8), ylim=(-3.0, 4.0))
        ax.set_xticks(np.arange(-3, 4))
        ax.set_yticks(np.arange(-3, 5))
        if index == 2:
            ax.text(-2.82, 3.55, 'shell 1: $\\mathbf{Q}=0$', fontsize=8, zorder=12, bbox={'facecolor': 'white', 'edgecolor': 'none', 'pad': 1})
        ax.xaxis.label.set_fontsize(6.4 if compact else 9)
        ax.yaxis.label.set_fontsize(6.4 if compact else 9)
        ax.tick_params(labelsize=6 if compact else 8, length=2.3 if compact else 3)
    for text in ax.texts:
        text.set_fontsize(text.get_fontsize() * factor)
    for coll in ax.collections:
        if hasattr(coll, 'get_sizes') and len(coll.get_sizes()):
            if index == 0 and len(coll.get_offsets()) > 1:
                is_graphene = len(coll.get_offsets()) == 37
                size = (0.55 if is_graphene else 0.3) if compact else 2.3 if is_graphene else 1.2
                coll.set_sizes([size])
                coll.set_linewidths([0.12 if compact else 0.22] if is_graphene else [0.0])
            else:
                coll.set_sizes(coll.get_sizes() * (0.42 if compact else 1.0))
    titles = ['(a) Reciprocal-space construction', '(b) hBN: 27 reciprocal vectors', '(c) Graphene: 37 reciprocal vectors']
    ax.set_title(titles[index], fontsize=7.5 if compact else 11, pad=10 if compact else 13)

def shell_table(fig, x, y, index, compact):
    if index == 0:
        return
    counts = '3  3  6  6  3  6' if index == 1 else '1  6  6  6 12  6'
    fig.text(x, y, 'shell: 1  2  3  4  5  6\ncount: ' + counts, ha='center', va='center', fontsize=6.2 if compact else 8.5, fontfamily='DejaVu Sans Mono', linespacing=1.5)

def save(fig, basename):
    for ext in ['pdf', 'svg', 'png']:
        fig.savefig(OUT / f'{basename}.{ext}', dpi=450, facecolor='white')
    plt.close(fig)
fig, axes = plt.subplots(1, 3, figsize=(7.25, 3.1), gridspec_kw={'width_ratios': [1.02, 1, 1]})
fig.subplots_adjust(left=0.015, right=0.987, top=0.865, bottom=0.235, wspace=0.32)
for i, ax in enumerate(axes):
    style_panel(ax, i, compact=True)
    if i:
        box = ax.get_position()
        shell_table(fig, (box.x0 + box.x1) / 2, 0.075, i, compact=True)
save(fig, 'tapw_qshell6_appendix')
basenames = ['panel_a_reciprocal_construction', 'panel_b_hbn_qshell6', 'panel_c_graphene_qshell6']
for i, name in enumerate(basenames):
    fig, ax = plt.subplots(figsize=(5.1, 5.35))
    fig.subplots_adjust(left=0.17 if i else 0.04, right=0.96, bottom=0.23, top=0.9)
    style_panel(ax, i, compact=False)
    shell_table(fig, 0.565, 0.07, i, compact=False)
    save(fig, name)
