"""Compact publication-size 2x6 layout for the saved four-setting benchmark."""
from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator, NullFormatter
import numpy as np

OUT=Path(__file__).resolve().parent
CASES=("mote2_k","zrs2_m","zrs2_gamma","aab_ka","aab_kb","aab_gamma")
TITLES=(r"Bilayer MoTe$_2$"+"\n"+r"$K$, 3.89°",
        r"Bilayer ZrS$_2$"+"\n"+r"$M$, 3.89°",
        r"Bilayer ZrS$_2$"+"\n"+r"$\Gamma$, 3.89°",
        r"Trilayer MoTe$_2$"+"\n"+r"$K_A$, 5.09°",
        r"Trilayer MoTe$_2$"+"\n"+r"$K_B$, 5.09°",
        r"Trilayer MoTe$_2$"+"\n"+r"$\Gamma$, 5.09°")
STYLES=(("#52748C","o"),("#D79A2E","s"),("#B67CA4","D"),("#BB534A","p"),("#47458A","X"))
LABELS=("Weights 0",r"Weighted $(6,6)$",r"Weighted $(20,20)$",r"Nonlinear $\lambda_b=1$",r"Nonlinear $\lambda_b=10$")


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def render_summary():
    source=OUT/'data/summary.json'
    payload=json.loads(source.read_text());data={'groups':payload['groups']};points=payload['stars'];source_points=source

    with np.load(OUT / "data/path_bands.npz", allow_pickle=False) as matrices:
        for index, group in enumerate(data["groups"]):
            for row in group["rows"]:
                prefix = f"s{index}_{row['case']}_k{row['kinetic_order']}"
                delta = 1000 * (matrices[prefix + "_model"] - matrices[prefix + "_tapw"])
                for field, shift in (("native_raw", 0.0),
                                     ("native_paper_aligned", row["paper_alignment_shift_mev"])):
                    np.testing.assert_allclose(np.sqrt(np.mean((delta + shift)**2)),
                                               row[field]["rms_mev"], atol=1e-8, rtol=0)
    verified=[]
    for index,g in enumerate(data['groups']):
        assert len(g['rows'])==(24 if index<3 else 30)
        verified.extend((index,r['case'],r['kinetic_order']) for r in g['rows'])
    plt.rcParams.update({
        "font.family":"sans-serif","font.sans-serif":["DejaVu Sans"],
        "font.size":7.2,"axes.titlesize":7.5,"axes.labelsize":7.5,
        "xtick.labelsize":7,"ytick.labelsize":7,
        "axes.spines.top":False,"axes.spines.right":False,
        "axes.linewidth":.6,"axes.edgecolor":"#535353",
        "xtick.major.size":2.3,"ytick.major.size":2.3,
        "xtick.minor.size":1.4,"ytick.minor.size":1.4,
        "xtick.major.width":.55,"ytick.major.width":.55,
        "pdf.fonttype":42,"ps.fonttype":42,
    })
    fig,axes=plt.subplots(2,6,figsize=(8.6,3.8),sharex=True)
    fig.subplots_adjust(left=.082,right=.992,bottom=.14,top=.79,wspace=.40,hspace=.25)
    plot_records=[]
    for col,(name,title) in enumerate(zip(CASES,TITLES)):
        values=[]
        for row_index,key in enumerate(("native_paper_aligned","native_raw")):
            ax=axes[row_index,col]
            # Draw the open zero-weight circles last so overlap with full-space
            # weighting remains visible for the KB raw-energy curve.
            for index in (2,1,0,3,4):
                rows=sorted((r for r in data["groups"][index]["rows"] if r["case"]==name),key=lambda r:r["kinetic_order"])
                x=np.array([r["kinetic_order"] for r in rows]);y=np.array([r[key]["rms_mev"] for r in rows])
                assert np.array_equal(x,[2,4,6,8,10] if index>=3 else [4,6,8,10]) and np.all(y>0)
                color,marker=STYLES[index]
                ax.plot(x,y,color=color,marker=marker,markersize=3.1 if index!=2 else 3.7,
                        markerfacecolor="white" if index in (0,2) else color,markeredgewidth=.65,
                        linewidth=1.2 if index>=3 else 1.05,linestyle=(0,(3,1.8)) if index==0 else "-",zorder=5 if index>=3 else (4 if index==0 else 2))
                values.extend(y.tolist())
                plot_records.append(dict(panel=[row_index,col],group=index,case=name,metric=key,x=x.tolist(),y=y.tolist()))
            point=points[name];star_key="aligned_rms_mev" if row_index==0 else "raw_rms_mev"
            reference_group=data["groups"][2] if name=="zrs2_gamma" else data["groups"][0]
            reference=next(r for r in reference_group["rows"] if r["case"]==name and r["kinetic_order"]==point["kinetic_order"])
            assert abs(reference[key]["rms_mev"]-point[star_key])<1e-8
            ax.scatter([point["kinetic_order"]],[point[star_key]],s=65,marker="*",facecolor="#F0CC52",
                       edgecolor="#202020",linewidth=.65,zorder=10)
            values.append(point[star_key])
            ax.set(yscale="log",xlim=(1.6,10.4),xticks=[2,4,6,8,10])
            ax.yaxis.set_major_locator(LogLocator(base=10,subs=(1.,),numticks=4))
            ax.yaxis.set_minor_locator(LogLocator(base=10,subs=(2.,5.),numticks=12))
            ax.yaxis.set_minor_formatter(NullFormatter())
            ax.grid(axis="y",which="major",color="#DADDE1",linewidth=.45,alpha=.7)
            ax.tick_params(axis="both",which="major",pad=2)
            if row_index==0:ax.set_title(title,pad=5)
        limits=(min(values)/1.24,max(values)*1.20)
        for ax in axes[:,col]:ax.set_ylim(limits)
        assert axes[0,col].get_ylim()==axes[1,col].get_ylim()
    axes[0,0].set_ylabel("Aligned\nRMS error (meV)",labelpad=4)
    axes[1,0].set_ylabel("Raw\nRMS error (meV)",labelpad=4)
    fig.supxlabel(r"Kinetic expansion order $M_{\rm kin}$",y=.025,fontsize=8)
    handles=[]
    for index,(color,marker) in enumerate(STYLES):
        handles.append(Line2D([],[],color=color,marker=marker,ms=3.2,mfc="white" if index in (0,2) else color,
                              mew=.65,lw=1.05,ls="--" if index==0 else "-",label=LABELS[index]))
    handles.append(Line2D([],[],ls="none",marker="*",ms=7.7,mfc="#F0CC52",mec="#202020",mew=.65,label="Manuscript"))
    fig.legend(handles=handles,loc='upper center',ncol=6,frameon=False,fontsize=7,
               bbox_to_anchor=(.53,.995),handlelength=1.6,handletextpad=.4,columnspacing=1.1)
    prefix=OUT/"figures/kinetic_order_convergence"
    outputs=[]
    for extension in ("png","pdf","svg"):
        file=prefix.with_suffix("."+extension)
        fig.savefig(file,dpi=330,bbox_inches="tight",pad_inches=.025);outputs.append(file)
    plt.close(fig)
    caption=(OUT/'captions.txt').read_text()
    file=prefix.with_suffix(".caption.txt");file.write_text(caption);outputs.append(file)
    report=dict(status="passed",layout=[2,6],row_order=["aligned","raw"],settings=5,
        numerical_checks=len(verified),plotted_metric_points=264,manuscript_markers=12,
        identical_y_limits_within_each_column=True,
        data_source_sha256={str(f.relative_to(OUT)):sha(f) for f in (source,source_points)},plot_records=plot_records,
        artifacts={str(f.relative_to(OUT)):dict(bytes=f.stat().st_size,sha256=sha(f)) for f in outputs})
    record=OUT/"render_verification.json";record.write_text(json.dumps(report,indent=2)+"\n")
    print("PASS: 2 rows x 6 models, 5 settings, 264 metric points and 12 manuscript markers verified")
    print("PDF size: 8.6 x 3.8 inches before tight cropping; fonts embedded as TrueType")



if __name__=="__main__":
    render_summary()
