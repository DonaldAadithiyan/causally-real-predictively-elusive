# REFERENCE ONLY: the camera-ready Figure 2 as drawn for the paper. Values are copied from
# runs/camera_ready/E2/RESULT.md (Phase-3 marginal best R^2; oracle gap ID / shift, per seed).
# The repo version must read them from the AGG_E2_*.csv files instead of hard-coding them.
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np
r2={"ETTh1":[-0.635,-0.718,-0.581,-0.614,-0.521,-0.579,-0.596,-0.616,-0.599,-0.603],
"ETTh2":[-0.925,-0.923,-0.896,-0.878,-0.913,-0.769,-0.807,-0.943,-0.846,-0.884],
"electricity":[-0.293,-0.288,-0.327,-0.289,-0.310],
"traffic":[-1.333,-0.506,-0.848,-0.696,-0.457]}
oid={"ETTh1":[.249,.236,.251,.257,.256,.259,.254,.252,.242,.238],"ETTh2":[.300,.294,.282,.306,.300,.306,.293,.289,.303,.301],
"electricity":[.183,.173,.180,.181,.183],"traffic":[.396,.391,.390,.395,.391]}
osh={"ETTh1":[.255,.254,.252,.255,.256,.272,.258,.245,.261,.264],"ETTh2":[.250,.251,.235,.239,.247,.238,.236,.236,.241,.241],
"electricity":[.379,.364,.372,.367,.377],"traffic":[.366,.363,.373,.374,.375]}
ds=list(r2); y=np.arange(len(ds))[::-1]
plt.rcParams.update({"font.family":"serif","font.size":7,"axes.linewidth":0.6,"xtick.major.width":0.6,"ytick.major.width":0.6})
ink="#0b0b0b"; mute="#52514e"; B="#2a78d6"; O="#eb6834"
fig,ax=plt.subplots(2,1,figsize=(3.35,3.05),gridspec_kw={"hspace":0.75})
rng=np.random.default_rng(0)
for i,d in enumerate(ds):
    v=np.array(r2[d]); ax[0].scatter(v,y[i]+rng.uniform(-.12,.12,len(v)),s=14,color=B,edgecolor="white",linewidth=0.5,zorder=3)
    ax[0].plot([np.median(v)]*2,[y[i]-.28,y[i]+.28],color=ink,lw=1.2,zorder=4)
    a=np.array(oid[d])*100; b=np.array(osh[d])*100
    ax[1].scatter(a,y[i]+0.14+rng.uniform(-.06,.06,len(a)),s=14,color=B,edgecolor="white",linewidth=0.5,zorder=3,label="in-distribution" if i==0 else None)
    ax[1].scatter(b,y[i]-0.14+rng.uniform(-.06,.06,len(b)),s=14,marker="s",facecolor="white",edgecolor=O,linewidth=1.0,zorder=3,label="shifted" if i==0 else None)
ax[0].axvline(0,color=mute,lw=0.8,ls=(0,(3,2)))
ax[0].text(-0.03,y[0]+0.38,"constant\npredictor",ha="right",va="top",fontsize=6.5,color=mute)
ax[0].set_xlim(-1.45,0.12); ax[1].set_xlim(0,45)
ax[0].set_xlabel("best held-out $R^2$ (one dot per run; bar = median)")
ax[1].set_xlabel("oracle width reduction vs. constant band (%)")
[a_.set_yticks(y) for a_ in ax]; [a_.set_yticklabels(ds) for a_ in ax]; ax[1].set_ylim(-0.6,3.6)
ax[0].set_title("(a) natural predictive information",fontsize=7.5,loc="left")
ax[1].set_title("(b) available difficulty variation (upper bound)",fontsize=7.5,loc="left")
for a_ in ax:
    a_.spines[["top","right"]].set_visible(False); a_.grid(axis="x",color="#e6e5e0",lw=0.5); a_.set_axisbelow(True)
    a_.tick_params(colors=ink,length=2)
ax[1].legend(frameon=False,loc="upper left",fontsize=6.5,handletextpad=0.2,borderaxespad=0.1)
ax[0].set_ylim(-0.6,3.6)
out = "fig_dissociation_col.pdf"
fig.savefig(out, bbox_inches="tight", pad_inches=0.02)
fig.savefig(out.replace(".pdf",".png"),dpi=200,bbox_inches="tight",pad_inches=0.02)
