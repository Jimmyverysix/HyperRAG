"""Generate a standalone Chinese manuscript and Python publication figures."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
REPO = PROJECT.parents[1]
ARTIFACTS = PROJECT / "artifacts/cross_dataset_20261003"
PAPER = PROJECT / "paper/restructured_zh"
FIGURES = PAPER / "figures"
NAMES = {"WikiTopics": "WikiTopics", "MetaQA-3hop-vanilla": "MetaQA-3hop",
         "pathquestion": "PathQuestion", "kqapro": "KQAPro-entity"}
LABELS = {"baseline": "Original", "random_mask": "Random Mask", "pcn_mask": "PCN Mask",
          "positive_relabel": "Positive Relabel", "all_shortest": "All-shortest"}
COLORS = {"baseline": "#63758A", "random_mask": "#A6AAB0", "pcn_mask": "#327B8C",
          "positive_relabel": "#B67C55", "all_shortest": "#7A6593"}


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def export(fig, name):
    FIGURES.mkdir(parents=True, exist_ok=True)
    for extension in ("svg", "pdf", "png"):
        fig.savefig(FIGURES/f"{name}.{extension}", dpi=300, facecolor="white", bbox_inches="tight")
    fig.savefig(FIGURES/f"{name}.pgf", backend="pgf", bbox_inches="tight")
    plt.close(fig)
    return (FIGURES/f"{name}.pgf").read_text(encoding="utf-8") + "\n"


def setup_fonts():
    found = subprocess.check_output(["kpsewhich", "texgyreheros-regular.otf"], text=True).strip()
    if not found:
        raise RuntimeError("existing XeLaTeX TeX Gyre Heros font is required for PGF export")
    for path in Path(found).parent.glob("texgyreheros*.otf"):
        font_manager.fontManager.addfont(str(path))
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["TeX Gyre Heros"],
        "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": .7, "svg.fonttype": "none", "pdf.fonttype": 42,
        "legend.frameon": False, "pgf.texsystem": "xelatex", "pgf.rcfonts": False,
        "pgf.preamble": r"\usepackage{fontspec}\setsansfont{TeX Gyre Heros}",
        "mathtext.fontset": "dejavusans"})


def overview():
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.2, 2.8), gridspec_kw={"width_ratios": [1.3, 1]})
    for a in (ax, bx):
        a.set(xlim=(0, 1), ylim=(0, 1));a.axis("off")
    ax.text(0, 1.04, "a  Complete paths and actual negative sampling", weight="bold", transform=ax.transAxes)
    coords = {"s": (.08, .57), "x": (.37, .84), "a": (.88, .57), "y": (.43, .43),
              "w": (.29, .10), "z": (.66, .10)}
    def edge(u, v, color, style="solid", rad=0):
        ax.add_patch(FancyArrowPatch(coords[u], coords[v], arrowstyle="-|>", mutation_scale=9,
                     shrinkA=10, shrinkB=10, color=color, linewidth=1.6,
                     linestyle=style, connectionstyle=f"arc3,rad={rad}"))
    for u, v in (("s","x"),("x","a")):
        edge(u,v,COLORS["pcn_mask"])
    edge("s","y","#858A91","dashed")
    edge("y","a",COLORS["positive_relabel"],"dashed")
    for u,v in (("s","w"),("w","z"),("z","a")):
        edge(u,v,"#B4B8BC","dotted", -.05)
    for node, (x,y) in coords.items():
        ax.add_patch(Circle((x,y),.037,facecolor="white",edgecolor="#4F5964",linewidth=1))
        ax.text(x,y,node,ha="center",va="center",fontsize=9)
    ax.text(.42,.97,"selected positive path",color=COLORS["pcn_mask"],ha="center")
    ax.text(.55,.61,"sampled PCN: y $\\to$ a\n1 + 1 + 0 = 2",ha="center",va="center",color=COLORS["positive_relabel"],fontsize=7)
    ax.text(.49,-.02,"detour: z $\\to$ a is not PCN (2 + 1 > 2)",ha="center",fontsize=7.5,color="#63686F")
    ax.text(.04,.71,"topic",fontsize=7);ax.text(.88,.71,"answer",fontsize=7)
    bx.text(0,1.04,"b  Supervision changes; inference is shared",weight="bold",transform=bx.transAxes)
    rows = [("Original", "y = 0; w = 1",COLORS["baseline"]),
            ("Random Mask", "equal-count random w = 0",COLORS["random_mask"]),
            ("PCN Mask", "PCN: y = 0; w = 0",COLORS["pcn_mask"]),
            ("Positive Relabel", "PCN: y = 1; w = 1",COLORS["positive_relabel"])]
    bx.text(.5,.94,"Same training candidates and MLP",ha="center",fontsize=7.5)
    for i,(label,operation,color) in enumerate(rows):
        y = .80-i*.15
        bx.add_patch(FancyBboxPatch((.02,y-.055),.96,.11,boxstyle="round,pad=.015",facecolor="white",edgecolor=color,linewidth=1))
        bx.text(.05,y,label,color=color,va="center",weight="bold",fontsize=7.2)
        bx.text(.98,y,operation,ha="right",va="center",fontsize=7)
    bx.text(.5,.20,"Training answers identify PCN only",ha="center",fontsize=7.5)
    bx.add_patch(FancyBboxPatch((.02,.02),.96,.12,boxstyle="round,pad=.015",facecolor="#EEF3F5",edgecolor="#63758A"))
    bx.text(.5,.08,"Inference: q + topic $\\to$ candidates $\\to$ rank",ha="center",va="center",fontsize=7)
    fig.subplots_adjust(wspace=.16,top=.85,bottom=.12)
    return export(fig,"overview")


def prevalence_data():
    wiki = load(ARTIFACTS/"wiki_prevalence_compact.json")
    rows = [{"dataset":"WikiTopics", "train_questions":sum(r["train_questions"] for r in wiki),
        "sampled_negatives_seed_sum":sum(r["sampled_negatives_seed_sum"] for r in wiki),
        "pcn_seed_sum":sum(r["pcn_seed_sum"] for r in wiki),
        "affected_questions_any_seed":sum(r["affected_questions_any_seed"] for r in wiki)}]
    rows.append(load(ARTIFACTS/"metaqa_prevalence.json"))
    for dataset in ("pathquestion","kqapro"):
        a=load(ARTIFACTS/f"{dataset}_inspection.json")
        rows.append({"dataset":dataset,"train_questions":a["eligible_train_questions"],
            "sampled_negatives_seed_sum":a["sampled_negatives_seed_sum"],"pcn_seed_sum":a["pcn_seed_sum"],
            "affected_questions_any_seed":a["affected_questions_any_seed"]})
    for row in rows:
        row["pcn_ratio"]=row["pcn_seed_sum"]/row["sampled_negatives_seed_sum"]
        row["affected_ratio_any_seed"]=row["affected_questions_any_seed"]/row["train_questions"]
    (ARTIFACTS/"prevalence_summary.json").write_text(json.dumps(rows,indent=2)+"\n",encoding="utf-8")
    return rows


def evidence(summary, prevalence):
    fig,(ax,bx)=plt.subplots(1,2,figsize=(7.2,3),gridspec_kw={"width_ratios":[.8,1.6]})
    names=[NAMES[d["dataset"]] for d in summary["datasets"]]
    ratios=[r["pcn_ratio"]*100 for r in prevalence]
    ax.barh(np.arange(4),ratios,color=COLORS["pcn_mask"],height=.52)
    ax.set(yticks=np.arange(4),yticklabels=names,xlim=(0,14),xlabel="Actual sampled PCNs (%)")
    ax.invert_yaxis()
    for i,v in enumerate(ratios):ax.text(v+.25,i,f"{v:.2f}",va="center",fontsize=7)
    ax.set_title("a  Training conflict prevalence",loc="left",weight="bold",pad=15)
    source=[]
    for j,control in enumerate(("baseline","random_mask","positive_relabel")):
        for i,d in enumerate(summary["datasets"]):
            comp=d["comparisons"].get("pcn_mask_minus_"+control)
            if not comp:continue
            value=comp["all_questions"]["reciprocal_rank"]
            x=value["difference"]*100;lo=value["ci_low"]*100;hi=value["ci_high"]*100
            y=i+(j-1)*.19
            bx.errorbar(x,y,xerr=[[x-lo],[hi-x]],fmt=("o","s","^")[j],color=COLORS[control],
                        markersize=4,capsize=2,linewidth=1,label=LABELS[control] if i==0 else None)
            source.append({"dataset":d["dataset"],"control":control,"difference_pp":x,"ci_low_pp":lo,"ci_high_pp":hi})
    bx.axvline(0,color="#7C8289",linewidth=.8,linestyle="--")
    bx.set(yticks=np.arange(4),yticklabels=names,xlabel="PCN Mask minus control: APC-MRR (pp)",ylim=(3.6,-.7))
    bx.set_title("b  Paired ranking differences",loc="left",weight="bold",pad=15)
    bx.legend(loc="upper center",bbox_to_anchor=(.48,-.20),ncol=3,fontsize=6.5,handletextpad=.2,columnspacing=.8)
    fig.subplots_adjust(wspace=.55,bottom=.24,top=.84,left=.13,right=.98)
    write_csv(FIGURES/"evidence_source_data.csv",source)
    return export(fig,"evidence")


def mechanism():
    fig,(ax,bx)=plt.subplots(1,2,figsize=(7.2,2.7))
    source=[]
    styles=[("pathquestion",COLORS["pcn_mask"],"o"),("kqapro",COLORS["all_shortest"],"s")]
    for dataset,color,marker in styles:
        d=load(ARTIFACTS/f"{dataset}_mechanisms.json")
        for panel,field in ((ax,"training_topic_pcn_ratio"),(bx,"maximum_answer_hops")):
            bins=d["bins"][field]
            if field=="maximum_answer_hops":bins=[b for b in bins if b["lower"]>0]
            x=[b["lower"] if field=="maximum_answer_hops" else i for i,b in enumerate(bins)]
            y=[b["mean_apc_delta"]*100 for b in bins]
            offset=-.045 if dataset=="pathquestion" else .045
            panel.plot(np.asarray(x)+offset,y,marker=marker,color=color,label=NAMES[dataset],linewidth=1.2,markersize=4)
            for a,b,row in zip(x,y,bins):
                dy = 6 if dataset == "pathquestion" or b < .2 else -11
                panel.annotate(str(row["question_count"]),(a+offset,b),xytext=(0,dy),
                               textcoords="offset points",ha="center",fontsize=6.5,color=color,
                               bbox={"facecolor":"white", "edgecolor":"none", "pad":.3})
                source.append({"dataset":dataset,"field":field,**row})
    ax.set(xticks=np.arange(5),xticklabels=("0–1","1–5","5–10","10–25","25–100"),
           xlabel="Training-topic PCN ratio bin (%)",ylabel="PCN Mask minus Original: APC-MRR (pp)",ylim=(-.4,5.8))
    ax.set_title("a  Actual training-topic conflicts",loc="left",weight="bold",pad=12)
    bx.set(xticks=(1,2,3),xlabel="Maximum shortest answer hops",ylabel="APC-MRR difference (pp)",ylim=(-.4,5.8))
    bx.set_title("b  Held-out answer-path length",loc="left",weight="bold",pad=12)
    for panel in (ax,bx):panel.axhline(0,color="#A2A7AD",linestyle="--",linewidth=.6)
    bx.legend(loc="upper right",fontsize=7)
    fig.subplots_adjust(wspace=.4,bottom=.22,top=.85)
    write_csv(FIGURES/"mechanism_source_data.csv",source)
    return export(fig,"mechanism")


def write_csv(path, rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)


def table(caption,label,columns,headers,rows,small=True):
    return "\n".join([r"\begin{table}[tbp]\centering",r"\caption{"+caption+r"}\label{"+label+"}",
        r"\small" if small else "",r"\begin{tabular}{"+columns+"}",r"\toprule",
        " & ".join(headers)+r" \\",r"\midrule",*[" & ".join(row)+r" \\" for row in rows],
        r"\bottomrule",r"\end{tabular}",r"\end{table}"])


def percent(value):return f"{100*value:.2f}"
def delta(value):return f"{100*value['difference']:+.3f} [{100*value['ci_low']:.3f}, {100*value['ci_high']:.3f}]"


def build_tables(summary, prevalence):
    datasets=summary["datasets"]
    rows=[]
    for d in datasets:
        for method in ("baseline","random_mask","pcn_mask","positive_relabel"):
            a=d["methods"][method]
            cells=[percent(a['all_questions'][m]) for m in ('reciprocal_rank','answer_reach_5','answer_reach_10')] if a['status']=='complete' else [r"\todo{五种子}"]*3
            rows.append([NAMES[d['dataset']],LABELS[method],*cells,percent(d['candidate_oracle']) if 'candidate_oracle' in d else r"\todo{Oracle}"])
    main=table("共享候选四策略的五种子均值（\\%）。每组 Oracle 在四策略间相同；WikiTopics 为领域等权。",'tab:main','llrrrr',
               ['数据','策略','APC-MRR','Reach@5','Reach@10','Oracle'],rows)
    rows=[]
    for d in datasets:
        for control in ('baseline','random_mask','positive_relabel'):
            c=d['comparisons'].get('pcn_mask_minus_'+control)
            rows.append([NAMES[d['dataset']],LABELS[control],delta(c['all_questions']['reciprocal_rank']) if c else r"\todo{五种子}",
                         delta(c['all_questions']['answer_reach_10']) if c else r"\todo{五种子}"])
    comparisons=table('PCN Mask 减去控制的差值及 95\\% CI（百分点）。区间条件于五种子运行，未作多重比较校正。','tab:delta','llrr',
                       ['数据','控制','APC-MRR：差值 [CI]','Reach@10：差值 [CI]'],rows)
    rows=[]
    for r in prevalence:
        rows.append([NAMES[r['dataset']],f"{r['train_questions']:,}",f"{r['sampled_negatives_seed_sum']:,}",f"{r['pcn_seed_sum']:,}",
                     percent(r['pcn_ratio']),f"{r['affected_questions_any_seed']:,}",percent(r['affected_ratio_any_seed'])])
    prev=table('实际训练 PCN。样本数对五种子累计，题数去重；受影响表示任一种子。WikiTopics 的影响比例为题级微平均。','tab:prevalence','lrrrrrr',
               ['数据','训练题','负例','PCN','比例(\\%)','受影响题','比例(\\%)'],rows)
    rows=[]
    for d in datasets:
        for method in ('baseline','random_mask','pcn_mask','positive_relabel'):
            a=d['methods'][method]
            cond=a.get('oracle_reachable_only')
            rows.append([NAMES[d['dataset']],LABELS[method],str(d.get('oracle_reachable_question_count','TODO')),
                         percent(cond['reciprocal_rank']) if cond else r"\todo{可达题}",percent(cond['answer_reach_10']) if cond else r"\todo{可达题}"])
    conditional=table('Oracle 可达条件指标（\\%），不替代主分母。WikiTopics 先逐领域求条件均值再等权平均；题数为领域总和。','tab:conditional','llrrr',
                      ['数据','策略','可达题','APC-MRR','Reach@10'],rows)
    rows=[]
    for d in datasets[2:]:
        a=d['methods'].get('all_shortest')
        for method in ('pcn_mask','positive_relabel','all_shortest'):
            b=d['methods'].get(method)
            rows.append([NAMES[d['dataset']],LABELS[method],*[percent(b['all_questions'][m]) if b else r"\todo{五种子}" for m in ('reciprocal_rank','answer_reach_5','answer_reach_10')]])
    optional=table('构造阶段的全最短路监督。仅后者改变训练正例及负例池；evaluation 候选与主实验共享。','tab:allshortest','llrrr',
                   ['数据','策略','APC-MRR','Reach@5','Reach@10'],rows)
    return {'main_table':main,'comparisons_table':comparisons,'prevalence_table':prev,'conditional_table':conditional,'all_shortest_table':optional}


def result_text(summary):
    ds={d['dataset']:d for d in summary['datasets']}
    parts=[]
    for name in ('WikiTopics','MetaQA-3hop-vanilla'):
        d=ds[name];a=d['methods']['positive_relabel'];c=d['comparisons'].get('pcn_mask_minus_positive_relabel')
        if a['status']=='complete':
            parts.append(f"{NAMES[name]} 上正例重标记的 APC-MRR 为 {percent(a['all_questions']['reciprocal_rank'])}\\%，屏蔽减重标记为 {delta(c['all_questions']['reciprocal_rank'])} 个百分点。")
        else:parts.append(r"\todo{"+NAMES[name]+" 正例重标记五种子尚未完成}")
    optional=[]
    for name in ('pathquestion','kqapro'):
        d=ds[name];a=d['methods'].get('all_shortest');c=d['comparisons'].get('pcn_mask_minus_all_shortest')
        if a:
            # Reverse the interval to state all-shortest minus mask.
            v=c['all_questions']['reciprocal_rank'];rev={'difference':-v['difference'],'ci_low':-v['ci_high'],'ci_high':-v['ci_low']}
            optional.append(f"{NAMES[name]} 上全最短路 APC-MRR 为 {percent(a['all_questions']['reciprocal_rank'])}\\%，相对 PCN Mask 为 {delta(rev)} 个百分点。")
        else:optional.append(r"\todo{"+NAMES[name]+" 全最短路五种子未完成}")
    optional.append("两个数据的这一构造对照均高于屏蔽，尤其 KQAPro 的差异较大。它增加了监督覆盖并重新生成负例，不能把改善归因于单独改标 PCN；但它明确限制了最小修正作为最佳训练方案的主张。")
    counts=load(ARTIFACTS/'all_shortest_training_counts.json')
    write_csv(PAPER/'all_shortest_training_counts.csv',counts)
    for row in counts:
        optional.append(f"{NAMES[row['dataset']]} 的相同 {row['eligible_questions']:,} 个训练问题中，正例转移从 {row['single_path_positive_count']:,} 增至 {row['all_shortest_positive_count']:,}；seed 42 实际负例从 {row['single_path_negative_count']:,} 变为 {row['all_shortest_negative_count']:,}，全最短路采样中的 PCN 为 {row['all_shortest_pcn_count']:,}。")
    domains=summary['wiki_domains'];wins=[];losses=[]
    for d in domains:
        v=d['comparisons']['pcn_mask_minus_baseline']['all_questions']['reciprocal_rank']
        if v['ci_low']>0:wins.append(d['dataset'])
        if v['ci_high']<0:losses.append(f"{d['dataset']}：{delta(v)} 个百分点")
    text=f"WikiTopics 十一领域中，{len(wins)} 个领域的 Original 对照 APC-MRR 差值区间完全在零以上，{len(losses)} 个完全在零以下，其余跨零。下降领域为 "+"；".join(losses)+"。领域宏平均改善不能掩盖这些失败，逐领域结果与 PCN 比例同时保留在源数据中。"
    return {'relabel_result_text':''.join(parts),'all_shortest_result_text':''.join(optional),'domain_result_text':text}


def main():
    summary=load(ARTIFACTS/'summary.json');prevalence=prevalence_data();setup_fonts()
    domains = {r['dataset']: r for r in load(ARTIFACTS/'wiki_prevalence_compact.json')}
    domain_rows=[]
    for result in summary['wiki_domains']:
        record=domains[result['dataset']]
        effect=result['comparisons']['pcn_mask_minus_baseline']['all_questions']['reciprocal_rank']
        domain_rows.append({'domain':result['dataset'], 'pcn_ratio':record['pcn_ratio'],
            'affected_ratio_any_seed':record['affected_ratio_any_seed'],
            **{method+'_apc_mrr':values['all_questions']['reciprocal_rank'] for method,values in result['methods'].items()},
            **{'mask_minus_original_'+k:v for k,v in effect.items()}})
    write_csv(PAPER/'wiki_domain_source_data.csv',domain_rows)
    replacements={**build_tables(summary,prevalence),**result_text(summary),
                  'overview_pgf':overview(),'evidence_pgf':evidence(summary,prevalence),'mechanism_pgf':mechanism()}
    for template,target in (('paper_template.tex',REPO/'paper_restructured_zh.tex'),
                            ('paper_template.before_polishing.tex',PAPER/'paper_restructured_zh.before_polishing.tex')):
        source=(PAPER/template).read_text(encoding='utf-8')
        for key,value in replacements.items():source=source.replace('@@'+key+'@@',value)
        if '@@' in source:raise ValueError('unfilled template marker')
        target.write_text(source,encoding='utf-8')
    print('Standalone manuscript and figures generated from real aggregate files')


if __name__=='__main__':main()
