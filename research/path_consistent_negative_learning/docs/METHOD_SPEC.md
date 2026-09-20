# 路径一致弱负例重加权：方法规范

## 1. 问题设置

对问题 (q)，令 (S_q) 为主题实体集合，(A_q) 为答案实体集合。把一个 (n)-ary fact 表示为 incidence graph 中的事实节点。候选转移记为 (t=(v,e,u))：从实体 (v) 进入事实 (e)，再到实体 (u)，因此在 incidence graph 中恰好消耗两条边。

官方弱监督从一条被选中的主题—答案最短路构造正例；其余采样候选被标为负例。问题在于最短路未必唯一。未被选中不等于已经获得语义负标签。

## 2. 完整路径一致条件

令 (d_I) 表示 incidence graph 上的无权最短距离。候选转移 (t=(v,e,u)) 对问题 (q) 路径一致，当且仅当

\[
\exists(s,a)\in S_q\times A_q:\quad
d_I(s,v)+2+d_I(u,a)=d_I(s,a).
\]

其中 `+2` 对应 (v\rightarrow e\rightarrow u) 两条 incidence edge。路径一致负例集合为

\[
D_q=\{t\in N_q: t\text{ 满足上述条件}\},
\]

其中 (N_q) 是实际进入训练的采样负例，而不是完整组合候选空间。

只检查 (d_I(u,A_q)=d_I(v,A_q)-2) 不够：该局部条件不知道从主题到 (v) 的前缀是否已经绕远。正式实现必须使用完整主题—答案成对距离条件；局部条件只保留为消融。

## 3. 最小案例

设问题主题为 (s)，答案为 (a)，图中有两条等长最短路：

\[
s\rightarrow e_1\rightarrow x\rightarrow e_3\rightarrow a,
\qquad
s\rightarrow e_2\rightarrow y\rightarrow e_4\rightarrow a.
\]

若弱监督过程只选择第一条路，则 ((s,e_1,x)) 与 ((x,e_3,a)) 是正例。第二条路未被选中，其中 ((s,e_2,y)) 可能被随机采入负例；但它满足

\[
d_I(s,s)+2+d_I(y,a)=0+2+2=4=d_I(s,a).
\]

因此它属于 (D_q)。这并不证明它在语义上必然相关，只说明“确定负例”的监督强度缺乏结构依据。直接翻成正例同样越过了现有证据。

## 4. 加权目标

保留原标签 (y_i\in\{0,1\})，只改变负监督强度：

\[
w_i(\lambda)=
\begin{cases}
\lambda,&i\in D_q,\\
1,&\text{其他候选},
\end{cases}
\qquad \lambda\in[0,1].
\]

对当前 batch (B)，正式损失为

\[
\mathcal L_B(\lambda)=
\frac{\sum_{i\in B}w_i(\lambda)\,
\operatorname{BCEWithLogits}(z_i,y_i)}
{\sum_{i\in B}w_i(\lambda)}.
\]

该定义不修改标签，不增加模型参数，也不改变推理过程。

- (lambda=1)：数值上等价于原 baseline mean BCE；
- (lambda=0)：数值上等价于删除 (D_q) 后按剩余样本数归一化；
- (0<\lambda<1)：保留负向约束，但承认其置信度较低。

## 5. 对照与识别目标

Matched Random Reweighting 对每个问题从普通负例中随机选择恰好 (|D_q|) 个候选，并赋予同一 (lambda)。若路径加权优于该对照，差异不能仅由“少惩罚了一些负例”解释。

All-shortest-positive 把 (D_q) 翻成正例，只作为诊断。它检验“结构一致是否足以支持强正标签”，不是主方法。

## 6. 复杂度

对固定问题，可分别从每个主题做单源最短路，并在反向图上从答案做单源最短路；候选判定随后为常数次哈希查询。距离索引的主要代价为图遍历，候选权重生成与候选数线性相关。训练模型结构、参数量与推理开销均不变。

## 7. 解释边界

路径一致是结构属性，不自动等于语义相关，也不自动等于 false negative。正式结论只能说这类候选的确定负标签存在冲突证据，因此研究降低其监督强度；语义比例必须由独立人工审计估计。
