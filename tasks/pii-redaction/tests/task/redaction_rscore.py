"""Independent R-Score reproduction, NOT the unpublished official implementation.

Source: arXiv:2606.18782v1, section 3.3 and algorithms 1–2.
Deliberate edge-case conventions are recorded in ASSUMPTIONS. This is an
versioned evaluation metric. Historical scores remain saved when rescoring.
"""
from collections import defaultdict
from statistics import mean
import string

PROTOCOL = 'redactionbench-paper-reproduction-v1'
METRIC = 'redactionbench_r_score_document_mean'
SOURCE = 'https://arxiv.org/html/2606.18782v1#S3.SS3'
ASSUMPTIONS = [
    'A zero-term document scores 1 (nothing required or wrongly redacted).',
    'Connectors are separate single-character contextual annotations; newline never joins entities.',
    'Delimiter matching stays within a touch-connected annotated region and uses properly nested ASCII pairs; identical quotes pair in encounter order.',
    'Pair grouping requires contextual interior spans only; matched delimiters around mandatory text may still form a closing-delimiter/whitespace bridge.',
    'Contextual component selection touches its continuous extent, including intervening effective connector markers; directly adjacent external spans do not trigger it.',
    'Bridges require one whitespace annotation after a matched closing delimiter, with all four participating spans adjacent.',
    'Only internal gaps of at least 3 characters receive the doubled whole-gap penalty; document-edge gaps receive weight 1.',
    'Report both document mean and equal-category mean; do not equate these held-out subset results with the published full-benchmark leaderboard.',
]


class Groups:
    def __init__(self,n):self.parent=list(range(n))
    def root(self,i):
        while self.parent[i]!=i:
            self.parent[i]=self.parent[self.parent[i]];i=self.parent[i]
        return i
    def join(self,a,b):self.parent[self.root(a)]=self.root(b)
    def components(self,indices):
        out=defaultdict(list)
        for i in indices:out[self.root(i)].append(i)
        return list(out.values())


def structure(text,spans):
    n=len(spans);red=Groups(n);yellow=Groups(n);markers=set();pairs=[]
    chunks=[text[s['start']:s['end']] for s in spans]
    ys={i for i,s in enumerate(spans) if s['label']=='contextual'}
    rs=set(range(n))-ys
    def adjacent(a,b):return spans[a]['end']==spans[b]['start']
    # Algorithm 1, Punct and Slash. Connector nodes themselves are markers,
    # not entity content: union the labeled spans on the two sides.
    excluded=set(string.ascii_letters+string.digits+'\\/@[]{}()<>\'"`\r\n')
    for i in range(1,n-1):
        if i not in ys or len(chunks[i])!=1 or not adjacent(i-1,i) or not adjacent(i,i+1):continue
        char=chunks[i]
        if char not in excluded:
            red.join(i-1,i+1);yellow.join(i-1,i+1);markers.add(i)
        elif char=='/' and chunks[i-1].isascii() and chunks[i-1].isdigit() and chunks[i+1].isascii() and chunks[i+1].isdigit():
            yellow.join(i-1,i+1);markers.add(i)
    # Match delimiters using only annotated delimiter characters. Gaps and
    # newlines break the current annotated region.
    opening={'(':')','[':']','{':'}','<':'>'};symmetric=set('\'"`')
    stack=[];matched=[]
    for i,chunk in enumerate(chunks):
        if (i and not adjacent(i-1,i)) or '\n' in chunk or '\r' in chunk:stack=[]
        if i not in ys or len(chunk)!=1:continue
        if chunk in opening or chunk in symmetric:
            if chunk in symmetric and stack and stack[-1][0]==chunk:
                _,start=stack.pop();matched.append((start,i))
            else:stack.append((chunk,i))
        elif chunk in opening.values():
            if stack and opening.get(stack[-1][0])==chunk:
                _,start=stack.pop();matched.append((start,i))
            else:stack=[]
    for start,end in matched:
        interior=list(range(start+1,end))
        if interior and all(i in ys for i in interior):
            pairs.append(list(range(start,end+1)));markers.update((start,end))
        # Algorithm 1, Bridge: ...a ) <yellow whitespace> b...
        d=end;w=d+1;b=d+2;a=d-1
        if a>=0 and b<n and w in ys and chunks[w].isspace() and '\n' not in chunks[w] and '\r' not in chunks[w] and adjacent(a,d) and adjacent(d,w) and adjacent(w,b):
            yellow.join(a,b);markers.update((d,w))
    red_entities=red.components(sorted(rs))
    all_components=yellow.components(range(n))
    context_components=[c for c in all_components if any(i in ys for i in c)]
    fused=Groups(n)
    for component in context_components+pairs:
        members=[i for i in component if i in ys]
        for i in members[1:]:fused.join(members[0],i)
    contextual_entities=[c for c in fused.components(sorted(ys)) if not(len(c)==1 and c[0] in markers)]
    return dict(red=red_entities,context=contextual_entities,components=context_components,pairs=pairs,markers=markers,yellow=ys)


def score_document(text,spans,predictions):
    spans=sorted(spans,key=lambda s:(s['start'],s['end']))
    previous=0
    for s in spans:
        if s.get('label') not in ('mandatory','contextual') or not 0<=previous<=s['start']<s['end']<=len(text):
            raise ValueError('Expected disjoint, nonempty Unicode reference spans')
        previous=s['end']
    mask=bytearray(len(text))
    for p in predictions:
        if type(p['start']) is not int or type(p['end']) is not int or not 0<=p['start']<p['end']<=len(text):
            raise ValueError('Invalid predicted Unicode span')
        mask[p['start']:p['end']]=b'\1'*(p['end']-p['start'])
    prefix=[0]
    for value in mask:prefix.append(prefix[-1]+value)
    def overlap(i):return prefix[spans[i]['end']]-prefix[spans[i]['start']]
    coverage=[overlap(i)/(s['end']-s['start']) for i,s in enumerate(spans)]
    groups=structure(text,spans)
    selected={i for i in groups['yellow'] if overlap(i)}
    # Algorithm 2: contextual components and selected pair delimiters can
    # activate more spans, repeatedly, before entity means are computed.
    while True:
        before=set(selected)
        for comp in groups['components']:
            lo=min(spans[i]['start'] for i in comp);hi=max(spans[i]['end'] for i in comp)
            if any(spans[i]['start']<hi and spans[i]['end']>lo for i in selected):
                selected.update(i for i in comp if i in groups['yellow'])
        for pair in groups['pairs']:
            if pair[0] in selected or pair[-1] in selected:selected.update(pair)
        if selected==before:break
    numerator=sum(mean(coverage[i] for i in entity) for entity in groups['red'])
    mandatory_denominator=len(groups['red']);context_denominator=0.;active_entities=0
    for entity in groups['context']:
        active=[i for i in entity if i in selected]
        if not active:continue
        active_entities+=1;c=mean(coverage[i] for i in active)
        if mandatory_denominator:context_denominator+=1-c
        else:numerator+=c;context_denominator+=1
    # Count connected predicted fragments outside all annotated support.
    gaps=[];start=0
    for s in spans:
        if start<s['start']:gaps.append((start,s['start']))
        start=s['end']
    if start<len(text):gaps.append((start,len(text)))
    fp_denominator=0;fp_fragments=0;double_gaps=0
    for start,end in gaps:
        count=prefix[end]-prefix[start]
        if not count:continue
        fragments=sum(bool(mask[i]) and (i==start or not mask[i-1]) for i in range(start,end))
        fp_fragments+=fragments
        double=int(start>0 and end<len(text) and end-start>=3 and count==end-start)
        double_gaps+=double;fp_denominator+=fragments+double
    denominator=mandatory_denominator+context_denominator+fp_denominator
    return dict(r_score=numerator/denominator if denominator else 1.,numerator=numerator,denominator=denominator,
        mandatory_entities=mandatory_denominator,active_contextual_entities=active_entities,
        contextual_denominator=context_denominator,false_positive_denominator=fp_denominator,
        false_positive_fragments=fp_fragments,double_penalty_gaps=double_gaps)


def score_rows(rows,predictions):
    if len(rows)!=len(predictions):raise ValueError('Wrong prediction count')
    out=[];categories=defaultdict(list)
    for row,pred in zip(rows,predictions):
        if row['id']!=pred['id']:raise ValueError('Prediction order or ID mismatch')
        result=score_document(row['text'],row['spans'],pred['spans'])
        result.update(id=row['id'],category=row['category']);out.append(result)
        categories[row['category']].append(result['r_score'])
    if not out:raise ValueError('Empty dataset')
    by_category={k:dict(samples=len(v),r_score=mean(v)) for k,v in sorted(categories.items())}
    return dict(protocol=PROTOCOL,official_implementation=False,source=SOURCE,assumptions=ASSUMPTIONS,
        samples=len(out),document_mean=mean(r['r_score'] for r in out),
        category_macro_mean=mean(v['r_score'] for v in by_category.values()),by_category=by_category,documents=out)
