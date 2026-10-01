"""
부적합 분류 추천 모델 재학습 스크립트
- 관리대장의 불량내용(S열) → 대/중/소분류(L/M/N열)를 학습해서 model.json을 만듭니다.
- GitHub Actions가 매주 자동으로 실행합니다 (.github/workflows/retrain.yml).

사용법 (직접 돌릴 때):
  pip install pandas openpyxl scikit-learn
  python tools/train_model.py <구글시트 xlsx 주소 또는 엑셀 파일> [저장할 model.json 경로]

시트 찾는 방법 (이름이 바뀌어도 동작):
  - 이름이 '수입공정 부적합 (연도)' 꼴인 시트를 모두 찾습니다. ('수입/공정', 괄호 앞 띄어쓰기 차이 허용)
  - 2026년 이후 시트는 그대로, 2025년 시트는 Rev.06 분류명으로 바꿔서 함께 학습합니다.
  - 열 위치는 제목 줄(귀책업체 / 대분류 / 중분류 / 소분류 / 불량내용 / 발생일)을 보고 찾습니다.
"""
import os, sys, re, io, json, urllib.request
import numpy as np, pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

SHEET_RE = re.compile(r'수입\s*/?\s*공정\s*부적합\s*\(?\s*(\d{4})\s*\)?')
UPPER = {'TAP': 'TAP', 'HOLE': 'HOLE', 'BURR': 'BURR', 'SPEC': 'SPEC'}
MIN_ACC = 0.45      # 최근 데이터 검증 정확도가 이보다 낮으면 model.json을 바꾸지 않음 (안전장치)

def norm(s): return re.sub(r'\s+', ' ', str(s).lower()).strip()
def cl(x):
    x = '' if x is None or (isinstance(x, float) and np.isnan(x)) else str(x).strip()
    return '' if x in ('-', 'nan', 'N/A') else x
def fix_sub(s):
    s = cl(s); u = s.replace(' ', '').upper()
    return UPPER.get(u, s)

def map25(b, m, s, v):
    """2025년(구 분류명) → Rev.06 분류명. 대응이 없으면 None(학습 제외)."""
    if b == '업체불량':
        if m == '조립불량':
            if s in ('오배선', '배선누락'): return b, '작업불량', '배선불량'
            if s == '라벨불량': return b, '제작불량', '라벨불량'
            return b, '작업불량', s
        if m == '입고불량' and s == '수량오류': return b, m, '오입고'
        if m == '외관불량' and s == '얼룩': return b, m, '오염'
        if m == '동작불량' and s in ('출력', '동작'): return b, m, '기능'
        if m == '기타불량': return None
        return b, m, s
    if b == '제조불량':
        elec = v.startswith('전장')
        if m == '외관불량': return b, ('전장작업불량' if elec else '기구작업불량'), '외관손상'
        if m == '기구불량': return b, '기구작업불량', '조립불량'
        if m == '전장불량':
            if s in ('오배선', '배선누락'): return b, '전장작업불량', '배선불량'
            if s == '라벨불량': return b, '전장작업불량', '라벨불량'
            return b, '전장작업불량', '조립불량'
        return b, m, s
    if b in ('전장설계불량', '기구설계불량') and s == '라벨오류': return b, m, '라벨불량'
    if b == '기타불량' and (not s or m in ('소음, 진동', '설계개선', '외관불량', '기타')): return None
    return b, m, s

def read_sheet(raw):
    """제목 줄을 찾아 필요한 열만 뽑기"""
    hdr = None
    for i in range(min(30, len(raw))):
        cells = [re.sub(r'\s', '', cl(x)) for x in raw.iloc[i].tolist()]
        f = lambda pat: next((j for j, c in enumerate(cells) if re.search(pat, c)), None)
        idx = dict(t=f(r'^불량내용'), b=f(r'대분류'), m=f(r'중분류'), s=f(r'소분류'), v=f(r'^귀책'), d=f(r'^발생일'))
        if None not in (idx['t'], idx['b'], idx['m'], idx['s']): hdr = (i, idx); break
    if not hdr: return None
    i, idx = hdr
    body = raw.iloc[i + 1:]
    g = lambda k: body.iloc[:, idx[k]] if idx[k] is not None else pd.Series([''] * len(body), index=body.index)
    d = pd.DataFrame({'t': g('t').map(cl), 'b': g('b').map(cl), 'm': g('m').map(cl), 's': g('s').map(fix_sub),
                      'v': g('v').map(cl), 'd': pd.to_datetime(g('d'), errors='coerce')})
    return d[(d.t != '') & (d.b != '') & (d.m != '') & (d.s != '')]

def load(src):
    data = urllib.request.urlopen(src, timeout=120).read() if src.startswith('http') else open(src, 'rb').read()
    xl = pd.ExcelFile(io.BytesIO(data))
    frames = []
    for name in xl.sheet_names:
        m = SHEET_RE.search(name)
        if not m: continue
        year = int(m.group(1))
        if year < 2025: continue
        d = read_sheet(xl.parse(name, header=None))
        if d is None or not len(d): print(f'  - {name}: 제목 줄을 찾지 못해 건너뜀'); continue
        if year == 2025:
            mp = d.apply(lambda r: map25(r.b, r.m, r.s, r.v), axis=1)
            d = d[mp.notna()].copy(); d[['b', 'm', 's']] = pd.DataFrame(mp[mp.notna()].tolist(), index=d.index)
        d['year'] = year; frames.append(d)
        print(f'  - {name}: {len(d)}건')
    if not frames: raise SystemExit('학습할 시트를 찾지 못했습니다 (이름에 "수입공정 부적합 (연도)"가 있어야 함)')
    df = pd.concat(frames, ignore_index=True)
    df['lab'] = df.b + ' > ' + df.m + ' > ' + df.s
    return df

class Trainer:
    def fit(self, tr):
        aug = pd.concat([tr, tr.assign(v='')])          # 업체를 아직 안 고른 상황도 학습
        self.v1 = TfidfVectorizer(analyzer='char_wb', ngram_range=(1, 3), sublinear_tf=True)
        self.v2 = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), sublinear_tf=True, token_pattern=r'[^\s,./()]+')
        t = aug.t.map(norm)
        A = self.v1.fit_transform(t); B = self.v2.fit_transform(t)
        self.vend = sorted(x for x in aug.v.unique() if x); self.vi = {v: i for i, v in enumerate(self.vend)}
        self.m = LogisticRegression(C=20, max_iter=5000).fit(hstack([A, B, self.vm(aug.v)]).tocsr(), aug.lab)
        self.vb = tr[tr.v != ''].groupby('v').b.agg(lambda x: x.mode()[0]).to_dict()
        return self
    def vm(self, s):
        s = list(s); r = [i for i, v in enumerate(s) if v in self.vi]
        return csr_matrix((np.ones(len(r)), (r, [self.vi[s[i]] for i in r])), shape=(len(s), len(self.vend)))
    def acc(self, te):
        X = hstack([self.v1.transform(te.t.map(norm)), self.v2.transform(te.t.map(norm)), self.vm(te.v)]).tocsr()
        Z = X @ self.m.coef_.T + self.m.intercept_; cls = np.array(self.m.classes_)
        big = np.array([c.split(' > ')[0] for c in cls])
        for i, v in enumerate(te.v):
            if v in self.vb: Z[i, big != self.vb[v]] = -1e9
        top = np.argsort(-Z, 1)[:, :3]; y = te.lab.values
        return float(np.mean(cls[top[:, 0]] == y)), float(np.mean([y[i] in cls[top[i]] for i in range(len(y))]))
    def export(self, path, n, thr=0.3):
        W, b, cls = self.m.coef_, self.m.intercept_, list(self.m.classes_)
        names = ['c:' + f for f in self.v1.get_feature_names_out()] + ['w:' + f for f in self.v2.get_feature_names_out()] + ['v:' + v for v in self.vend]
        idf = list(self.v1.idf_) + list(self.v2.idf_); feat = {}
        for j, nm in enumerate(names):
            nz = np.nonzero(np.abs(W[:, j]) >= thr)[0]
            if len(nz) == 0 and j >= len(idf): continue
            feat[nm] = [(round(float(idf[j]), 3) if j < len(idf) else 0)] + [[int(k), round(float(W[k, j]), 3)] for k in nz]
        json.dump({'version': pd.Timestamp.now(tz='Asia/Seoul').strftime('%Y-%m-%d'), 'n': int(n), 'labels': cls,
                   'bias': [round(float(x), 3) for x in b], 'feat': feat, 'vendorBig': self.vb},
                  open(path, 'w', encoding='utf-8'), ensure_ascii=False, separators=(',', ':'))

def main(src, out='model.json'):
    print('시트 읽는 중...')
    df = load(src)
    print(f'총 {len(df)}건, 분류 {df.lab.nunique()}개')
    # 검증: 최근 60일 데이터를 빼고 학습 → 최근 데이터 맞히는지 확인
    last = df.d.max()
    te = df[df.d > last - pd.Timedelta(days=60)] if pd.notna(last) else df.iloc[0:0]
    if len(te) >= 50:
        a1, a3 = Trainer().fit(df.drop(te.index)).acc(te)
        print(f'검증(최근 60일 {len(te)}건): 1순위 {a1:.1%} / 3순위 안 {a3:.1%}')
        if a1 < MIN_ACC: raise SystemExit(f'정확도가 너무 낮아({a1:.1%}) model.json을 바꾸지 않습니다. 시트 구조를 확인하세요.')
    else:
        a1 = a3 = None
    print('전체 데이터로 학습 중...')
    Trainer().fit(df).export(out, len(df))
    json.dump({'date': pd.Timestamp.now(tz='Asia/Seoul').strftime('%Y-%m-%d %H:%M'), 'rows': int(len(df)),
               'labels': int(df.lab.nunique()), 'recent_top1': a1, 'recent_top3': a3},
              open(os.path.join(os.path.dirname(os.path.abspath(out)), 'model_report.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    print(f'완료 → {out}')

if __name__ == '__main__':
    if len(sys.argv) < 2: print(__doc__); sys.exit(1)
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else 'model.json')
