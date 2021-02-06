import six
import math
from typing import Dict, List


class Evaluate:
    def __init__(
        self, recs: Dict[int, List[int]], gt: Dict[int, List[int]], topn: int = 100
    ):
        self.recs = recs
        self.gt = gt
        self.topn = topn

    def _ndcg(self) -> float:
        Q, S = 0.0, 0.0
        for u, seen in six.iteritems(self.gt):
            seen = list(set(seen))
            rec = self.recs.get(u, [])
            if not rec or len(seen) == 0:
                continue

            dcg = 0.0
            idcg = sum(
                [1.0 / math.log(i + 2, 2) for i in range(min(len(seen), len(rec)))]
            )
            for i, r in enumerate(rec):
                if r not in seen:
                    continue
                rank = i + 1
                dcg += 1.0 / math.log(rank + 1, 2)
            ndcg = dcg / idcg
            S += ndcg
            Q += 1
        return S / Q

    def _map(self) -> float:
        n, ap = 0.0, 0.0
        for u, seen in six.iteritems(self.gt):
            seen = list(set(seen))
            rec = self.recs.get(u, [])
            if not rec or len(seen) == 0:
                continue

            _ap, correct = 0.0, 0.0
            for i, r in enumerate(rec):
                if r in seen:
                    correct += 1
                    _ap += correct / (i + 1.0)
            _ap /= min(len(seen), len(rec))
            ap += _ap
            n += 1.0
        return ap / n

    def _entropy_diversity(self) -> float:
        sz = float(len(self.recs)) * self.topn
        freq = {}
        for u, rec in six.iteritems(self.recs):
            for r in rec:
                freq[r] = freq.get(r, 0) + 1
        ent = -sum([v / sz * math.log(v / sz) for v in six.itervalues(freq)])
        return ent

    def _evaluate(self):
        print(f"MAP@{self.topn}: {self._map()}")
        print(f"NDCG@{self.topn}: {self._ndcg()}")
        print(f"EntDiv@{self.topn}: {self._entropy_diversity()}")
