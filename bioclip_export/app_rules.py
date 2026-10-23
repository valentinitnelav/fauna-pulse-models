"""Python copy of the FaunaPulse app's identification rules: its defaults and its top-down answer
over a label pack (the best order, then the best family inside it, and so on; a taxon's mass is
the sum of its names' probabilities).

Used by verify_parity.py (same answer per rank for the phone file and PyTorch) and by the
calibration benchmark, so both follow one copy. Keep it in step with the app by hand:
defaults in lib/fauna_pulse/identification/identification_assets.dart, the answer in
lib/fauna_pulse/identification/track_fusion.dart.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse

RANKS = ["kingdom", "phylum", "class", "order", "family", "genus", "species"]
EVAL = {3: "order", 4: "family", 5: "genus", 6: "species"}  # rank index -> label column
APP_MIN_PX, MAX_CROPS, DROP = 48, 10, 10.0  # the app's defaults (identification_assets.dart)


class Taxonomy:
    """Integer id per (rank, key) of the pack, so a roll-up is one bincount per rank."""

    def __init__(self, labels):
        rows = len(labels)
        self.ids = np.full((7, rows), -1, np.int64)
        self.names, self.parent = [[] for _ in range(7)], [[] for _ in range(7)]
        index = [dict() for _ in range(7)]
        for r, lab in enumerate(labels):
            lab = (list(lab) + [""] * 8)[:7]
            for k in range(7):
                if not lab[k]:
                    break
                key = "|".join(lab[: k + 1])
                if key not in index[k]:
                    index[k][key] = len(self.names[k])
                    self.names[k].append(key)
                    self.parent[k].append(index[k - 1]["|".join(lab[:k])] if k else -1)
                self.ids[k, r] = index[k][key]
        self.n = [len(x) for x in self.names]
        self.parent = [np.array(p) for p in self.parent]
        # Rows unknown below some rank go to one extra bin that is never chosen.
        self.bins = [np.where(self.ids[k] >= 0, self.ids[k], self.n[k]) for k in range(7)]
        self.sink = index[0].get("none", -1)
        self.sink_rows = self.ids[0] == self.sink
        self._named = {}
        # keys x rows 0/1 matrices: the mass of every key of a rank for many crops at once
        self.At = []
        for k in range(7):
            r = np.nonzero(self.ids[k] >= 0)[0]
            self.At.append(sparse.csr_matrix((np.ones(len(r)), (self.ids[k][r], r)), shape=(self.n[k], rows)))

    def masses(self, p, k):
        return np.bincount(self.bins[k], weights=p, minlength=self.n[k] + 1)[: self.n[k]]

    def masses_batch(self, P, k):
        """Mass of every key at rank k for each row of P (crops x pack rows)."""
        return np.asarray((self.At[k] @ np.atleast_2d(P).T).T)

    def ladder_batch(self, P):
        """The app's top-down path for each row of P: key id and mass per rank (-1 / nan below
        the deepest rank reached); best child of the chosen parent, ties to the first key as
        in track_fusion.dart."""
        P = np.atleast_2d(P)
        n = len(P)
        keys, mass = np.full((n, 7), -1, np.int64), np.full((n, 7), np.nan)
        parent, alive = np.full(n, -1), np.ones(n, bool)
        for k in range(7):
            m = self.masses_batch(P, k)
            if k:
                m = np.where(self.parent[k][None, :] == parent[:, None], m, -1.0)
            j = m.argmax(1)
            mj = m[np.arange(n), j]
            alive &= mj >= 0
            keys[alive, k], mass[alive, k] = j[alive], mj[alive]
            parent = np.where(alive, j, -2)
        return keys, mass

    def ladder(self, p):
        """One crop's path as a list of (key id, mass), deepest rank last."""
        keys, mass = self.ladder_batch(p)
        return [(int(keys[0, k]), float(mass[0, k])) for k in range(7) if keys[0, k] >= 0]

    def ids_named(self, k, name):
        """Key ids at rank k whose last name is [name] (case-insensitive); species = 'genus epithet'."""
        name = name.lower()
        if (k, name) in self._named:
            return self._named[(k, name)]
        hits = []
        for j, key in enumerate(self.names[k]):
            parts = key.lower().split("|")
            label = f"{parts[5]} {parts[6]}" if k == 6 else parts[-1]
            if label == name:
                hits.append(j)
        self._named[(k, name)] = set(hits)
        return self._named[(k, name)]


def softmax_rows(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)
