#!/usr/bin/env python3
"""Neighbouring countries for the country narrowing of the label packs: writes the tracked
country_neighbours.csv (+ .json), which build_label_pack.py --country-lists copies into a pack.

Rule (owner's choice, 2026-10-27): two countries are neighbours when they share a land border
or when they come within --max-km (100) of each other over the sea alone, by a connection that
crosses no third country (Germany and Italy, 46 km apart across Austria, are not neighbours;
Croatia and Italy are, about 20 km apart over the Gulf of Trieste). The sea part
matters for islands and straits: at 100 km Malta gets Italy (82 km) and Cyprus Turkey (71 km);
Iceland stays alone. With it Germany also gets Sweden (73 km over the Baltic).

Source: Natural Earth admin-0 map units 1:10m, v5.1.2 (public domain, "Made with Natural
Earth", https://www.naturalearthdata.com), downloaded on first use and checked by SHA-256. The
map-units layer, not the countries layer: it gives overseas regions their own ISO code (French
Guiana GF, Guadeloupe GP, Caribbean Netherlands BQ, Svalbard SJ), as GBIF does, while the
countries layer counts them with France, the Netherlands and Norway (France would then border
Brazil). Codes are ISO 3166-1 alpha-2, the codes GBIF gives its records. Units without an ISO
code in Natural Earth get the code that GBIF's reverse geocoder gave a point inside them on
2026-10-27 (UNIT_CODES; e.g. Northern Cyprus CY, Somaliland SO); overlays such as the Korean
demilitarised zone lie on top of other units and are left out (so the borders under the DMZ and
the UNDOF zone show as gaps of under 2 km, not 0). Borders follow Natural Earth's de facto view,
which differs from GBIF in a few disputed areas: Crimea (here RU, GBIF UA), the Golan (IL, GBIF
SY), the Ilemi triangle (KE, GBIF SS), the north of the Paracels (VN, GBIF CN at Woody Island).
The pairs are the same either way; a GPS-to-country lookup with these shapes is not.

Distance: in a local flat projection (longitude scaled by the cosine of the latitude where the
two shapes come closest), within about 1 % at these distances; a shared border has distance 0.
When the shortest connection crosses a third country, the shortest one over sea only is searched
from every border point of each country (sea_gap_km): one straight line from a border point to
the nearest point of the other country, so the km can be a little longer than the shortest sea
route, and an island whose coast towards the other country belongs to a third country misses
that neighbour (Anguilla and Sint Maarten: between them lies French Saint-Martin). Natural Earth 1:10m also
lacks some small islands: Greece and Italy come out at 102.6 km (Corfu), because the Diapontia
islands (about 80 km from Italy) are missing. Both pairs are added by hand (MANUAL_PAIRS, owner's
choice 2026-10-28; listed with their reason in the .json, km as measured on Natural Earth).

Needs shapely (requirements.txt). A few seconds.
    python country_neighbours.py            # writes country_neighbours.csv and .json here
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import urllib.request
from datetime import date
from pathlib import Path

import numpy as np
import shapely
from shapely import STRtree, affinity, clip_by_rect
from shapely.geometry import LineString, shape
from shapely.ops import nearest_points, substring, unary_union

NE_URL = ("https://raw.githubusercontent.com/nvkelso/natural-earth-vector/v5.1.2/geojson/"
          "ne_10m_admin_0_map_units.geojson")
NE_SHA256 = "57da82be755f4afccd8f3b14251bb2752f5df1395f47d2d86f817470c4a48862"
KM_PER_DEGREE = 6371.0088 * math.pi / 180  # mean Earth radius
# Natural Earth GEOUNIT -> the ISO code GBIF's reverse geocoder (api.gbif.org/v1/geocode/reverse,
# "Political" answer) gave a point inside the unit on 2026-10-27; these units have none in Natural
# Earth or a code GBIF does not use for them.
UNIT_CODES = {
    "Northern Cyprus": "CY", "Cyprus No Mans Area": "CY", "Akrotiri Sovereign Base Area": "CY",
    "Dhekelia Sovereign Base Area": "CY", "Somaliland": "SO", "US Naval Base Guantanamo Bay": "CU",
    "Siachen Glacier": "IN", "Southern Patagonian Ice Field": "AR", "Bir Tawil": "EG",
    "Paracel Islands": "VN", "Spratly Islands": "PH", "Scarborough Reef": "PH", "Serranilla Bank": "CO",
    "Bajo Nuevo Bank (Petrel Is.)": "CO", "Jan Mayen": "SJ",
}


# Pairs the rule means but the data or the straight-line search cannot find (owner, 2026-10-28).
MANUAL_PAIRS = {
    ("AI", "SX"): "neighbouring islands over the sea; every straight connection from Anguilla to Sint Maarten "
                  "crosses French Saint-Martin (MF), which lies between them on the same island",
    ("GR", "IT"): "Natural Earth 1:10m lacks the Diapontia islands (Othonoi, Mathraki, Ereikoussa), about 80 km "
                  "from Italy; measured from Corfu instead: just over 100 km",
}


def unit_code(p: dict) -> str | None:
    if p["TYPE"] == "Overlay":
        return None
    if p["GEOUNIT"] in UNIT_CODES:
        return UNIT_CODES[p["GEOUNIT"]]
    if p["ADMIN"] == "United States Minor Outlying Islands":
        return "UM"  # Natural Earth codes them US; GBIF and ISO 3166 use UM
    code = p["ISO_A2_EH"] if p["ISO_A2_EH"] != "-99" else p["ISO_A2"]
    return None if code == "-99" else code


def load_countries(path: Path) -> dict:
    """ISO code -> one (multi)polygon from all Natural Earth units with that code."""
    parts: dict[str, list] = {}
    skipped = []
    for f in json.loads(path.read_text(encoding="utf-8"))["features"]:
        code = unit_code(f["properties"])
        if code is None:
            skipped.append(f["properties"]["GEOUNIT"])
            continue
        parts.setdefault(code, []).append(shape(f["geometry"]))
    print(f"{len(parts)} countries; units left out: {', '.join(sorted(skipped))}")
    return {c: unary_union(g) for c, g in parts.items()}


def near_parts(a, b, max_km: float):
    """The parts of shapes a and b that can lie within max_km of each other, or None. Clipping
    keeps large countries cheap."""
    # Margins that no point within max_km can lie beyond: a degree of latitude is always 111 km,
    # a degree of longitude is shortest at the highest latitude of the two shapes.
    lat = max(abs(v) for v in (a.bounds[1], a.bounds[3], b.bounds[1], b.bounds[3]))
    dy = max_km / KM_PER_DEGREE
    dx = dy / max(math.cos(math.radians(min(lat, 89.0))), 0.02)
    ax0, ay0, ax1, ay1 = a.bounds
    bx0, by0, bx1, by1 = b.bounds
    if bx0 > ax1 + dx or ax0 > bx1 + dx or by0 > ay1 + dy or ay0 > by1 + dy:
        return None
    na = clip_by_rect(a, bx0 - dx, by0 - dy, bx1 + dx, by1 + dy)
    nb = clip_by_rect(b, ax0 - dx, ay0 - dy, ax1 + dx, ay1 + dy)
    return None if na.is_empty or nb.is_empty else (na, nb)


def distance_km(na, nb) -> tuple[float, LineString | None, float]:
    """Shortest distance in km between na and nb, the shortest connection (None for a shared
    border) and the longitude scale used (cosine of the latitude where they come closest)."""
    if na.intersects(nb):
        return 0.0, None, 1.0
    pa, pb = nearest_points(na, nb)
    scale = math.cos(math.radians((pa.y + pb.y) / 2))
    sa = affinity.scale(na, xfact=scale, yfact=1.0, origin=(0, 0))
    sb = affinity.scale(nb, xfact=scale, yfact=1.0, origin=(0, 0))
    return sa.distance(sb) * KM_PER_DEGREE, LineString([pa, pb]), scale


def sea_gap_km(na, nb, blockers, scale: float, max_km: float) -> float | None:
    """Length in km of the shortest connection of at most max_km between na and nb that touches no
    third country (blockers, prepared shapes), or None. Searched from every border point of
    each country to the nearest point of the other (Natural Earth 1:10m has a point every few
    hundred metres), so it can be slightly longer than the true shortest crossing."""
    best = None
    for src, dst in ((na, nb), (nb, na)):
        xy = shapely.get_coordinates(src.boundary)
        sdst = affinity.scale(dst, xfact=scale, yfact=1.0, origin=(0, 0))
        spts = shapely.points(xy * [scale, 1.0])
        km = shapely.distance(spts, sdst) * KM_PER_DEGREE
        for i in np.argsort(km):
            if km[i] > max_km or (best is not None and km[i] >= best):
                break
            q = nearest_points(sdst, spts[i])[0]
            line = LineString([xy[i], (q.x / scale, q.y)])
            # The ends lie on the two countries; a third one may touch them only at a border corner.
            inner = substring(line, 0.001, 0.999, normalized=True)
            if not any(t.intersects(inner) for t in blockers):
                best = float(km[i])
                break
    return best


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-km", type=float, default=100.0, help="largest sea gap between neighbours (default 100)")
    ap.add_argument("--ne-dir", type=Path, default=Path("out"), help="where the Natural Earth file is kept")
    ap.add_argument("--out", type=Path, default=Path(__file__).with_name("country_neighbours.csv"))
    args = ap.parse_args()

    path = args.ne_dir / Path(NE_URL).name
    if not path.exists():
        print(f"Downloading {path.name} (about 14 MB) to {args.ne_dir} ...")
        args.ne_dir.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(NE_URL, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != NE_SHA256:
        raise SystemExit(f"{path}: SHA-256 {digest} is not the one of Natural Earth v5.1.2; delete it to download again")
    countries = load_countries(path)

    codes = sorted(countries)
    pairs = []
    # Antimeridian: Natural Earth splits shapes at 180 degrees, so a pair facing each other across
    # it is also measured with one side shifted by 360 degrees (none of today's pairs needs it).
    shifted = {c: (affinity.translate(g, xoff=360), affinity.translate(g, xoff=-360))
               for c, g in countries.items() if g.bounds[0] < -170 or g.bounds[2] > 170}
    entries = list(countries.items()) + [(c, g) for c, gs in shifted.items() for g in gs]
    tree = STRtree([g for _, g in entries])
    shapely.prepare([g for _, g in entries])  # fast intersects in sea_gap_km
    for i, a in enumerate(codes):
        for b in codes[i + 1:]:
            best = None
            for other in (countries[b], *shifted.get(b, ())):
                near = near_parts(countries[a], other, args.max_km)
                if near is None:
                    continue
                km, line, scale = distance_km(*near)
                if km > args.max_km:
                    continue
                # A sea gap must not run over a third country: Germany and Italy, 46 km apart across
                # Austria, are not neighbours. When the shortest connection does, the shortest one
                # over sea only counts (Croatia and Italy: 12 km across Slovenia, but about 20 km
                # over the Gulf of Trieste).
                if line is not None:
                    region = shapely.box(*unary_union(near).bounds)
                    third = [entries[j][1] for j in tree.query(region) if entries[j][0] not in (a, b)]
                    if any(line.relate_pattern(t, "T********") for t in third):
                        km = sea_gap_km(*near, third, scale, args.max_km)
                        if km is None:
                            continue
                best = km if best is None else min(best, km)
            if best is not None:
                pairs.append((a, b, round(best, 1)))
    found = {(a, b) for a, b, _ in pairs}
    for a, b in sorted(MANUAL_PAIRS):
        if (a, b) not in found:
            near = near_parts(countries[a], countries[b], max(2 * args.max_km, 250))
            pairs.append((a, b, round(distance_km(*near)[0], 1)))
    pairs.sort()
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["country", "neighbour", "km"])  # one row per pair, country < neighbour; km 0 = land border
        w.writerows(pairs)
    meta = {"rule": f"shared land border, or at most {args.max_km:g} km apart over the sea by a connection "
                    "that crosses no third country; added by hand: "
                    + ", ".join(f"{a}-{b}" for a, b in sorted(MANUAL_PAIRS)), "max_km": args.max_km,
            "manual_pairs": {f"{a}-{b}": why for (a, b), why in sorted(MANUAL_PAIRS.items())},
            "source": {"data": "Natural Earth admin-0 map units 1:10m v5.1.2 (public domain)", "url": NE_URL,
                       "sha256": NE_SHA256},
            "unit_codes": "ISO 3166-1 alpha-2; units without one: the code of GBIF's reverse geocoder (2026-10-27)",
            "countries": len(codes), "pairs": len(pairs), "date": date.today().isoformat()}
    args.out.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
    land = sum(1 for p in pairs if p[2] == 0)
    print(f"wrote {args.out}: {len(pairs)} pairs ({land} land borders, {len(pairs) - land} across the sea)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
