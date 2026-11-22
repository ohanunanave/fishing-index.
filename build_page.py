#!/usr/bin/env python3
"""Build a self-contained mobile HTML page from results/fishing_index.csv."""
import json
import sys

import pandas as pd

import fishing_index as fi

src = sys.argv[1] if len(sys.argv) > 1 else "results/fishing_index.csv"
out = sys.argv[2] if len(sys.argv) > 2 else "results/index.html"

d = pd.read_csv(src, index_col=0)
d.index = pd.to_datetime(d.index, utc=True).tz_convert(fi.TZ)
d.index.name = "time"

beaches, species = {}, {}
for bkey, g in d.groupby("beach", sort=False):
    g1 = g[g.species == g.species.iloc[0]]
    beaches[bkey] = dict(
        name=fi.BEACHES[bkey]["name"],
        t=[t.strftime("%Y-%m-%dT%H") for t in g1.index],
        wave=g1["wave_height"].round(2).tolist(),
        sst=g1["sea_surface_temperature"].round(1).tolist(),
        wind=g1["wind_speed_10m"].round(0).tolist(),
        onshore=g1["onshore_wind"].round(0).tolist(),
        series={},
    )
for (bkey, skey), g in d.groupby(["beach", "species"], sort=False):
    wins = fi.best_windows(g)
    beaches[bkey]["series"][skey] = dict(
        score=g["score"].round(0).astype(int).tolist(),
        lim=[fi.FACTOR_HE.get(x, x) for x in g["limiting"]],
        unsafe=g["unsafe"].astype(bool).tolist(),
        wins=[dict(s=w["start"].strftime("%Y-%m-%dT%H"), e=w["end"].strftime("%Y-%m-%dT%H"),
                   mean=round(w["mean"]), peak=round(w["peak"]),
                   lim=fi.FACTOR_HE.get(w["limiting"], w["limiting"])) for w in wins],
    )
for skey, sp in fi.SPECIES.items():
    species[skey] = dict(name=sp.name.split(" (")[0])

data = dict(updated=pd.Timestamp.now(tz=fi.TZ).strftime("%d/%m/%Y %H:%M"),
            beaches=beaches, species=species)
tpl = open("page_template.html", encoding="utf-8").read()
open(out, "w", encoding="utf-8").write(tpl.replace("/*__DATA__*/null", json.dumps(data, ensure_ascii=False)))
print("wrote", out)
