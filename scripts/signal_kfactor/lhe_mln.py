"""m(lN) histogram and cross section of an LO MG5_aMC LHE file: lhe_mln.py <out.json> <events dir or .lhe(.gz)>
(python 3.6 compatible: it runs inside the CMSSW_10_6 environment)."""
import glob, gzip, json, math, os, re, sys

out, src = sys.argv[1], sys.argv[2]
lhe = [src] if os.path.isfile(src) else glob.glob(os.path.join(src, "unweighted_events.lhe*"))
res = {"xsec": None, "xsec_err": None, "nevents": 0, "lhe": lhe[0] if lhe else None}
if lhe:
    op = gzip.open if lhe[0].endswith(".gz") else open
    edges = [25.0 * i for i in range(401)]
    hist = [0.0] * 400
    init, procs, inev = False, [], False
    with op(lhe[0], "rt") as f:
        for line in f:
            m_int = re.search(r"Integrated weight \(pb\)\s*:\s*([0-9.eE+-]+)", line)
            if m_int:
                res["xsec"] = float(m_int.group(1))
            if "<init>" in line:
                init = True; f.readline(); continue
            if init:
                if line.lstrip().startswith("<") and "</init>" not in line:
                    continue
                if "</init>" in line:
                    init = False
                    res["xsec_err"] = math.sqrt(sum(e * e for _, e in procs))
                    if res["xsec"] is None:
                        res["xsec"] = sum(x for x, _ in procs)
                else:
                    w = line.split(); procs.append((float(w[0]), float(w[1])))
                continue
            if line.startswith("<event"):
                inev, parts, first = True, [], True
                continue
            if inev:
                if line.startswith("<"):
                    if line.startswith("</event"):
                        inev = False
                        n = [p for p in parts if abs(p[0]) in (9900012, 9900014, 9900016)]
                        l = [p for p in parts if abs(p[0]) in (11, 13)]
                        if n and l:
                            e, px, py, pz = (n[0][i] + l[0][i] for i in (4, 1, 2, 3))
                            m = math.sqrt(max(e * e - px * px - py * py - pz * pz, 0.0))
                            hist[min(int(m // 25.0), 399)] += 1
                            res["nevents"] += 1
                    continue
                w = line.split()
                if first:
                    first = False; continue
                if len(w) >= 10 and w[1] == "1":
                    parts.append((int(w[0]), float(w[6]), float(w[7]), float(w[8]), float(w[9])))
    res["mln_edges"], res["mln_counts"] = edges, hist
json.dump(res, open(out, "w"))
