#!/usr/bin/env python3
"""Revision 1: exact source baseline and fixed current-orbit readability."""
from __future__ import annotations
from concurrent.futures import ProcessPoolExecutor
from fractions import Fraction as Q
import argparse
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
SRC = HERE / "SOURCE_ROOT"
CROSS = HERE / "CROSS_ROOT"
MASKS = [0, 5, 8, 13, 16, 21, 24, 29]
PREFIX = "projects/active/research_program_director/manual_dpa_intakes/DISC-009/CP-HYP-033-X21R1/"
BASE_A = "audits/"
BLOCK = [45, 105, 29]  # |140>, |410>, |104>, in that fixed order.
ZERO = (Q(0), Q(0))
ONE = (Q(1), Q(0))

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def load(path):
    return json.loads(Path(path).read_text())

def save(path, obj):
    path = Path(path)
    if path.exists():
        raise RuntimeError(f"refuse to overwrite retained artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")

def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    sys.modules[name] = obj
    spec.loader.exec_module(obj)
    return obj

def primary():
    sys.path.insert(0, str(SRC / PREFIX))
    a61 = module("readout_a61", SRC / BASE_A / "DPA_A61_SOURCE_TRANSFER_FIXED_POINT_STATE_GATE_20260915/evaluate.py")
    a62 = module("readout_a62", SRC / BASE_A / "DPA_A62_PRIMITIVE_FACTOR_STATE_AND_SOURCE_CAYLEY_GATE_20260915/evaluate.py")
    a63 = module("readout_a63", SRC / BASE_A / "DPA_A63_INTRINSIC_SPECTRAL_TRANSFER_GENERATOR_WARD_GATE_20260915/evaluate.py")
    o14 = module("readout_o14", SRC / "projects/active/discovery_partner/formal_checks/ocbfh014_source_native_refinement_naturality_check.py")
    run = module("readout_original_gram", SRC / PREFIX / "run_x21r1.py")
    return a61, a62, a63, o14, run

def check_sources():
    manifest = load(HERE / "SOURCE_MANIFEST.json")
    for item in manifest["entries"]:
        assert sha(HERE / item["copy_root"] / item["relative_path"]) == item["sha256"]
    return manifest

def canonical(mask):
    columns = [0,7,2,5,20,3,6,1,8,21,10,17,12,15,22,13,16,11,18,23,4,9,14,19,24]
    signs = np.ones(25, dtype=np.int64)
    for present, rows in [
        (mask in [5,13,21,29], [3,5,11,17]),
        (mask in [8,13,24,29], [4,14,20,22]),
        (mask in [16,21,24,29], [9,19,21,23]),
    ]:
        if present:
            signs[rows] = -1
    r = np.zeros((25,25), dtype=np.int64)
    r[np.arange(25), columns] = signs
    j = [0,3,2,1,4]
    def cr(t):
        return np.array([[t[5*b+j[d], 5*j[a]+c] for c,d in itertools.product(range(5),repeat=2)]
                         for a,b in itertools.product(range(5),repeat=2)], dtype=np.int64)
    xi = np.zeros(25, dtype=np.int64)
    xi[[0,8,12,16,24]] = 1
    p = np.outer(xi, xi)
    c = cr(r)
    h = 6*c + 3*r@c - 3*r@c@c + 3*cr(r@c) - 25*p - 6*c@p@c + 12*p@c + 12*c@p
    numerator = h@h - 10*h + 12*np.eye(25,dtype=np.int64)
    assert np.all(numerator % 12 == 0)
    f = numerator // 12
    cert = load(SRC / "audits/UB442_PROSPECTIVE_SIGNED_BRAID_TEHC_GENERALIZATION_GATE/SIGNED_BRAID_TEHC_CERTIFICATE.json")
    presentation = next(x for x in cert["presentation_records"] if x["orbit_index"] == 4 and x["rows"] == [6,2,3,8])
    archived = next(x for x in presentation["TEHC_pass_packages"] if x["quotient_mask"] == mask)
    hashes = {}
    for name, value, key in [("R",r,"source_R_sha256"),("H",h,"H_sha256"),("F",f,"F_sha256"),("P",p,"P_sha256")]:
        digest = hashlib.sha256(np.asarray(value,dtype="<i8").tobytes(order="C")).hexdigest()
        assert digest == archived[key], (mask,name,digest,archived[key])
        hashes[name] = digest
    return r,h,p,f,hashes

def spatial():
    ob11 = load(CROSS / "audits/OB011_FULL_EIGHT_SECTOR_TIME_PARITY_CLOCK_ACTION/RESULT.json")
    ob13 = load(CROSS / "audits/OB013_CANONICAL_COMMUTING_TIME_SPACE_QUOTIENT/RESULT.json")
    classes = ob13["exact_results"]["space_quotient_classes"]
    reps = [x[0] for x in classes]
    assert reps == [0,8,16,24]
    representation = []
    for cls in classes:
        matrices = []
        for source in cls:
            bits = ob11["exact_results"]["mismatch_records"][str(source)]["one_site_factor_bits"]
            d = np.diag([-1 if bit else 1 for bit in bits])
            matrices.append(np.kron(d,d).astype(np.int64))
        assert np.array_equal(matrices[0],matrices[1])
        representation.append(matrices[0])
    index = {x:i for i,x in enumerate(reps)}
    product = [[index[a^b] for b in reps] for a in reps]
    chars = sorted(v for v in itertools.product((-1,1),repeat=4)
                   if v[0] == 1 and all(v[product[i][j]] == v[i]*v[j] for i in range(4) for j in range(4)))
    nums = [sum((v[i]*representation[i] for i in range(4)), np.zeros((25,25),dtype=np.int64))
            for v in chars if v != (1,1,1,1)]
    assert [int(np.trace(n)//4) for n in nums] == [8,4,4]
    return nums

def serial(v):
    return [str(v[0]),str(v[1])]

def parse(v):
    return Q(v[0]),Q(v[1])

def qadd(a,b):
    return a[0]+b[0],a[1]+b[1]

def qmul(a,b):
    return a[0]*b[0]+17*a[1]*b[1],a[0]*b[1]+a[1]*b[0]

def qscale(a,n):
    return a[0]*n,a[1]*n

def qdiv(a,b):
    d = b[0]*b[0]-17*b[1]*b[1]
    return qmul(a,(b[0]/d,-b[1]/d))

def qsum(values):
    out = ZERO
    for v in values:
        out = qadd(out,v)
    return out

def rref(rows, cols=None):
    work = [[v if isinstance(v,tuple) else (Q(v),Q(0)) for v in row] for row in rows]
    n = cols if cols is not None else len(work[0])
    pivots = []
    row = 0
    for col in range(n):
        pivot = next((i for i in range(row,len(work)) if work[i][col] != ZERO),None)
        if pivot is None:
            continue
        work[row],work[pivot] = work[pivot],work[row]
        scale = work[row][col]
        work[row] = [qdiv(v,scale) for v in work[row]]
        for i in range(len(work)):
            if i != row and work[i][col] != ZERO:
                scale = work[i][col]
                work[i] = [qadd(v,qscale(qmul(scale,w),-1)) for v,w in zip(work[i],work[row])]
        pivots.append(col)
        row += 1
        if row == len(work):
            break
    return work,pivots

def nullspace(rows):
    work,piv = rref(rows)
    out = []
    for free in [x for x in range(len(rows[0])) if x not in piv]:
        v = [ZERO]*len(rows[0])
        v[free] = ONE
        for i,p in enumerate(piv):
            v[p] = qscale(work[i][free],-1)
        out.append(v)
    return out

def observer(mask):
    e = -1 if mask in [16,21,24,29] else 1
    z = -1 if mask in [8,13,24,29] else 1
    hs = [3*np.array([[1,e,0],[e,1,0],[0,0,0]],dtype=np.int64),
          3*np.array([[1,0,z],[0,0,0],[z,0,1]],dtype=np.int64)]
    q = np.diag([1,0,0]).astype(np.int64)
    basis,words,vectors = [],[],[]
    queue = [(np.zeros((3,3),dtype=np.int64),h@q-q@h,f"i[H{k+1},Q]") for k,h in enumerate(hs)]
    while queue:
        real,imag,word = queue.pop(0)
        flat = list(map(int,real.flat))+list(map(int,imag.flat))
        if len(rref(vectors+[flat])[1]) <= len(vectors):
            continue
        assert np.array_equal(real,real.T) and np.array_equal(imag,-imag.T)
        basis.append((real,imag)); words.append(word); vectors.append(flat)
        for k,h in enumerate(hs):
            queue.append((-(h@imag-imag@h),h@real-real@h,f"i[H{k+1},{word}]"))
        assert len(basis) <= 9
    assert len(basis) == 7
    real_rank = len(rref([list(map(int,a.flat)) for a,b in basis])[1])
    assert real_rank == 4
    return basis,words

def response(beta,basis):
    return [qsum(qscale(beta[i][j],int(real[j,i])) for i in range(3) for j in range(3))
            for real,imag in basis]

def capacity(betas,basis):
    columns = [response(b,basis) for b in betas]
    y = [list(row) for row in zip(*columns)]
    j = [[qadd(row[i],qscale(row[3],-1)) for i in range(3)] for row in y]
    return y,j,len(rref(j)[1]),nullspace(j)

def hidden_actions(kernel, am):
    hidden = []
    for v in kernel:
        d = list(v)+[qscale(qsum(v),-1)]
        action = [qsum(qmul(row[k],d[k]) for k in range(4)) for row in am]
        hidden.append({"probability_direction":[serial(x) for x in d],
                       "fingerprint_action":[serial(x) for x in action],
                       "fingerprint_sensitive":any(x != ZERO for x in action)})
    return hidden

def witness_kernels():
    # Squared overlaps of four explicitly specified pure records.
    # Positive: a,b,c,(b+c)/sqrt(2), fixed environment.
    # Negative: fixed output a, environment 0,+,1,-.
    # Comparator: a0,b0,(a0+b1)/sqrt(2),(a+b)0/sqrt(2).
    pairs = {
        "positive":{(1,3):Q(1,2),(2,3):Q(1,2)},
        "negative":{(0,1):Q(1,2),(0,3):Q(1,2),(1,2):Q(1,2),(2,3):Q(1,2)},
        "comparator":{(0,2):Q(1,2),(0,3):Q(1,2),(1,3):Q(1,2),(2,3):Q(1,4)},
    }
    out = {}
    for key,values in pairs.items():
        a = [[ZERO]*4 for _ in range(4)]
        for (i,j),v in values.items():
            a[i][j] = a[j][i] = (v,Q(0))
        out[key] = a
    return out

def witness_states():
    def density(v):
        return [[(Q(v[i])*Q(v[j]),Q(0)) for j in range(3)] for i in range(3)]
    pa,pb,pc = [density([int(k==i) for k in range(3)]) for i in range(3)]
    bc = [[(Q(int(i in [1,2] and j in [1,2]),2),Q(0)) for j in range(3)] for i in range(3)]
    ab = [[(Q(int(i in [0,1] and j in [0,1]),2),Q(0)) for j in range(3)] for i in range(3)]
    middle = [[qscale(qadd(pa[i][j],pb[i][j]),Q(1,2)) for j in range(3)] for i in range(3)]
    return {"positive":[pa,pb,pc,bc],"negative":[pa,pa,pa,pa],"comparator":[pa,pb,middle,ab]}

def witnesses():
    check_sources()
    basis,words = observer(0)
    raw = {}
    expected = {"positive":3,"negative":0,"comparator":2}
    ams = witness_kernels()
    for key,states in witness_states().items():
        y,j,rank,kernel = capacity(states,basis)
        assert rank == expected[key]
        hidden = hidden_actions(kernel,ams[key])
        readable = all(not x["fingerprint_sensitive"] for x in hidden)
        assert readable == (key == "positive")
        raw[key] = {"states":[[[serial(v) for v in row] for row in b] for b in states],
                    "Y":[[serial(v) for v in row] for row in y],"J":[[serial(v) for v in row] for row in j],
                    "A":[[serial(v) for v in row] for row in ams[key]],
                    "hidden_directions":hidden,"fingerprint_only_readable":readable,
                    "rank":rank,"expected_rank":expected[key],"status":"PASS"}
        save(HERE / "witnesses_r1" / f"{key}.json",raw[key])
    dark = [[(Q(a*b,3),Q(0)) for b in [1,-1,-1]] for a in [1,-1,-1]]
    bright = [[qscale(qadd((Q(i==j),Q(0)),qscale(dark[i][j],-1)),Q(1,2)) for j in range(3)] for i in range(3)]
    assert response(dark,basis) == response(bright,basis) == [ZERO]*7
    save(HERE / "witnesses_r1" / "observer.json",{"words":words,
         "basis":[{"real":a.tolist(),"imag":b.tolist()} for a,b in basis],
         "dark":[[serial(v) for v in row] for row in dark],
         "bright_average":[[serial(v) for v in row] for row in bright],
         "observer_rank":7,"real_sensitive_rank":4,"status":"PASS"})
    print("E0_WITNESS_CAPACITY_PASS 3 0 2")

def baseline_one(mask):
    a61,a62,a63,o14,run = primary()
    r,h,p,f,hashes = canonical(mask)
    rho_num = a61.selected_state_numerator(r,p,o14)
    sigma,_ = a61.stationary_local_numerator(rho_num)
    source = a62.raw_contraction(r)
    family,vals,ranks,poly,projs = a63.exact_projectors(source)
    weights,pair_weights,triples = a63.spectral_weights(sigma,projs)
    transition,_ = a63.exact_edge_current(weights,pair_weights,len(projs))
    clock,_,_,_ = o14.source_g1(r)
    screen = [clock]+[np.kron(n,np.eye(5,dtype=np.int64)) for n in spatial()]
    raw = {"spectral_projectors":projs,"spectral_ranks":ranks,
           "spectral_weights":weights,"pair_weights":pair_weights}
    gram = run.ordered_edge_gram_q17(a63,raw,screen)
    packet = load(SRC / PREFIX / "RAW_PACKET_OUTPUTS.json")
    expected = next(x for x in packet["carrier_records"] if x["mask"] == mask)["ordered_edge_gram"]
    assert all(gram[i][j] == (Q(expected[i][j]),Q(0)) for i in range(4) for j in range(4)),mask
    np.savez_compressed(HERE / "baseline" / f"mask_{mask}_integers.npz",
                        R=r,H=h,P=p,F=f,raw_contraction=source,sigma=sigma,screen=np.array(screen))
    record = {"mask":mask,"matrix_hashes":hashes,"family":family,"ranks":ranks,"polynomial":poly,
              "projectors":[[[serial(v) for v in row] for row in a] for a in projs],
              "weights":[serial(v) for v in weights],
              "transition":[[serial(v) for v in row] for row in transition],
              "gram":[[serial(v) for v in row] for row in gram],
              "original_gram_entries_exact":16,"source_units_order":"spectral triples first,second,third lexicographic"}
    save(HERE / "baseline" / f"mask_{mask}.json",record)
    return mask

def baseline():
    check_sources()
    e0 = load(HERE / "E0_RECORD.json")
    assert e0["overall_status"] == "PASS" and e0["producer_artifact"]["sha256"] == sha(__file__)
    (HERE / "baseline").mkdir(exist_ok=True)
    with ProcessPoolExecutor(max_workers=4) as pool:
        for mask in pool.map(baseline_one,MASKS):
            print("BASELINE_MASK_MATCH",mask,flush=True)
    files = sorted((HERE / "baseline").iterdir())
    save(HERE / "BASELINE_GATE.json",{
        "source_identity":"PASS","path_manifest":"PASS","split_manifest_status":"PASS_NOT_APPLICABLE",
        "baseline_intermediate":"PASS_EXACT_R_H_F_P_HASHES_AND_FULL_SPECTRAL_TRANSFER",
        "baseline_primary":"PASS_128_EXACT_RATIONAL_GRAM_ENTRIES",
        "intervention_invariants":"PASS_FIXED_MASKS_SCALES_AND_OBSERVER_RULE",
        "science_outcome_authorized":True,"script_sha256":sha(__file__),
        "masks":MASKS,"artifacts":{p.name:sha(p) for p in files}})
    print("BASELINE_PASS_128_GRAM_ENTRIES")

def matrix_pair(values):
    return (np.array([[v[0] for v in row] for row in values],dtype=object),
            np.array([[v[1] for v in row] for row in values],dtype=object))

def trace_pair(mat,projector):
    a,b = mat; c,d = projector
    support = np.argwhere((c != 0)|(d != 0))
    return qsum(qmul((a[j,i],b[j,i]),(c[i,j],d[i,j])) for i,j in support)

def left_integer(s,mat):
    out = [np.full(mat[0].shape,Q(0),dtype=object),np.full(mat[1].shape,Q(0),dtype=object)]
    for i in range(len(s)):
        for k in np.flatnonzero(s[i]):
            for part in range(2):
                out[part][i,:] += int(s[i,k])*mat[part][k,:]
    return tuple(out)

def focal_one(mask):
    _,_,a63,_,_ = primary()
    data = load(HERE / "baseline" / f"mask_{mask}.json")
    integer = np.load(HERE / "baseline" / f"mask_{mask}_integers.npz")
    projectors = [matrix_pair([[parse(v) for v in row] for row in a]) for a in data["projectors"]]
    triples = [a63.qmkron_pair(a63.qmkron_pair(projectors[a],projectors[b]),projectors[c])
               for a,b,c in itertools.product(range(len(projectors)),repeat=3)]
    dims = [data["ranks"][a]*data["ranks"][b]*data["ranks"][c]
            for a,b,c in itertools.product(range(len(projectors)),repeat=3)]
    pi = [qscale(parse(v),Q(1,18000)) for v in data["weights"]]
    trans = [[parse(v) for v in row] for row in data["transition"]]
    rho = [np.full((125,125),Q(0),dtype=object),np.full((125,125),Q(0),dtype=object)]
    for p,d,proj in zip(pi,dims,triples):
        coef = qscale(p,Q(1,d))
        rho[0] += coef[0]*proj[0]+17*coef[1]*proj[1]
        rho[1] += coef[0]*proj[1]+coef[1]*proj[0]
    basis,words = observer(mask)
    states = []
    for i,s in enumerate(integer["screen"]):
        avg = [qscale(trace_pair((s.astype(object),np.zeros((125,125),dtype=object)),proj),Q(1,d))
               for proj,d in zip(triples,dims)]
        input1 = [qmul(p,a) for p,a in zip(pi,avg)]
        srho = left_integer(s,tuple(rho))
        srs = left_integer(s,(srho[0].T,srho[1].T))
        srs = (srs[0].T,srs[1].T)
        input2 = [trace_pair(srs,proj) for proj in triples]
        coef1 = [qsum(qmul(trans[x][y],input1[x]) for x in range(len(triples))) for y in range(len(triples))]
        coef2 = [qsum(qmul(trans[x][y],input2[x]) for x in range(len(triples))) for y in range(len(triples))]
        def phi_entry(coefs,a,b):
            return qsum(qscale(qmul(coef,(proj[0][a,b],proj[1][a,b])),Q(1,d))
                        for coef,proj,d in zip(coefs,triples,dims))
        g = parse(data["gram"][i][i])
        comp = []
        for a in BLOCK:
            row = []
            for b in BLOCK:
                cross1 = qsum(qscale(phi_entry(coef1,int(k),b),int(s[a,k])) for k in np.flatnonzero(s[a]))
                cross2 = qsum(qscale(phi_entry(coef1,int(k),a),int(s[b,k])) for k in np.flatnonzero(s[b]))
                value = qadd((srs[0][a,b],srs[1][a,b]),qscale(qadd(cross1,cross2),-1))
                value = qadd(value,phi_entry(coef2,a,b))
                row.append(qdiv(value,g))
            comp.append(row)
        assert all(comp[a][b] == comp[b][a] for a in range(3) for b in range(3))
        states.append(comp)
    y,j,rank,kernel = capacity(states,basis)
    gram = [[parse(v) for v in row] for row in data["gram"]]
    am = [[ZERO if i==j else qdiv(qmul(gram[i][j],gram[i][j]),qmul(gram[i][i],gram[j][j]))
           for j in range(4)] for i in range(4)]
    hidden = hidden_actions(kernel,am)
    numeric = np.array([[float(v[0])+np.sqrt(17)*float(v[1]) for v in row] for row in j])
    norms = [np.sqrt(float(np.sum(a*a)+np.sum(b*b))) for a,b in basis]
    sv = np.linalg.svd(numeric/np.array(norms)[:,None],compute_uv=False)
    record = {"mask":mask,"beta_block":[[[serial(v) for v in row] for row in b] for b in states],
              "block_traces":[serial(qsum(b[i][i] for i in range(3))) for b in states],
              "compression_renormalized":False,"observer_words":words,
              "observer_basis":[{"real":a.tolist(),"imag":b.tolist()} for a,b in basis],
              "Y":[[serial(v) for v in row] for row in y],"J":[[serial(v) for v in row] for row in j],
              "affine_rank_exact":rank,"hidden_directions":hidden,
              "A_m":[[serial(v) for v in row] for row in am],
              "fingerprint_only_readable":all(not h["fingerprint_sensitive"] for h in hidden),
              "radicals_cancel_in_compression":all(v[1]==0 for b in states for row in b for v in row),
              "singular_values_supplemental":sv.tolist(),"classification":"AWAITING_INDEPENDENT_E2"}
    save(HERE / "focal" / f"mask_{mask}.json",record)
    return mask

def focal():
    check_sources()
    baseline_gate = load(HERE / "BASELINE_GATE.json")
    e0 = load(HERE / "E0_RECORD.json")
    assert baseline_gate["science_outcome_authorized"] is True
    assert baseline_gate["script_sha256"] == sha(__file__) == e0["producer_artifact"]["sha256"]
    assert e0["overall_status"] == "PASS"
    for name,digest in baseline_gate["artifacts"].items():
        assert sha(HERE / "baseline" / name) == digest
    with ProcessPoolExecutor(max_workers=4) as pool:
        for mask in pool.map(focal_one,MASKS):
            print("FOCAL_RAW_SAVED",mask,flush=True)
    save(HERE / "RESULT_REVISION.json",{"revision":"r1","script_sha256":sha(__file__),
         "plan_sha256":sha(HERE / "SCOUTING_PLAN.md"),
         "classification":"AWAITING_INDEPENDENT_E2",
         "artifacts":{p.name:sha(p) for p in sorted((HERE / "focal").iterdir())}})

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode",choices=["witnesses","baseline","focal"])
    args = parser.parse_args()
    {"witnesses":witnesses,"baseline":baseline,"focal":focal}[args.mode]()
