"""Recheck the finite record compiler and its complete fixed update cube."""
from pathlib import Path
import contextlib,importlib.util,io,json,argparse
import itertools
from reproduce import ROOT,dump
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
 path=ROOT/'src/verify_stinespring_environment_compiler.py';s=importlib.util.spec_from_file_location('compiler',path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
 # Only replace transport of the same hash-verified frozen source bytes.
 m.git_bytes=lambda rel:(ROOT/'src/finite_source/SOURCE_ROOT'/rel).read_bytes()
 with contextlib.redirect_stdout(io.StringIO()) as buf:m.main()
 report=json.loads(buf.getvalue());assert all(report['checks'].values())
 old=json.loads((ROOT/'reference/ENVIRONMENT_COMPILER_CERTIFICATE.json').read_text())
 assert old['source_sha256']==report['source_sha256']
 assert all(old['checks'].values())
 # Exact separation of all eight source purities is certified above using
 # rational polynomial interval bounds. XOR updates use the fixed directions.
 masks=[0,5,8,13,16,21,24,29];directions=[5,8,16]
 edges=[{'from':i,'to':i^d,'direction':d} for i in masks for d in directions]
 histories=[{'from':i,'directions':[d,e],'intermediate':[i^d,i^e],'common_endpoint':i^d^e} for i in masks for d,e in itertools.combinations(directions,2)]
 assert len(edges)==len(histories)==24
 assert all(x['from']!=x['to'] for x in edges)
 assert all(x['intermediate'][0]!=x['intermediate'][1] for x in histories)
 dump(a.output/'ENVIRONMENT_COMPILER_REPRODUCTION.json',report)
 dump(a.output/'ORDER_RECORD_REPRODUCTION.json',{'status':'PASS','directed_edges':edges,'two_order_histories':histories,'nonzero_intermediate_purity_differences':'follows from the reproduced exact all-eight separation certificate','scope':'fixed finite source and fixed ideal probability packet; not human choice or experience'})
 print('PASS record compiler and all 24 order-sensitive edges/histories')
if __name__=='__main__':main()
