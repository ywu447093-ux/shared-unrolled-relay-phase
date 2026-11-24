"""Regenerate finite evidence-study tasks into a separate output directory."""
import argparse,sys,json
from pathlib import Path,PureWindowsPath
import torch
import suite as s
import reviewer_validation as v

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--mode',required=True,choices=['prepare-corrective','train-corrective','train-fouriergs','train-no-diffraction','balanced-evaluation'])
    ap.add_argument('--assets-root',required=True,type=Path,help='Original frozen-development asset project')
    ap.add_argument('--raw-qpm-root',required=True,type=Path,help='Directory containing A2058/G361/HOB/PC3/PNT1A')
    ap.add_argument('--run-dir',required=True,type=Path,help='Separate empty directory for this reproduction task')
    args=ap.parse_args();root=args.run_dir.resolve()
    if root.exists() and any(root.iterdir()):raise ValueError('Use a new empty --run-dir; archived evidence is read-only.')
    root.mkdir(parents=True,exist_ok=True);s.OLD=args.assets_root.resolve();raw=args.raw_qpm_root.resolve()
    for name in ['A2058','G361','HOB','PC3','PNT1A']:assert (raw/name).is_dir(),raw/name
    original_read=v.gen._read_qpm
    def mapped(path):
        path=PureWindowsPath(str(path))
        return raw/path.parent.name/path.name
    def read(path):return original_read(str(mapped(path)))
    v.gen._read_qpm=read
    original_hash=v.source_hash;v.source_hash=lambda path:original_hash(mapped(path))
    split=json.loads((s.OLD/'outputs/R1_controlled_experiments_v2/source_split.json').read_text(encoding='utf8'))
    for records in split.values():
        for r in {r['content_id']:r for r in records}.values():assert original_hash(mapped(r['source_qpm']))==r['content_id'],r['source_qpm']
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    import calibrated;calibrated.register()
    if args.mode in ['prepare-corrective','train-corrective']:
        import reviewer_training as task;task.OUT=root;task.prepare()
        if args.mode=='train-corrective':
            train=torch.load(root/'train_cache.pt',weights_only=False);val=torch.load(root/'validation_cache.pt',weights_only=False)
            for arm in task.ARMS:task.train_arm(arm,train,val)
            del train,val
            for arm in task.ARMS:task.evaluate_arm(arm)
    elif args.mode=='balanced-evaluation':
        import balanced_pairing as task;task.OUT=root;sys.argv=[str(s.HERE/'balanced_pairing.py')];task.main()
    elif args.mode=='train-fouriergs':
        import fouriergs_train as task;task.OUT=root;sys.argv=[str(s.HERE/'fouriergs_train.py')];task.main()
    else:
        import no_diffraction_train as task;task.OUT=root;sys.argv=[str(s.HERE/'no_diffraction_train.py')];task.main()
    print('Reproduction task completed in',root,flush=True)

if __name__=='__main__':main()
