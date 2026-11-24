"""Run the archived experiment code with an explicit external assets location."""
from pathlib import Path
import argparse,sys
import suite as s

ap=argparse.ArgumentParser();ap.add_argument('--assets-root',required=True)
ap.add_argument('--mode',choices=['train','calibrated','stability','classical','analyze'],required=True)
args,extra=ap.parse_known_args();s.OLD=Path(args.assets_root).resolve()
if not (s.OLD/'outputs/R1_controlled_experiments_v2/training_cache.pt').exists():raise SystemExit('Missing frozen original simulation cache')
sys.argv=[args.mode,*extra]
if args.mode=='train':s.main()
elif args.mode=='calibrated':
 import calibrated;calibrated.main()
elif args.mode=='stability':
 import stability;stability.main()
elif args.mode=='classical':
 import classical;classical.main()
else:
 import analyze;analyze.main()
