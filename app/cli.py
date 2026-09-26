import argparse
from app.data.generators import synthetic_series
from app.models.anomaly import AnomalyDetector
from app.models.forecasting import Forecaster
def main():
    parser=argparse.ArgumentParser(prog='neurosentinel'); sub=parser.add_subparsers(dest='command',required=True); demo=sub.add_parser('demo'); demo.add_argument('--length',type=int,default=40); args=parser.parse_args()
    if args.command=='demo':
        values=synthetic_series(args.length); print({'forecast':Forecaster().predict(values,3),'anomaly':AnomalyDetector().detect(values)})
if __name__=='__main__': main()
