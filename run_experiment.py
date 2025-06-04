import argparse
import os

#python run_experiment.py --experiment 'simple' --weaklabels 
#python run_experiment.py --experiment 'simple' --no-weaklabels

parser = argparse.ArgumentParser(description='')
parser.add_argument('--experiment', type=str)
parser.add_argument('--weaklabels', dest='weaklabels', action='store_true')
parser.add_argument('--no-weaklabels', dest='weaklabels', action='store_false')
parser.set_defaults(weaklabels=True)
args = parser.parse_args()
experiment = args.experiment
weaklabels = args.weaklabels

assert experiment in ['persona', 'simple', 'bias']


if __name__ == "__main__":
    
    if weaklabels:
        os.system(f"python weak_labels_generation.py --experiment '{experiment}'")

    os.system(f"python get_intent_and_improve.py --experiment '{experiment}'")
    os.system(f"python fine_tune.py --experiment '{experiment}'")
    os.system(f"python get_test_answers.py --experiment '{experiment}'")
    os.system(f"python eval.py --experiment '{experiment}' --eval 'test'")

    #if experiment in ['persona', 'simple']:
    #    os.system(f"python eval.py --experiment '{experiment}' --eval 'train'")