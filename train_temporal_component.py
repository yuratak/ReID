import argparse
import sys

from framework.TemporalComponent import TemporalComponent


# CLI command
# python train_temporal_component.py
def main():
    parser = argparse.ArgumentParser(description='Temporal Component Training.')
    
    args = parser.parse_args()

    print(f"[INFO] Training the Temporal Component")
    temporal = TemporalComponent(train=True)

    return


if __name__ == "__main__":
    #sys.argv = [''] #To run this script in jupyter notebook using the default arguments
    main()

