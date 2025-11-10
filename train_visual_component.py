import argparse
import sys

from framework.VisualComponent import VisualComponent


# CLI command
# python train_visual_component.py
def main():
    parser = argparse.ArgumentParser(description='Visual Component Training.')
    parser.add_argument('--VRAI_train', action='store_true', default=False)
    args = parser.parse_args()

    print(args)
    print(args.VRAI_train)

    print(f"[INFO] Training the Visual Component")
    visual = VisualComponent()
    visual.train(VRAI_train=args.VRAI_train)

    return

if __name__ == "__main__":
    # Uncomoment it when the you are running the script from the terminal
    #sys.argv = ['--VRAI_train'] #To run this script in jupyter notebook using the default arguments
    main()

