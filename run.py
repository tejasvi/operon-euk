from main import main
from model import args, train


def run():
    args.data_x, args.data_y = main()
    train(args)


if __name__ == "__main__":
    run()
