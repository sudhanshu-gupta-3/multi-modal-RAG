import argparse

def main(pdf):
    print('Ingesting PDF:', pdf)
    print('TODO: Add ingestion logic here.')

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--pdf', required=True)
    args = p.parse_args()
    main(args.pdf)
