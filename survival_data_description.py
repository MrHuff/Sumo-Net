from pycox_local.pycox.datasets import kkbox,support,metabric,gbsg,flchain

if __name__ == '__main__':
    df = kkbox.read_df()
    example = df.head(10)
    example.to_csv("example.csv")

