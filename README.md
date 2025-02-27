# ETH_UE

To add new dependencies use `uv add`

For logger please use loguru

## Setup

### ENV variables

in .env specify COMET_API_KEY

### Dependencies installation

before running any build scripts install uv and run 
`uv venv`

Then specify necessary names and credentials in container_setup/credentials file

```bash
chmod +x container_setup/build.sh container_setup/launch_container.sh 
container_setup/build.sh
container_setup/launch_container.sh
```


### Data acquisition

- ETH 1 min data was downloaded from kaggle(https://www.kaggle.com/datasets/prasoonkottarathil/ethereum-historical-dataset)

For this 4 datasets don't forget to remove first line with ad link from csvs.
Also cast to lowercase first letters of columns in files
(I am too lazy to implement this for now, we will eventually switch to pure blockchain data)
- 2024 minute data (https://www.cryptodatadownload.com/cdd/Binance_ETHUSDT_2024_minute.csv)
- 2023 minute data (https://www.cryptodatadownload.com/cdd/Binance_ETHUSDT_2023_minute.csv)
- 2022 minute data (https://www.cryptodatadownload.com/cdd/Binance_ETHUSDT_2022_minute.csv)
- 2021 minute data (https://www.cryptodatadownload.com/cdd/Binance_ETHUSDT_2021_minute.csv)

