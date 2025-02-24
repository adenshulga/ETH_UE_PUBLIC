# ETH_UE

To add new dependencies use `uv add`

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


### Data aqcuisition

- ETH 1 min data was downloaded from kaggle(https://www.kaggle.com/datasets/prasoonkottarathil/ethereum-historical-dataset)
- 2024 minute data (https://www.cryptodatadownload.com/cdd/Binance_ETHUSDT_2024_minute.csv)
- 2023 minute data (https://www.cryptodatadownload.com/cdd/Binance_ETHUSDT_2023_minute.csv)
- 2022 minute data (https://www.cryptodatadownload.com/cdd/Binance_ETHUSDT_2022_minute.csv)
- 2021 minute data (https://www.cryptodatadownload.com/cdd/Binance_ETHUSDT_2021_minute.csv)

