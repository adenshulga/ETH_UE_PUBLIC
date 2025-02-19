# ETH_UE


price is always nominator/denominator

for e.g. price of ETH/USDC is 2600/1, ETH = 2600 USDC

general rule is that any interval is extended to closest outer ticks

correct value changes when changing the position lays in responsibility of strategy


## Setup

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

- ETH 1 min data was downloaded from kaggle(add link)


