import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from src.evaluation.uncertainty_evaluator import PICP, ECE, CRPS
from tqdm.notebook import tqdm


def print_metrics(model, test_dataset, batch_size=None):
    tqdm_disable = False
    if batch_size is None:
        batch_size = len(test_dataset)
        tqdm_disable = True
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
    output = []
    prediction = []
    for input_batch, output_batch in tqdm(test_dataloader, disable=tqdm_disable):
        with torch.no_grad():
            prediction_batch = model.predict(input_batch)
        prediction.append(prediction_batch)
        output_batch = output_batch[:, -model.output_len:]
        output.append(output_batch)
    output = torch.cat(output)
    prediction = torch.cat(prediction)

    metrics = dict()

    picp_evaluator = PICP(model.quantile_levels)
    res = picp_evaluator.evaluate(prediction, output)
    metrics['picp'] = res.metrics['picp']
    print(f'PICP: {res.metrics['picp']:.4f}')
    
    ece_evaluator = ECE(model.quantile_levels)
    res = ece_evaluator.evaluate(prediction, output)
    metrics['ece'] = res.metrics['ece']
    print(f'ECE: {res.metrics['ece']:.4f}')
    
    crps_evaluator = CRPS(model.quantile_levels)
    res = crps_evaluator.evaluate(prediction, output)
    metrics['crps'] = res.metrics['crps']
    print(f'CRPS: {res.metrics['crps']:.4f}')

    return metrics


def plot_forecast(model, test_dataset, idx):
    input_range = range(0, model.input_len)
    output_range = range(model.input_len, model.input_len + model.output_len)
    full_range = range(0, model.input_len + model.output_len)
    
    input_seq, output_seq = test_dataset[idx]
    output_seq = output_seq[None, -model.output_len:]
    input_seq = input_seq[None, :, :]
    full_seq = torch.cat([input_seq, output_seq], dim=1)
    with torch.no_grad():
        predicted_seq = model.predict(input_seq)
    lower_bound = predicted_seq[0, :, :, 0]
    upper_bound = predicted_seq[0, :, :, -1]
    plt.plot(full_range, full_seq[0])
    alpha = model.quantile_levels[-1] - model.quantile_levels[0]
    plt.fill_between(output_range, lower_bound[:, 0], upper_bound[:, 0], alpha=0.5, label=f'forecast High confidence {alpha:.2f}')
    plt.fill_between(output_range, lower_bound[:, 1], upper_bound[:, 1], alpha=0.5, label=f'forecast Low confidence {alpha:.2f}')
    plt.legend()
    plt.title('Example of forecast')
    plt.grid()
    plt.show()
