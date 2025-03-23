import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from src.evaluation.uncertainty_evaluator import PICP, ECE, CRPS


def print_metrics(model, test_dataset):
    test_dataloader = DataLoader(test_dataset, batch_size=len(test_dataset))
    for input_batch, ouput_batch in test_dataloader:
        break
    prediction_batch = model.predict(input_batch)
    ouput_batch = ouput_batch[:, -model.output_len:]

    picp_evaluator = PICP(model.quantile_levels)
    res = picp_evaluator.evaluate(prediction_batch, ouput_batch)
    print(f'PICP: {res.metrics['picp']:.4f}')
    
    ece_evaluator = ECE(model.quantile_levels)
    res = ece_evaluator.evaluate(prediction_batch, ouput_batch)
    print(f'ECE: {res.metrics['ece']:.4f}')
    
    crps_evaluator = CRPS(model.quantile_levels)
    res = crps_evaluator.evaluate(prediction_batch, ouput_batch)
    print(f'CRPS: {res.metrics['crps']:.4f}')


def plot_forecast(model, test_dataset, idx):
    input_range = range(0, model.input_len)
    output_range = range(model.input_len, model.input_len + model.output_len)
    full_range = range(0, model.input_len + model.output_len)
    
    input_seq, output_seq = test_dataset[idx]
    output_seq = output_seq[None, -model.output_len:]
    input_seq = input_seq[None, :, :]
    full_seq = torch.cat([input_seq, output_seq], dim=1)
    predicted_seq = model.predict(input_seq)
    lower_bound = predicted_seq[0, :, :, 0]
    upper_bound = predicted_seq[0, :, :, -1]
    plt.plot(full_range, full_seq[0])
    alpha = model.quantile_levels[-1] - model.quantile_levels[0]
    plt.fill_between(output_range, lower_bound[:, 0], upper_bound[:, 0], alpha=0.5, label=f'forecast target0 confidence {alpha:.2f}')
    plt.fill_between(output_range, lower_bound[:, 1], upper_bound[:, 1], alpha=0.5, label=f'forecast target1 confidence {alpha:.2f}')
    plt.legend()
    plt.title('Example of forecast')
    plt.grid()
    plt.show()
