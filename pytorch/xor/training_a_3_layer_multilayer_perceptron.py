import numpy as np
import torch
from torch import optim, nn

from pytorch.xor.multilayer_perceptron import MultilayerPerceptron
from pytorch.xor.utils import LABELS, get_toy_data, visualize_results, plot_intermediate_representations
import matplotlib.pyplot as plt

input_size = 2
output_size = len(set(LABELS))
num_hidden_layers = 2
hidden_size = 2

seed = 399

torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)

mlp3 = MultilayerPerceptron(input_size=input_size,
                           hidden_size=hidden_size,
                           num_hidden_layers=num_hidden_layers,
                           output_size=output_size)
print(mlp3)
batch_size = 1000

x_data_static, y_truth_static = get_toy_data(batch_size)
fig, ax = plt.subplots(1, 1, figsize=(10,5))
visualize_results(mlp3, x_data_static, y_truth_static,
                  ax=ax, title='Initial 3-Layer MLP State', levels=[0.5])

plt.axis('off')
plt.savefig('images/mlp3_initial.png');
plt.show()

losses = []
batch_size = 10000
n_batches = 10
max_epochs = 10

loss_change = 1.0
last_loss = 10.0
change_threshold = 1e-5
epoch = 0
all_imagefiles = []

lr = 0.01
optimizer = optim.Adam(params=mlp3.parameters(), lr=lr)
cross_ent_loss = nn.CrossEntropyLoss()


def early_termination(loss_change, change_threshold, epoch, max_epochs):
    terminate_for_loss_change = loss_change < change_threshold
    terminate_for_epochs = epoch > max_epochs

    # return terminate_for_loss_change or
    return terminate_for_epochs


while not early_termination(loss_change, change_threshold, epoch, max_epochs):
    for _ in range(n_batches):
        # step 0: fetch the data
        x_data, y_target = get_toy_data(batch_size)

        # step 1: zero the gradients
        mlp3.zero_grad()

        # step 2: run the forward pass
        y_pred = mlp3(x_data).squeeze()

        # step 3: compute the loss
        loss = cross_ent_loss(y_pred, y_target.long())

        # step 4: compute the backward pass
        loss.backward()

        # step 5: have the optimizer take an optimization step
        optimizer.step()

        # auxillary: bookkeeping
        loss_value = loss.item()
        losses.append(loss_value)
        loss_change = abs(last_loss - loss_value)
        last_loss = loss_value

    print("epoch: {}: loss_value: {}".format(epoch, loss_value))
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    visualize_results(mlp3, x_data_static, y_truth_static, ax=ax, epoch=epoch,
                      title=f"{loss_value:0.2f}; {loss_change:0.4f}")
    plt.axis('off')
    epoch += 1
    all_imagefiles.append(f'images/mlp3_epoch{epoch}_toylearning.png')
    plt.savefig(all_imagefiles[-1])

_, ax = plt.subplots(1,1,figsize=(10,5))
visualize_results(mlp3, x_data_static, y_truth_static, epoch=None, levels=[0.5], ax=ax)
plt.axis('off');
plt.savefig('images/mlp3_final.png')

plot_intermediate_representations(mlp3,
                                  "The 3-layer Multilayer Perceptron's Input and Intermediate Representation",
                                  figsize=(13, 3))
plt.savefig("images/mlp3_intermediate.png")
plt.savefig("images/mlp3_intermediate.pdf")