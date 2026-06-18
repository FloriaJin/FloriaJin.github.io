---
layout: post
title: "Few-Shot Learning Notes on Food-101"
date: 2026-06-18
description: "A learning note on few-shot classification using the Food-101 dataset."
tags: few-shot-learning machine-learning computer-vision food101
categories: research-notes
---

# Few-Shot Learning Notes on Food-101# Few-Shot Image Classification on Food101 — Learning Note

> **Type:** Project-learning note (paper/code reading & line-by-line walkthrough)
> **Source project:** [Few-Shot Image Classification on Food101 — Kaggle](https://www.kaggle.com/code/adityakotari/few-shot-image-classification-on-food101)
> **About this note:** This is my personal study note based on the tutorial above. The implementation code belongs to the original author; the structure, explanations, and the self-posed Q&A throughout reflect my own understanding while working through it.

---

## 1. What Is Few-Shot Classification?

- **Dataset:** a labeled *support set* + a *query set*.
- **Few-shot:** the support set contains very few images per label (typically fewer than 10).
- **Notation — N-way K-shot:** N different classes, K examples per class.
- **Classification method:** *metric-based*.

**The metric-based procedure:**
1. Use a CNN to project both support and query images into a feature space.
2. Classify query images by comparing them to the support images. If, in the feature space, an image is *closer* to class A than to class B, we guess it is A.

**Two core challenges:**
1. **Find a good feature space.** A CNN is essentially a function that takes an image as input and outputs a *representation* (an embedding) of that image in a given feature space.
2. **Find a good way to compare representations** in that feature space — this is the job of **Prototypical Networks**.

> **Q: What is the relationship between a feature space and a representation?**
>
> The feature space comes first. It is *not* hand-designed; it is determined by the CNN's architecture and parameters. The logic chain is:
>
> - You design/choose a CNN architecture (e.g., the last layer outputs a 512-dim vector) → this *implicitly defines* a 512-dim feature space.
> - You train the CNN (adjust its parameters) → this determines the *semantic structure* of that space, i.e., which regions mean what. How well it is trained directly determines the quality of the space.
> - You feed an image into the trained CNN → it outputs a 512-dim vector → that vector is the image's *representation* in this feature space.
>
> **CNN defines the space → training shapes the space → a representation is produced only after an image is fed in.**
>
> Analogy: the feature space is a *map*; the representation is the image's *coordinates* on that map. The CNN's job is to "locate" each image — to compute its latitude/longitude. Images of the same class should land in nearby positions; images of different classes should land far apart.

---

## 2. Code Walkthrough

### 2.1 Introducing the Prototypical Network

Initialize `PrototypicalNetworks` with a **backbone**: a ResNet18 pretrained on ImageNet, with its head chopped off and replaced by a `Flatten` layer.

> **Q: What is the relationship between a backbone and a feature extractor?**
>
> Here they are the *same thing*, just named from different angles:
> - **Feature extractor** — the *functional* name: its job is to extract features from an image and output a feature vector.
> - **Backbone** — the *architectural* name: it is the model's "spine" / main network, the core part.
>
> In this code, the backbone is a ResNet18 with its **classification head removed**. Originally, ResNet18's last layer is a classification layer (e.g., outputting probabilities over 1000 classes); here that head is chopped off and replaced by a `Flatten` layer so the network directly outputs a 512-dim feature vector. It no longer classifies — it purely extracts features. So: we use a (headless) ResNet18 as the backbone, which *is* our feature extractor, turning images into 512-dim vectors. Prototypical Networks then take these vectors to compute prototypes and compare distances.

### 2.2 Why `forward()` Takes Three Inputs

The `forward` method here does not take a single input tensor — it takes **three**.

`forward` is the model's *forward-pass* function: you feed data in, the model processes it and produces an output. In PyTorch, **every model has a `forward()` method** that defines how incoming data flows step by step to the output.

The key contrast is between a normal classifier and a Prototypical Network:

- **Normal classifier `forward`:** input one image → output a predicted class. Only one input needed.
- **Prototypical Network `forward`:** needs three inputs —
  - `support images`
  - `support labels` (tells the model which image belongs to which class)
  - `query images` (the images to be classified)

Why three? Because the classification logic is "compare distances." The model must first see the support images *and* their labels to compute each class's prototype, before it can decide which class a query image belongs to. Without the support set, it has nothing to compare against. In other words, calling this model is not as simple as passing in a single image — you must also tell it "what the references are" (support images + labels) before it can predict on the query images.

### 2.3 The Model, Line by Line

```python
class PrototypicalNetworks(nn.Module):
    def __init__(self, backbone: nn.Module):
        super(PrototypicalNetworks, self).__init__()
        self.backbone = backbone
```
Defines a Prototypical Networks model that inherits from PyTorch's **`nn.Module`**. Initialization does just one thing: store the passed-in backbone (the feature extractor). All later feature extraction relies on it.

```python
z_support = self.backbone.forward(support_images)
z_query = self.backbone.forward(query_images)
```
Feed the support images and query images separately into the backbone, each producing **feature vectors**. For example, 5 input images → 5 vectors of 512 dims each. `z_support` and `z_query` are the two sets of embeddings.

```python
n_way = len(torch.unique(support_labels))
z_proto = torch.cat(
    [
        z_support[torch.nonzero(support_labels == label)].mean(0)
        for label in range(n_way)
    ]
)
```
- `torch.unique(support_labels)` finds how many distinct classes are in the support set — e.g., 3 classes (pug, labrador, Saint-Bernard) → `n_way = 3`.
- For each class, `support_labels == label` selects the embeddings of all images in that class, and `.mean(0)` averages them to obtain that class's **prototype**.
- `torch.cat` stacks all class prototypes into a matrix. The final `z_proto` has shape `(n_way, 512)`, where each row is one class's prototype vector.

```python
dists = torch.cdist(z_query, z_proto)
scores = -dists
return scores
```
- `torch.cdist` computes the Euclidean distance from each query embedding to each prototype, giving a distance matrix. E.g., 10 queries, 3 classes → `dists` is `(10, 3)`.
- `scores = -dists`: negate the distances to turn them into scores. **Smaller distance means more similar**; after negation, a larger value means more similar — aligning with the usual "higher score is better" convention, so you can directly apply `softmax` or `argmax` afterward to pick the top-scoring class.

```python
convolutional_network = resnet18(pretrained=True)
convolutional_network.fc = nn.Flatten()
model = PrototypicalNetworks(convolutional_network).cuda()
```
- Load a ResNet18 pretrained on ImageNet.
- Replace the classification head `fc` with `Flatten`, so the network outputs a 512-dim feature vector instead of probabilities over 1000 classes.
- Use this modified ResNet18 as the backbone to create the Prototypical Networks model; `.cuda()` moves the model to the GPU.

---

## 3. Reading the Model's Printed Structure

```text
(1): BasicBlock(
        (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (avgpool): AdaptiveAvgPool2d(output_size=(1, 1))
    (fc): Flatten(start_dim=1, end_dim=-1)
  )
)
```

Reading the network structure:

- **`BasicBlock`** — the basic building unit of ResNet18: the standard flow of conv → norm → activation → conv → norm. The `512` indicates this layer processes feature maps with 512 channels.
- **`avgpool: AdaptiveAvgPool2d(output_size=(1, 1))`** — global average pooling. Regardless of the preceding feature map's size, it compresses to `(512, 1, 1)` — one number per channel. This step turns a 2D feature map into a 512-dim vector.
- **`fc: Flatten(start_dim=1, end_dim=-1)`** — this is the layer you swapped in. Originally it was a fully connected layer `Linear(512, 1000)` outputting scores over 1000 classes. Now it is `Flatten`, which merely reshapes `(512, 1, 1)` into `(512,)` without doing any computation.

**How to verify the head was successfully chopped off?** Look at the `fc` line. If it shows `Flatten` instead of `Linear(512, 1000)`, the classification head has been successfully replaced. The model now outputs a 512-dim feature vector — exactly the embedding Prototypical Networks need.

---

## 4. Building a DataLoader for Few-Shot Tasks

> **Problem:** A standard PyTorch DataLoader feeds in batches of images *without* considering labels or whether an image belongs to the support set or the query set. Given this, two key operations are required:
> - Images must be **evenly distributed across a given number of classes**.
> - The batch must be **split between support and query sets**.

A PyTorch DataLoader has three components, each with one job:

- **Sampler — "which ones to take?"** Decides which data indices go into each batch. Here a custom sampler is used: first randomly pick `n_way` classes, then take `n_shot + n_query` images from each class, ensuring each class is evenly represented.
- **Dataset — "how to take them?"** Given the indices from the Sampler, it actually loads the data: reads image files, applies preprocessing (cropping, normalization, etc.), and returns images and labels.
- **Collate Fn — "how to assemble them?"** Reorganizes the fetched data: splits it into support and query groups, re-indexes labels, and packs everything into the format the model expects.

```python
N_WAY = 5  # Number of classes in a task
N_SHOT = 5  # Number of images per class in the support set
N_QUERY = 10  # Number of images per class in the query set
N_EVALUATION_TASKS = 100

test_set.get_labels = lambda: test_set._labels

test_sampler = TaskSampler(
    test_set, n_way=N_WAY, n_shot=N_SHOT, n_query=N_QUERY, n_tasks=N_EVALUATION_TASKS
)

test_loader = DataLoader(
    test_set,
    batch_sampler=test_sampler,
    num_workers=12,
    pin_memory=True,
    collate_fn=test_sampler.episodic_collate_fn,
)
```

**Built into PyTorch:**
- `DataLoader` — PyTorch's native data loader.
- `num_workers=12` — PyTorch's built-in multi-process loading parameter.
- `pin_memory=True` — PyTorch's built-in GPU-memory optimization parameter.

**Custom:**
- `TaskSampler` — a custom sampler that samples classes and images evenly according to the few-shot task requirements. PyTorch's built-in samplers only grab randomly or sequentially; they don't understand the "pick classes first, then pick images" logic.
- `test_sampler.episodic_collate_fn` — a custom collate function attached to the `TaskSampler`. It splits a batch into support/query groups and re-indexes labels. PyTorch's default `collate_fn` only stacks data into a tensor; it doesn't know how to split.
- `test_set.get_labels = lambda: test_set._labels` — temporarily adds a method to the dataset so the `TaskSampler` can access all image labels, letting it know which images belong to which class so it can sample by class.

**Present but not really "custom":**
- `test_set` — the dataset object. Although it may be created with a library like easy-set, it follows PyTorch's `Dataset` interface, so it isn't custom logic.
- `N_WAY, N_SHOT, N_QUERY, N_EVALUATION_TASKS` — just configuration numbers: 5 classes, 5 support images each, 10 query images each, 100 evaluation tasks total.

**Takeaway:** the `DataLoader` is PyTorch's shell, but the Sampler and Collate Fn inside it are self-written — these two are what turn ordinary data loading into few-shot *task* loading.

---

## 5. Model Evaluation

### 5.1 `evaluate_on_one_task`

```python
def evaluate_on_one_task(
    support_images: torch.Tensor,
    support_labels: torch.Tensor,
    query_images: torch.Tensor,
) -> [int, int]:
```
- Function purpose: given one few-shot task's support and query, return the predictions for the query.
- Three inputs: support images, support labels, query images.
- The `: torch.Tensor` is a Python *type hint*, indicating that the argument should be a `torch.Tensor` — a multi-dimensional array, similar to a NumPy `ndarray`.

```python
y_pred = torch.max(
            model(support_images.cuda(), support_labels.cuda(), query_images.cuda())
            .detach()
            .data,
            1,
        )[1]
    return y_pred
```
- `support_images.cuda(), support_labels.cuda(), query_images.cuda()` — move all three inputs to the GPU, since the model is on the GPU and the data must be there too in order to compute.
- `model(...)` — feed into the Prototypical Networks; internally it extracts features → computes prototypes → computes distances → returns a score matrix of shape `(num_queries, num_classes)`, e.g., `(50, 5)`, where each row is one query's scores over the 5 classes.
- `.detach().data` — detach the result from the computation graph; evaluation needs no gradients, saving memory.
- `torch.max(..., 1)` — take the max along dim 1 (the class dimension), returning a tuple `(max_value, index_of_max)`.
- `[1]` — keep only the index, i.e., the predicted class id. E.g., if a query's scores are `[-1.2, -0.3, -2.5, -4.0, -1.8]`, the max is `-0.3` at index 1, so the prediction is class 1.
- Finally returns `y_pred`, a tensor containing the predicted classes for all queries.

> **Q: What is `.detach().data`?**
>
> During computation, PyTorch by default records every operation, forming a *computation graph* so that gradients can be backpropagated and parameters updated during training. But at evaluation time we only want a prediction — no parameter updates — so this graph is unnecessary. `.detach().data` tells PyTorch: I only want the numeric values, drop the computation history so it doesn't waste memory. There is already a `torch.no_grad()` outside, which serves a similar purpose (also not recording the graph), so adding `.detach().data` here is double insurance, ensuring `y_pred` is a clean, history-free pure-data tensor.

### 5.2 `evaluate`

```python
def evaluate(data_loader: DataLoader):
    total_predictions = 0
    correct_predictions = 0
    f1_scores = []
```
Purpose: iterate over all test tasks and compute overall accuracy and F1. First, initialize three counters.

```python
model.eval()
    with torch.no_grad():
```
`model.eval()` switches the model to evaluation mode, so layers like BatchNorm and Dropout use evaluation behavior rather than training behavior. `torch.no_grad()` tells PyTorch not to record the computation graph, saving memory since evaluation needs no backpropagation.

> **Q: Are `model.eval()` and `torch.no_grad()` built-in or custom?**
>
> Both are built into PyTorch.
> - `model` is an object you created, but the `.eval()` method is *inherited* from `nn.Module`. Because `PrototypicalNetworks` inherits from `nn.Module`, it automatically has `.eval()` — no need to define it yourself.
> - `torch.no_grad()` is a framework-level tool; you call `torch.no_grad()` directly, independent of your own model definition.

```python
for episode_index, (
            support_images,
            support_labels,
            query_images,
            query_labels,
            class_ids,
        ) in enumerate(data_loader):
```
Iterate over the DataLoader, taking one batch — i.e., one few-shot task — at a time. Unpack the five items mentioned earlier: support images, support labels, query images, query labels, and the true class mapping.

```python
y_pred = evaluate_on_one_task(
                support_images, support_labels, query_images,
            )
```
Call the function above to get predictions for all queries in this task.

```python
total_predictions += len(query_labels)
            correct_predictions += (y_pred == query_labels.cuda()).sum().item()
            f1_scores.append(f1_score(query_labels.cpu(), y_pred.cpu(), average='macro'))
```
Three lines, one job each:
- Accumulate the total number of predictions.
- `y_pred == query_labels.cuda()` compares predictions with true labels element-wise, producing a True/False tensor; `.sum()` counts the correct ones; `.item()` converts to a Python number, added to the correct count.
- Use sklearn's `f1_score` to compute this task's macro F1 (the average of per-class F1). Note that sklearn only handles CPU data, so `.cpu()` is needed to move the tensors back.

```python
accuracy = (correct_predictions/total_predictions)
    avg_f1 = sum(f1_scores)/len(f1_scores)
    print(
        f"Model tested on {len(data_loader)} tasks. Accuracy: {accuracy:.2%}, F1: {avg_f1}"
    )
    return accuracy
```
After the loop, compute overall accuracy (correct / total) and average F1 (mean of all tasks' F1), print the results, and return the accuracy.

---

## 6. Meta-Training (Meta-Learning)

**Definition:** Meta-learning refers to a *training strategy*, not a change of model. (We keep the same model, so the weights stay pretrained on ImageNet.) Previously we used the pretrained ResNet directly, without extensive training on few-shot tasks. Now we will train this model on a large number of few-shot tasks so it learns *how to do few-shot classification* itself.

Concretely: prepare 40,000 few-shot tasks, each one being "here are a few support images, predict which class the queries are." The model repeatedly does these tasks; when it gets one wrong, it computes a loss and updates parameters (mainly the ResNet backbone's parameters), so that the features it extracts become better suited to few-shot classification — embeddings of the same class get closer, and those of different classes get farther apart.

That is why it is called *meta-learning*: ordinary learning learns "how to recognize cats and dogs," while meta-learning learns **"how to quickly learn to recognize new things from a few samples"** — *learning to learn*.

### 6.1 Loss + Optimizer

- **Loss:** cross-entropy (predict the query-set labels based on information from the support set and the ground truth).
- **Optimizer:** Adam.
- **Custom `fit` method:**
  - Input: a classification task (support set + query set).
  - Output: the predicted labels of the query set based on the support set.

**Meta-training loop:** `fit` → loss → update model parameters.

### 6.2 Implementation

**a. Define `fit`**

```python
def fit(
    support_images: torch.Tensor,
    support_labels: torch.Tensor,
    query_images: torch.Tensor,
    query_labels: torch.Tensor,
) -> float:
    optimizer.zero_grad()
    classification_scores = model(
        support_images.cuda(), support_labels.cuda(), query_images.cuda()
    )

    loss = criterion(classification_scores, query_labels.cuda())
    loss.backward()
    optimizer.step()

    return loss.item()
```

- `optimizer.zero_grad()` — zero out gradients. PyTorch *accumulates* gradients by default, so you must clear them before each training step to prevent the previous step's gradients from affecting this one.
- `classification_scores = model(...)` — forward pass: feed the data into the model to get the queries' scores over each class. Same as in evaluation, but this time PyTorch *records* the computation graph because gradients will be computed next.
- `loss = criterion(classification_scores, query_labels.cuda())` — compute the loss by comparing the model's scores against the queries' true labels. `criterion` is typically `CrossEntropyLoss`; the further the scores are from the correct answer, the larger the loss.
- `loss.backward()` — backpropagation: trace back through the computation graph from the loss to compute each parameter's gradient, i.e., "which direction and how much to adjust each parameter to reduce the loss."
- `optimizer.step()` — update parameters: using the computed gradients, actually adjust the backbone (ResNet) parameters so feature extraction improves.
- `return loss.item()` — return this task's loss as a number; `.item()` converts from tensor to a plain Python number, convenient for logging and printing.

**Note:** `evaluate_on_one_task` only predicts and does *not* update parameters; `fit` predicts *and* updates parameters based on the result. One is like taking an exam; the other is like doing practice problems and learning from the mistakes.

**b. Run the parameter updates**

```python
log_update_frequency = 100
acc_update_frequency = 500
all_loss = []
all_acc = []
```
Initialize configuration: update the loss shown on the progress bar every 100 tasks; measure accuracy every 500 tasks (though this is commented out below). The two lists record all losses and accuracies.

```python
model.train()
```
Switch the model to training mode — the opposite of the earlier `model.eval()`. Layers like BatchNorm and Dropout will use training behavior.

```python
with tqdm(enumerate(train_loader), total=len(train_loader)) as tqdm_train:
```
`tqdm` is a progress-bar library wrapped around the DataLoader so you can see training progress, e.g., "3000/40000 tasks done."

```python
for episode_index, (
        support_images,
        support_labels,
        query_images,
        query_labels,
        _,
    ) in tqdm_train:
```
Iterate over each few-shot task and unpack the five items. The last one `_` is the true class mapping, unused during training, so it is ignored with an underscore.

```python
loss_value = fit(support_images, support_labels, query_images, query_labels)
        all_loss.append(loss_value)
```
For this task, call `fit` to do one forward pass + loss + backpropagation + parameter update, then record the returned loss value.

```python
if episode_index % log_update_frequency == 0:
            tqdm_train.set_postfix(loss=sliding_average(all_loss, log_update_frequency))
```
Every 100 tasks, display the average loss over the last 100 tasks on the right side of the progress bar. `sliding_average` computes a moving average, which is more stable than a single loss reading and shows whether the loss is trending downward overall.

```python
# if episode_index % acc_update_frequency == 0:
        #     all_acc.append(evaluate(test_loader))
        #     model.train()
```
This block is commented out. It would have paused training every 500 tasks to run a test-set evaluation, then switched back to training mode and continued. It was likely commented out to save time. Note that `evaluate` calls `model.eval()` internally, so after evaluating you must call `model.train()` again to switch back.

**On the relationship between `fit`, loss, and update:** these three are *not* parallel — they are nested. The `fit` function does three things internally: forward pass to get scores; compute the loss (`criterion(classification_scores, query_labels.cuda())`); update parameters (`loss.backward()` + `optimizer.step()`). So this training loop simply calls `fit` repeatedly, and each `fit` completes one loss computation and one parameter update.

**c. Plot the loss curve**

```python
window_width = 50
cumsum_vec = np.cumsum(np.insert(all_loss, 0, 0))
ma_vec = (cumsum_vec[window_width:] - cumsum_vec[:-window_width]) / window_width
```
These three lines compute a *moving average*. Because each task's loss fluctuates a lot, plotting the raw loss curve would be very jagged and hard to read for a trend. So a window of 50 is used to smooth it: each point is the average of itself and the previous 49 points. This uses the cumulative-sum (`cumsum`) trick to compute the moving average quickly — the result is the same as averaging window by window, just faster.

```python
plt.title("Loss vs # Training Episodes")
plt.xlabel("# Training Episodes")
plt.ylabel("Loss")
plt.plot(ma_vec, color = "green")
plt.show()
```
Plot with matplotlib. The x-axis is the training task index, the y-axis is the loss, drawn as a green curve.

```python
evaluate(test_loader)
```

**Result:**

```text
Model tested on 100 tasks. Accuracy: 89.92%, F1: 0.8966519882510688
0.8992
```
