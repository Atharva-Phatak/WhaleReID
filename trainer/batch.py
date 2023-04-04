def minibatch_whale_embedder(step_fn, dataloader, stage, metric_name):
    avg_loss = 0
    acc_1, acc_5 = 0, 0
    for batch in dataloader:
        loss, top_k = step_fn(batch)
        # print(top_k)
        avg_loss += loss.item()
        acc_1 += top_k[f"{metric_name}_1"]
        acc_5 += top_k[f"{metric_name}_5"]
    loss = avg_loss / len(dataloader)
    acc_1 = acc_1 / len(dataloader)
    acc_5 = acc_5 / len(dataloader)

    return {f"{stage}_loss": loss, f"{stage}_{metric_name}_1": acc_1, f"{stage}_{metric_name}_5": acc_5}
