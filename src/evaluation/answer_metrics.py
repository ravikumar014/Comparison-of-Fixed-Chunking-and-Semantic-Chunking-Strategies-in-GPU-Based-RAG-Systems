from bert_score import score


def compute_bertscore(predictions, references):
    P, R, F1 = score(
        predictions,
        references,
        lang="en",
        verbose=False
    )
    return F1.mean().item()