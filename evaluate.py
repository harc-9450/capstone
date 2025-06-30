from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

def evaluate_answer(predicted, reference):
    smooth = SmoothingFunction().method1
    bleu = sentence_bleu([reference.split()], predicted.split(), smoothing_function=smooth)

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    rouge = scorer.score(reference, predicted)

    exact_match = int(predicted.strip().lower() == reference.strip().lower())

    return {
        "BLEU": round(bleu, 3),
        "ROUGE-1": round(rouge['rouge1'].fmeasure, 3),
        "ROUGE-L": round(rouge['rougeL'].fmeasure, 3),
        "Exact Match": exact_match
    }
